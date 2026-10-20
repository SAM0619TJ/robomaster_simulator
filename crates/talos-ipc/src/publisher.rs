use crate::layout::*;
use crate::shm::{ShmError, ShmRegion};
use crate::triple_buffer::TripleBufferProducer;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct ShmPublisher {
    meta_region: ShmRegion,
    image_pool: ShmRegion,
    current_buffer_id: u8,
    last_synchronized_frame_seq: Option<u64>,
}

impl ShmPublisher {
    pub fn create() -> Result<Self, ShmError> {
        let mut meta_region = ShmRegion::create(SHM_NAME_META, size_of::<ShmMetaRegion>())?;
        let image_pool = ShmRegion::create(SHM_NAME_IMAGE_POOL, IMAGE_POOL_SIZE)?;

        unsafe {
            let meta = meta_region.as_mut::<ShmMetaRegion>();

            // 初始化 header
            meta.header = ShmHeader {
                magic: SHM_MAGIC,
                version: SHM_VERSION,
                created_ns: Self::now_ns(),
                heartbeat_ns: Self::now_ns(),
                image_width: IMAGE_WIDTH,
                image_height: IMAGE_HEIGHT,
                _pad: [0; 32],
            };

            // 初始化所有 TripleBuffer (CRITICAL: 零填充破坏了正确的初始状态)
            // 正确初始状态: state=1 (ready slot), write_idx=0, read_idx=2
            Self::init_triple_buffer(&mut meta.image);
            for pose in &mut meta.poses {
                Self::init_triple_buffer(pose);
            }
            Self::init_triple_buffer(&mut meta.gimbal_cmd);
        }

        Ok(Self {
            meta_region,
            image_pool,
            current_buffer_id: 0,
            last_synchronized_frame_seq: None,
        })
    }

    pub fn publish_image(&mut self, data: &[u8], seq: u64, timestamp_ns: u64) {
        self.publish_image_with(data, seq, timestamp_ns, |_| {});
    }

    pub fn publish_image_with<F>(
        &mut self,
        data: &[u8],
        seq: u64,
        timestamp_ns: u64,
        before_commit: F,
    ) where
        F: FnOnce(&mut Self),
    {
        assert_eq!(data.len(), IMAGE_SIZE, "Image size mismatch");

        let buffer_id = self.current_buffer_id;
        self.current_buffer_id = (self.current_buffer_id + 1) % 3;

        unsafe {
            let pool_ptr = self.image_pool.as_ptr();
            let dst = pool_ptr.add(buffer_id as usize * IMAGE_SIZE);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, IMAGE_SIZE);
        }

        // Publish data associated with this image only after the expensive pixel copy. The image
        // metadata below is the commit marker observed by consumers.
        before_commit(self);

        unsafe {
            let meta = self.meta_region.as_mut::<ShmMetaRegion>();
            let mut producer = TripleBufferProducer::new(
                &meta.image.state,
                &mut meta.image.write_idx,
                &mut meta.image.slots,
            );

            let slot = producer.borrow_mut();
            slot.seq = seq;
            slot.timestamp_ns = timestamp_ns;
            slot.width = IMAGE_WIDTH;
            slot.height = IMAGE_HEIGHT;
            slot.buffer_id = buffer_id;
            slot.format = 0;
            producer.publish();
        }
    }

    #[must_use]
    pub fn try_publish_synchronized_image<F>(
        &mut self,
        data: &[u8],
        seq: u64,
        timestamp_ns: u64,
        before_commit: F,
    ) -> bool
    where
        F: FnOnce(&mut Self),
    {
        // Never publish an older async readback, and never overwrite one half of an image/pose
        // bundle while the consumer is between the two triple buffers.
        if self
            .last_synchronized_frame_seq
            .is_some_and(|last_seq| seq <= last_seq)
            || !self.synchronized_frame_consumed()
        {
            return false;
        }

        self.publish_image_with(data, seq, timestamp_ns, before_commit);
        self.last_synchronized_frame_seq = Some(seq);
        true
    }

    fn synchronized_frame_consumed(&self) -> bool {
        unsafe {
            let meta = self.meta_region.as_ref::<ShmMetaRegion>();
            let image_consumed = meta.image.state.load(Ordering::Acquire) & FLAG_NEW == 0;
            // Gimbal, odom, muzzle and camera are consumed with each image. Slot 4 is the legacy
            // chassis-observation channel and is intentionally not part of this handshake.
            let poses_consumed = meta.poses[..=PoseIndex::Camera as usize]
                .iter()
                .all(|pose| pose.state.load(Ordering::Acquire) & FLAG_NEW == 0);
            image_consumed && poses_consumed
        }
    }

    pub fn publish_pose(
        &mut self,
        index: PoseIndex,
        position: [f32; 3],
        quaternion: [f32; 4],
        frame_seq: u64,
        timestamp_ns: u64,
    ) {
        self.publish_pose_with_aux(
            index,
            position,
            quaternion,
            [0.0; 4],
            frame_seq,
            timestamp_ns,
        );
    }

    pub fn publish_pose_with_aux(
        &mut self,
        index: PoseIndex,
        position: [f32; 3],
        quaternion: [f32; 4],
        aux_f32: [f32; 4],
        frame_seq: u64,
        timestamp_ns: u64,
    ) {
        unsafe {
            let meta = self.meta_region.as_mut::<ShmMetaRegion>();
            let pose_buf = &mut meta.poses[index as usize];
            let mut producer = TripleBufferProducer::new(
                &pose_buf.state,
                &mut pose_buf.write_idx,
                &mut pose_buf.slots,
            );

            let slot = producer.borrow_mut();
            slot.frame_seq = frame_seq;
            slot.position = position;
            slot.quaternion = quaternion;
            slot.timestamp_ns = timestamp_ns;
            slot._pad = aux_f32_to_bytes(aux_f32);

            producer.publish();
        }
    }

    pub fn set_camera_info(&mut self, info: CameraInfo) {
        unsafe {
            let meta = self.meta_region.as_mut::<ShmMetaRegion>();
            meta.camera_info = info;
        }
    }

    pub fn publish_chassis_observation(&mut self, observation: ChassisObservation) {
        unsafe {
            let meta = self.meta_region.as_mut::<ShmMetaRegion>();
            meta.chassis_observation = observation;
        }
    }

    pub fn publish_ground_truth(&mut self, batch: &GroundTruthBatch) {
        unsafe {
            let meta = self.meta_region.as_mut::<ShmMetaRegion>();
            meta.ground_truth = *batch;
        }
    }

    pub fn publish_runtime_state(&mut self, state: RuntimeState) {
        unsafe {
            let runtime = self
                .meta_region
                .as_ptr()
                .add(std::mem::offset_of!(ShmMetaRegion, runtime_state))
                .cast::<RuntimeState>();
            store_runtime_state(runtime, state);
        }
    }

    pub fn update_heartbeat(&mut self) {
        unsafe {
            let heartbeat = self
                .meta_region
                .as_ptr()
                .add(
                    std::mem::offset_of!(ShmMetaRegion, header)
                        + std::mem::offset_of!(ShmHeader, heartbeat_ns),
                )
                .cast::<u64>();
            AtomicU64::from_ptr(heartbeat).store(Self::now_ns(), Ordering::Release);
        }
    }

    fn now_ns() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }

    /// 初始化 TripleBuffer 到正确的初始状态
    ///
    /// ShmRegion::create() 使用零填充，会破坏 TripleBuffer 的正确初始状态。
    /// 必须手动重新初始化。
    ///
    /// 正确初始状态:
    /// - state = 1 (ready slot 是 1, 无 FLAG_NEW)
    /// - write_idx = 0 (生产者写入 slot 0)
    /// - read_idx = 2 (消费者上次读取 slot 2)
    fn init_triple_buffer(buf: &mut impl TripleBufferInit) {
        buf.init_state();
    }
}

// The two optional Talos v2 diagnostics must not tear across byte carries. They remain
// independent counters, rather than a coherent snapshot of the complete runtime state.
unsafe fn store_runtime_state(runtime: *mut RuntimeState, state: RuntimeState) {
    unsafe {
        AtomicU64::from_ptr(std::ptr::addr_of_mut!((*runtime).timestamp_ns))
            .store(state.timestamp_ns, Ordering::Release);
        AtomicU8::from_ptr(std::ptr::addr_of_mut!((*runtime).following))
            .store(state.following, Ordering::Release);
        let pad = std::ptr::addr_of_mut!((*runtime)._pad).cast::<u8>();
        std::ptr::copy_nonoverlapping(state._pad.as_ptr(), pad, 3);
        std::ptr::copy_nonoverlapping(state._pad.as_ptr().add(11), pad.add(11), 44);
        for offset in [3, 7] {
            let counter = u32::from_le_bytes(state._pad[offset..offset + 4].try_into().unwrap());
            AtomicU32::from_ptr(pad.add(offset).cast::<u32>())
                .store(counter.to_le(), Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triple_buffer::TripleBufferConsumer;

    #[test]
    fn synchronized_image_waits_for_all_four_poses_and_rejects_late_readbacks() {
        // Use private names so tests cannot replace a running simulator's mappings.
        let name = format!(
            "talos_bundle_test_{}_{}",
            std::process::id(),
            ShmPublisher::now_ns()
        );
        let meta_name = format!("{name}_meta");
        let pool_name = format!("{name}_pool");
        let mut meta_region = ShmRegion::create(&meta_name, size_of::<ShmMetaRegion>()).unwrap();
        unsafe { *meta_region.as_mut::<ShmMetaRegion>() = ShmMetaRegion::default() };
        let mut publisher = ShmPublisher {
            meta_region,
            image_pool: ShmRegion::create(&pool_name, IMAGE_POOL_SIZE).unwrap(),
            current_buffer_id: 0,
            last_synchronized_frame_seq: None,
        };
        let publish_poses = |publisher: &mut ShmPublisher, seq, timestamp| {
            for index in [
                PoseIndex::Gimbal,
                PoseIndex::Odom,
                PoseIndex::Muzzle,
                PoseIndex::Camera,
            ] {
                publisher.publish_pose(index, [0.0; 3], [1.0, 0.0, 0.0, 0.0], seq, timestamp);
            }
        };
        let pixels = vec![64u8; IMAGE_SIZE];
        assert!(
            publisher.try_publish_synchronized_image(&pixels, 10, 100, |publisher| {
                publish_poses(publisher, 10, 100);
            })
        );
        assert!(
            !publisher.try_publish_synchronized_image(&pixels, 11, 110, |_| {
                panic!("a pending frame must not overwrite its pose bundle");
            })
        );
        unsafe {
            let meta = publisher.meta_region.as_mut::<ShmMetaRegion>();
            let image = &mut meta.image;
            let image = *TripleBufferConsumer::new(&image.state, &mut image.read_idx, &image.slots)
                .borrow()
                .unwrap();
            assert_eq!((image.seq, image.timestamp_ns), (10, 100));
            assert_eq!(
                publisher
                    .image_pool
                    .as_ptr()
                    .add(image.buffer_id as usize * IMAGE_SIZE)
                    .read(),
                64
            );
            for pose in &mut meta.poses[..3] {
                let pose = *TripleBufferConsumer::new(&pose.state, &mut pose.read_idx, &pose.slots)
                    .borrow()
                    .unwrap();
                assert_eq!((pose.frame_seq, pose.timestamp_ns), (10, 100));
            }
        }
        assert!(
            !publisher.try_publish_synchronized_image(&pixels, 11, 110, |_| {
                panic!("image consumed alone must not release the final pending pose");
            })
        );
        unsafe {
            let meta = publisher.meta_region.as_mut::<ShmMetaRegion>();
            let camera = &mut meta.poses[PoseIndex::Camera as usize];
            let pose =
                *TripleBufferConsumer::new(&camera.state, &mut camera.read_idx, &camera.slots)
                    .borrow()
                    .unwrap();
            assert_eq!((pose.frame_seq, pose.timestamp_ns), (10, 100));
        }
        // Slot 4 remains unread deliberately: it is a legacy auxiliary channel.
        assert!(
            publisher.try_publish_synchronized_image(&pixels, 12, 120, |publisher| {
                publish_poses(publisher, 12, 120);
            })
        );
        unsafe {
            let meta = publisher.meta_region.as_mut::<ShmMetaRegion>();
            let image = &mut meta.image;
            assert!(
                TripleBufferConsumer::new(&image.state, &mut image.read_idx, &image.slots)
                    .borrow()
                    .is_some()
            );
            for pose in &mut meta.poses[..4] {
                assert!(
                    TripleBufferConsumer::new(&pose.state, &mut pose.read_idx, &pose.slots)
                        .borrow()
                        .is_some()
                );
            }
        }
        assert!(
            !publisher.try_publish_synchronized_image(&pixels, 11, 110, |_| {
                panic!("late GPU completion must never replace a newer frame");
            })
        );
        drop(publisher);
        for name in [meta_name, pool_name] {
            let _ = std::fs::remove_file(format!("/tmp/.{name}.lock"));
        }
    }

    #[test]
    fn diagnostic_counters_are_aligned_and_keep_v2_bytes() {
        assert_eq!(std::mem::offset_of!(RuntimeState, _pad), 9);
        assert_eq!(std::mem::align_of::<RuntimeState>(), 64);
        let mut output = RuntimeState::default();
        for count in [255u32, 256, 65535, 65536, u32::MAX] {
            let mut state = RuntimeState {
                timestamp_ns: 123,
                following: 1,
                _pad: [17; 55],
            };
            state._pad[3..7].copy_from_slice(&count.to_le_bytes());
            state._pad[7..11].copy_from_slice(&(count / 2).to_le_bytes());
            unsafe {
                store_runtime_state(&mut output, state);
            }
            let base = std::ptr::addr_of_mut!(output).cast::<u8>();
            for (offset, expected) in [(12, count), (16, count / 2)] {
                let address = unsafe { base.add(offset) }.cast::<u32>();
                assert_eq!((address as usize) % std::mem::align_of::<AtomicU32>(), 0);
                let actual = unsafe { AtomicU32::from_ptr(address) }.load(Ordering::Acquire);
                assert_eq!(u32::from_le(actual), expected);
            }
            assert_eq!(output.timestamp_ns, 123);
            assert_eq!(output.following, 1);
            assert_eq!(&output._pad[..3], &[17; 3]);
            assert_eq!(&output._pad[11..], &[17; 44]);
        }
    }
}

fn aux_f32_to_bytes(aux_f32: [f32; 4]) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    for (i, value) in aux_f32.iter().enumerate() {
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Trait for initializing triple buffer state
trait TripleBufferInit {
    fn init_state(&mut self);
}

impl TripleBufferInit for ImageTripleBuffer {
    fn init_state(&mut self) {
        self.state.store(1, Ordering::Relaxed);
        self.write_idx = 0;
        self.read_idx = 2;
    }
}

impl TripleBufferInit for PoseTripleBuffer {
    fn init_state(&mut self) {
        self.state.store(1, Ordering::Relaxed);
        self.write_idx = 0;
        self.read_idx = 2;
    }
}

impl TripleBufferInit for GimbalTripleBuffer {
    fn init_state(&mut self) {
        self.state.store(1, Ordering::Relaxed);
        self.write_idx = 0;
        self.read_idx = 2;
    }
}
