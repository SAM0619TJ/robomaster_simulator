use crate::capture::{
    CameraFov, CaptureBundle, CaptureCamera, ImageHandle, compute_camera_intrinsics,
    driver::{
        CaptureConfig, CaptureFrameId, CapturedFrame, CapturedFrameKind, GpuCaptureHandler,
        SnapshotAsync, SnapshotSync,
    },
    setup_capture_camera, setup_preview_window, sync_capture_camera,
};
use crate::components::{Controlled, InfantryGimbal, InfantryLaunchOffset, SubscribeAutoAim};
use crate::statistic::ProjectileStatistics;
use crate::systems::{ChassisObservationFrame, GameplaySystems};
use crate::talos::plugin::{to_ros_quat, to_ros_translation};
use bevy::ecs::world::DeferredWorld;
use bevy::prelude::*;
use bevy::render::{Extract, ExtractSchedule, RenderApp, RenderSystems};
use std::f32::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use talos_ipc::*;

static FRAME_SEQ: AtomicU64 = AtomicU64::new(0);

#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct TalosFrameStamp {
    pub frame_seq: u64,
    pub timestamp_ns: u64,
}

pub fn advance_talos_frame_stamp(mut stamp: ResMut<TalosFrameStamp>) {
    stamp.frame_seq = FRAME_SEQ.fetch_add(1, Ordering::Relaxed);
    stamp.timestamp_ns = now_ns();
}

/// Extracted pose data from MainApp to RenderApp for synchronized publishing
#[derive(Resource, Clone, Default)]
pub struct ExtractedPoseData {
    pub frame_seq: u64,
    pub timestamp_ns: u64,
    pose: Option<CapturedPoseData>,
    pub valid: bool,
}

/// Pose data captured at frame snapshot time
#[derive(Clone)]
struct CapturedPoseData {
    gimbal_ros: [f32; 3],
    gimbal_quat: [f32; 4],
    muzzle_rel: [f32; 3],
    camera_rel: [f32; 3],
    camera_quat: [f32; 4],
    muzzle_quat: [f32; 4],
    chassis_observation: ChassisObservation,
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

struct TalosSnapshotSync {
    frame_seq: u64,
    timestamp_ns: u64,
    pose: CapturedPoseData,
}

impl SnapshotSync for TalosSnapshotSync {
    fn captured(
        self: Box<Self>,
        world: &mut DeferredWorld,
        _config: &CaptureConfig,
    ) -> Box<dyn SnapshotAsync> {
        let ctx = world.resource::<TalosCaptureContextShared>().0.clone();

        Box::new(TalosSnapshot {
            ctx,
            frame_seq: self.frame_seq,
            timestamp_ns: self.timestamp_ns,
            pose: self.pose,
        })
    }
}

struct TalosSnapshot {
    ctx: Arc<Mutex<ShmPublisher>>,
    frame_seq: u64,
    timestamp_ns: u64,
    pose: CapturedPoseData,
}

impl SnapshotAsync for TalosSnapshot {
    fn captured(&mut self, frame: CapturedFrame<'_>) {
        if frame.kind != CapturedFrameKind::Rgb8 {
            return;
        }

        let expected_size = (frame.width * frame.height * 3) as usize;
        if frame.data.len() != expected_size {
            warn!(
                "图像大小不匹配: expected {} bytes, got {} bytes",
                expected_size,
                frame.data.len()
            );
            return;
        }

        if frame.width != IMAGE_WIDTH || frame.height != IMAGE_HEIGHT {
            warn!(
                "image reesolution mismatched: expected {}x{}, got {}x{}",
                IMAGE_WIDTH, IMAGE_HEIGHT, frame.width, frame.height
            );
            return;
        }

        if let Ok(mut publisher) = self.ctx.lock() {
            let _ = publisher.try_publish_synchronized_image(
                frame.data,
                self.frame_seq,
                self.timestamp_ns,
                |publisher| {
                    publish_pose_data(publisher, self.frame_seq, self.timestamp_ns, &self.pose);
                },
            );
        }
    }
}

#[derive(Default)]
struct TalosSnapshotCreator {}

impl GpuCaptureHandler for TalosSnapshotCreator {
    fn captured(
        &self,
        world: &World,
        _frame_id: Option<CaptureFrameId>,
    ) -> Option<Box<dyn SnapshotSync>> {
        // Timestamp, frame sequence and pose must come from the same ExtractSchedule snapshot.
        let extracted = world.get_resource::<ExtractedPoseData>()?;
        if !extracted.valid {
            return None;
        }
        let pose = extracted.pose.clone()?;

        Some(Box::new(TalosSnapshotSync {
            frame_seq: extracted.frame_seq,
            timestamp_ns: extracted.timestamp_ns,
            pose,
        }))
    }
}

#[derive(Resource, Clone, Deref, DerefMut)]
pub struct TalosCaptureContextShared(pub Arc<Mutex<ShmPublisher>>);

#[derive(Resource, Clone)]
pub struct TalosCaptureContext {
    pub publisher: Arc<Mutex<ShmPublisher>>,
    pub fov_y: f32,
}

pub struct TalosCapturePlugin {
    pub config: CaptureConfig,
    pub context: TalosCaptureContext,
}

pub fn publish_talos_runtime_state_system(
    context: Option<Res<TalosCaptureContext>>,
    frame_stamp: Res<TalosFrameStamp>,
    following: Res<SubscribeAutoAim>,
    statistics: Res<ProjectileStatistics>,
) {
    let Some(ctx) = context else {
        return;
    };

    if let Ok(mut publisher) = ctx.publisher.lock() {
        // Talos v2 extension: naturally aligned little-endian diagnostic counters.
        let mut diagnostic = [0; 55];
        diagnostic[3..7].copy_from_slice(&statistics.launch_count.to_le_bytes());
        diagnostic[7..11].copy_from_slice(&statistics.accurate_count.to_le_bytes());
        publisher.publish_runtime_state(RuntimeState {
            timestamp_ns: frame_stamp.timestamp_ns,
            following: u8::from(following.load(Ordering::Acquire)),
            _pad: diagnostic,
        });
    }
}

impl Plugin for TalosCapturePlugin {
    fn build(&self, app: &mut App) {
        let capture = CaptureBundle::color(
            app,
            self.config.clone(),
            vec![Box::new(TalosSnapshotCreator::default())],
        );
        let render_target_handle = capture.color_target().unwrap().clone();

        {
            let mut publisher = self.context.publisher.lock().unwrap();
            let intrinsics = compute_camera_intrinsics(
                self.config.width,
                self.config.height,
                self.context.fov_y,
            );

            publisher.set_camera_info(CameraInfo {
                timestamp_ns: now_ns(),
                fx: intrinsics.fx,
                fy: intrinsics.fy,
                cx: intrinsics.cx,
                cy: intrinsics.cy,
                distortion: [0.0; 5],
                width: intrinsics.width,
                height: intrinsics.height,
                _pad: [0; 24],
            });
        }

        app.add_plugins(capture)
            .insert_resource(ImageHandle(render_target_handle))
            .insert_resource(CameraFov(self.context.fov_y))
            .insert_resource(self.context.clone())
            .add_systems(Startup, setup_capture_camera)
            .add_systems(Startup, setup_preview_window)
            .add_systems(
                Update,
                sync_capture_camera
                    .after(GameplaySystems::Camera)
                    .before(RenderSystems::Render),
            );

        app.sub_app_mut(RenderApp)
            .insert_resource(TalosCaptureContextShared(self.context.publisher.clone()))
            .insert_resource(self.context.clone())
            .insert_resource(ExtractedPoseData::default())
            .add_systems(ExtractSchedule, extract_pose_data);
    }
}

/// Extract pose data from MainApp to RenderApp
fn extract_pose_data(
    mut pose_data: ResMut<ExtractedPoseData>,
    frame_stamp: Extract<Res<TalosFrameStamp>>,
    camera: Extract<Query<&GlobalTransform, With<CaptureCamera>>>,
    gimbal: Extract<Query<&GlobalTransform, (With<Controlled>, With<InfantryGimbal>)>>,
    muzzle_offset: Extract<Query<&GlobalTransform, (With<InfantryLaunchOffset>, With<Controlled>)>>,
    chassis_obs: Extract<Res<ChassisObservationFrame>>,
) {
    pose_data.frame_seq = frame_stamp.frame_seq;
    pose_data.timestamp_ns = frame_stamp.timestamp_ns;

    let Ok(cam_transform) = camera.single() else {
        pose_data.pose = None;
        pose_data.valid = false;
        return;
    };
    let Ok(gimbal_transform) = gimbal.single() else {
        pose_data.pose = None;
        pose_data.valid = false;
        return;
    };
    let Ok(muzzle_global) = muzzle_offset.single() else {
        pose_data.pose = None;
        pose_data.valid = false;
        return;
    };

    pose_data.pose = Some(captured_pose_data(
        cam_transform,
        gimbal_transform,
        muzzle_global,
        &chassis_obs,
        pose_data.frame_seq,
        pose_data.timestamp_ns,
    ));
    pose_data.valid = true;
}

fn captured_pose_data(
    cam_transform: &GlobalTransform,
    gimbal_transform: &GlobalTransform,
    muzzle_global: &GlobalTransform,
    chassis_obs: &ChassisObservationFrame,
    frame_seq: u64,
    timestamp_ns: u64,
) -> CapturedPoseData {
    // The gun model launches along local +Y. This aiming basis points -Z along
    // that direction, which becomes +X after the Bevy-to-ROS basis conversion.
    // Use the actual world muzzle rotation so intervening scene parents are included.
    let gimbal_rot = muzzle_global.rotation() * Quat::from_rotation_x(PI / 2.0);
    let reference = GlobalTransform::from(
        Transform::from_translation(gimbal_transform.translation()).with_rotation(gimbal_rot),
    );
    let cam_rel = cam_transform.reparented_to(&reference);
    let muzzle_rel = muzzle_global.reparented_to(&reference);
    let camera_quat = to_ros_quat(cam_rel.rotation);
    let muzzle_quat = to_ros_quat(muzzle_rel.rotation);

    let gimbal_ros = to_ros_translation(gimbal_transform.translation());
    let gimbal_rot = to_ros_quat(gimbal_rot);
    let muzzle = to_ros_translation(muzzle_rel.translation);
    let camera = to_ros_translation(cam_rel.translation);

    CapturedPoseData {
        gimbal_ros: [gimbal_ros.x, gimbal_ros.y, gimbal_ros.z],
        gimbal_quat: [gimbal_rot.w, gimbal_rot.x, gimbal_rot.y, gimbal_rot.z],
        muzzle_rel: [muzzle.x, muzzle.y, muzzle.z],
        camera_rel: [camera.x, camera.y, camera.z],
        camera_quat: [camera_quat.w, camera_quat.x, camera_quat.y, camera_quat.z],
        muzzle_quat: [muzzle_quat.w, muzzle_quat.x, muzzle_quat.y, muzzle_quat.z],
        chassis_observation: ChassisObservation {
            frame_seq,
            timestamp_ns,
            dt_s: chassis_obs.dt_s,
            v_body: [chassis_obs.v_body.x, chassis_obs.v_body.y],
            wz_radps: chassis_obs.wz_radps,
            wheel_linear_mps: chassis_obs.wheel_linear_mps,
            wheel_angular_radps: chassis_obs.wheel_angular_radps,
            a_body: [chassis_obs.a_body.x, chassis_obs.a_body.y],
            alpha_z_radps2: chassis_obs.alpha_z_radps2,
            rpy_rad: [
                chassis_obs.rpy_rad.x,
                chassis_obs.rpy_rad.y,
                chassis_obs.rpy_rad.z,
            ],
            gyro_xyz_radps: [
                chassis_obs.gyro_xyz_radps.x,
                chassis_obs.gyro_xyz_radps.y,
                chassis_obs.gyro_xyz_radps.z,
            ],
            accel_xyz_mps2: [
                chassis_obs.accel_xyz_mps2.x,
                chassis_obs.accel_xyz_mps2.y,
                chassis_obs.accel_xyz_mps2.z,
            ],
            _pad: [0; 16],
        },
    }
}

fn publish_pose_data(
    publisher: &mut ShmPublisher,
    frame_seq: u64,
    timestamp_ns: u64,
    pose: &CapturedPoseData,
) {
    publisher.publish_pose(
        PoseIndex::Odom,
        pose.gimbal_ros,
        [1.0, 0.0, 0.0, 0.0],
        frame_seq,
        timestamp_ns,
    );

    publisher.publish_pose(
        PoseIndex::Gimbal,
        [0.0, 0.0, 0.0],
        pose.gimbal_quat,
        frame_seq,
        timestamp_ns,
    );

    publisher.publish_pose(
        PoseIndex::Muzzle,
        pose.muzzle_rel,
        pose.muzzle_quat,
        frame_seq,
        timestamp_ns,
    );

    publisher.publish_pose(
        PoseIndex::Camera,
        pose.camera_rel,
        pose.camera_quat,
        frame_seq,
        timestamp_ns,
    );

    let mut observation = pose.chassis_observation;
    observation.frame_seq = frame_seq;
    observation.timestamp_ns = timestamp_ns;
    publisher.publish_chassis_observation(observation);

    // Legacy compatibility path for consumers still reading pose slot 4.
    publisher.publish_pose_with_aux(
        PoseIndex::ChassisObservation,
        [
            observation.v_body[0],
            observation.v_body[1],
            observation.wz_radps,
        ],
        observation.wheel_angular_radps,
        [
            observation.a_body[0],
            observation.a_body[1],
            observation.alpha_z_radps2,
            observation.dt_s,
        ],
        frame_seq,
        timestamp_ns,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quaternion(wxyz: [f32; 4]) -> Quat {
        Quat::from_xyzw(wxyz[1], wxyz[2], wxyz[3], wxyz[0])
    }

    fn assert_vec_close(actual: Vec3, expected: Vec3) {
        assert!(
            (actual - expected).length() < 2e-5,
            "{actual:?} != {expected:?}"
        );
    }

    fn assert_quat_close(actual: Quat, expected: Quat) {
        assert!(
            actual.dot(expected).abs() > 1.0 - 2e-6,
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn pose_chain_reconstructs_actual_camera_and_muzzle_world_poses() {
        let gimbal =
            GlobalTransform::from(
                Transform::from_translation(Vec3::new(3.2, 0.8, -6.7))
                    .with_rotation(Quat::from_euler(EulerRot::YXZ, 0.8, -0.3, 0.1)),
            );
        // Include an intermediate scene parent: local muzzle rotation alone is
        // insufficient to reconstruct the true world gun axis.
        let muzzle = gimbal
            .mul_transform(Transform::from_rotation(Quat::from_rotation_z(0.12)))
            .mul_transform(
                Transform::from_translation(Vec3::new(0.04, 0.2, -0.15))
                    .with_rotation(Quat::from_euler(EulerRot::XYZ, -1.3, 0.07, -0.05)),
            );
        let aiming_rotation = muzzle.rotation() * Quat::from_rotation_x(PI / 2.0);
        let camera = GlobalTransform::from(
            Transform::from_translation(
                gimbal.translation() + aiming_rotation * Vec3::new(0.08, 0.11, -0.03),
            )
            .with_rotation(aiming_rotation * Quat::from_euler(EulerRot::XYZ, 0.02, -0.04, 0.01)),
        );
        let pose = captured_pose_data(
            &camera,
            &gimbal,
            &muzzle,
            &ChassisObservationFrame::default(),
            1234,
            987654321,
        );
        let q_world_gimbal = quaternion(pose.gimbal_quat);
        let p_world_gimbal = Vec3::from_array(pose.gimbal_ros);
        assert_quat_close(q_world_gimbal, to_ros_quat(aiming_rotation));
        assert_quat_close(
            q_world_gimbal * quaternion(pose.camera_quat),
            to_ros_quat(camera.rotation()),
        );
        assert_quat_close(
            q_world_gimbal * quaternion(pose.muzzle_quat),
            to_ros_quat(muzzle.rotation()),
        );
        assert_vec_close(
            p_world_gimbal + q_world_gimbal * Vec3::from_array(pose.camera_rel),
            to_ros_translation(camera.translation()),
        );
        assert_vec_close(
            p_world_gimbal + q_world_gimbal * Vec3::from_array(pose.muzzle_rel),
            to_ros_translation(muzzle.translation()),
        );
        assert_vec_close(
            q_world_gimbal * Vec3::X,
            to_ros_translation(muzzle.rotation() * Vec3::Y),
        );
        assert_vec_close(
            q_world_gimbal * quaternion(pose.camera_quat) * Vec3::X,
            to_ros_translation(camera.rotation() * Vec3::NEG_Z),
        );
        assert_eq!(pose.chassis_observation.frame_seq, 1234);
        assert_eq!(pose.chassis_observation.timestamp_ns, 987654321);
    }

    #[test]
    fn aligned_camera_has_identity_relative_orientation() {
        let gimbal = GlobalTransform::from(
            Transform::from_translation(Vec3::new(-2.0, 1.5, 4.0))
                .with_rotation(Quat::from_rotation_y(-0.6)),
        );
        let muzzle =
            gimbal.mul_transform(Transform::from_rotation(Quat::from_rotation_x(-PI / 2.0)));
        let pose = captured_pose_data(
            &gimbal,
            &gimbal,
            &muzzle,
            &ChassisObservationFrame::default(),
            0,
            0,
        );
        assert_quat_close(quaternion(pose.camera_quat), Quat::IDENTITY);
        assert_quat_close(quaternion(pose.gimbal_quat), to_ros_quat(gimbal.rotation()));
    }
}
