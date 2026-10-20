use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::schedule::camera_driver;
use bevy::ecs::world::DeferredWorld;
use bevy::render::texture::GpuImage;
use bevy::tasks::AsyncComputeTaskPool;
use bevy::{
    image::TextureFormatPixelInfo,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        render_asset::RenderAssets,
        render_resource::{
            Buffer, BufferDescriptor, BufferUsages, Extent3d, MapMode, Origin3d,
            TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
            TextureFormat, TextureUsages,
        },
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems},
    },
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CaptureFrameId(u64);

impl CaptureFrameId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CaptureChannels(u8);

impl CaptureChannels {
    pub const RGB: Self = Self(1 << 0);
    pub const DEPTH: Self = Self(1 << 1);
    pub const R32_UINT: Self = Self(1 << 2);
    pub const ALL: Self = Self(Self::RGB.0 | Self::DEPTH.0 | Self::R32_UINT.0);

    const fn for_kind(kind: CapturedFrameKind) -> Self {
        match kind {
            CapturedFrameKind::Rgb8 => Self::RGB,
            CapturedFrameKind::Depth32F => Self::DEPTH,
            CapturedFrameKind::R32Uint => Self::R32_UINT,
        }
    }

    const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CaptureSubmission {
    id: CaptureFrameId,
    required: CaptureChannels,
    submitted: CaptureChannels,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureSubmissionError {
    AlreadyActive {
        active: CaptureFrameId,
        requested: CaptureFrameId,
    },
    NoActiveSubmission,
    WrongFrame {
        active: CaptureFrameId,
        received: CaptureFrameId,
    },
    UnexpectedChannel {
        id: CaptureFrameId,
        kind: CapturedFrameKind,
    },
    DuplicateChannel {
        id: CaptureFrameId,
        kind: CapturedFrameKind,
    },
    MissingChannels {
        id: CaptureFrameId,
        required: CaptureChannels,
        submitted: CaptureChannels,
    },
}

#[derive(Resource, Default)]
pub struct CaptureFrameSubmission(Mutex<Option<CaptureSubmission>>);

impl CaptureFrameSubmission {
    pub fn begin(
        &self,
        id: CaptureFrameId,
        required: CaptureChannels,
    ) -> Result<(), CaptureSubmissionError> {
        let mut guard = self.0.lock().unwrap();
        if let Some(active) = *guard {
            return Err(CaptureSubmissionError::AlreadyActive {
                active: active.id,
                requested: id,
            });
        }
        *guard = Some(CaptureSubmission {
            id,
            required,
            submitted: CaptureChannels::default(),
        });
        Ok(())
    }

    pub fn active_id(&self) -> Option<CaptureFrameId> {
        self.0.lock().unwrap().as_ref().map(|active| active.id)
    }

    fn claim(
        &self,
        id: CaptureFrameId,
        kind: CapturedFrameKind,
    ) -> Result<(), CaptureSubmissionError> {
        let mut guard = self.0.lock().unwrap();
        let active = guard
            .as_mut()
            .ok_or(CaptureSubmissionError::NoActiveSubmission)?;
        if active.id != id {
            return Err(CaptureSubmissionError::WrongFrame {
                active: active.id,
                received: id,
            });
        }
        let channel = CaptureChannels::for_kind(kind);
        if !active.required.contains(channel) {
            return Err(CaptureSubmissionError::UnexpectedChannel { id, kind });
        }
        if active.submitted.contains(channel) {
            return Err(CaptureSubmissionError::DuplicateChannel { id, kind });
        }
        active.submitted.insert(channel);
        Ok(())
    }

    fn finish(&self, id: CaptureFrameId) -> Result<(), CaptureSubmissionError> {
        let active = self
            .0
            .lock()
            .unwrap()
            .take()
            .ok_or(CaptureSubmissionError::NoActiveSubmission)?;
        if active.id != id {
            return Err(CaptureSubmissionError::WrongFrame {
                active: active.id,
                received: id,
            });
        }
        if !active.submitted.contains(active.required) {
            return Err(CaptureSubmissionError::MissingChannels {
                id,
                required: active.required,
                submitted: active.submitted,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapturedFrameKind {
    Rgb8,
    Depth32F,
    R32Uint,
}

#[derive(Resource, Clone)]
pub struct CaptureConfig {
    pub width: u32,
    pub height: u32,
    pub texture_format: TextureFormat,
    pub frame_kind: CapturedFrameKind,
}

pub struct CapturedFrame<'a> {
    pub frame_id: Option<CaptureFrameId>,
    pub kind: CapturedFrameKind,
    pub width: u32,
    pub height: u32,
    pub data: &'a [u8],
}

pub fn create_capture_image_handle(
    app: &mut App,
    width: u32,
    height: u32,
    texture_format: TextureFormat,
    asset_usages: RenderAssetUsages,
    texture_usages: TextureUsages,
) -> Handle<Image> {
    let extent = Extent3d {
        width,
        height,
        ..Default::default()
    };

    let mut image = if matches!(
        texture_format,
        TextureFormat::Depth16Unorm
            | TextureFormat::Depth24Plus
            | TextureFormat::Depth24PlusStencil8
            | TextureFormat::Depth32Float
            | TextureFormat::Depth32FloatStencil8
    ) {
        Image::new_uninit(
            extent,
            bevy::render::render_resource::TextureDimension::D2,
            texture_format,
            asset_usages,
        )
    } else {
        Image::new_target_texture(width, height, texture_format, Some(texture_format))
    };

    image.texture_descriptor.usage |= texture_usages;
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    images.add(image)
}

enum CapturePluginNode {
    Camera(CameraCapturePlugin),
    ViewCopy(crate::capture::view_copy::ViewTextureCopyPlugin),
}

pub struct CaptureBundle {
    plugins: Vec<CapturePluginNode>,
    color_target: Option<Handle<Image>>,
    depth_target: Option<Handle<Image>>,
}

impl CaptureBundle {
    pub fn color(
        app: &mut App,
        config: CaptureConfig,
        snapshots: Vec<Box<dyn GpuCaptureHandler>>,
    ) -> Self {
        let (plugin, color_target) = CameraCapturePlugin::new(app, config, snapshots);
        Self {
            plugins: vec![CapturePluginNode::Camera(plugin)],
            color_target: Some(color_target),
            depth_target: None,
        }
    }

    pub fn color_and_depth(
        app: &mut App,
        color_config: CaptureConfig,
        color_snapshots: Vec<Box<dyn GpuCaptureHandler>>,
        depth_snapshots: Vec<Box<dyn GpuCaptureHandler>>,
    ) -> Self {
        Self::color(app, color_config.clone(), color_snapshots).with_depth_from_camera_order(
            app,
            CaptureConfig {
                width: color_config.width,
                height: color_config.height,
                texture_format: TextureFormat::Depth32Float,
                frame_kind: CapturedFrameKind::Depth32F,
            },
            crate::capture::CAPTURE_CAMERA_ORDER,
            depth_snapshots,
        )
    }

    pub fn depth_from_camera_order(
        app: &mut App,
        config: CaptureConfig,
        camera_order: isize,
        snapshots: Vec<Box<dyn GpuCaptureHandler>>,
    ) -> Self {
        let mut bundle = Self {
            plugins: Vec::new(),
            color_target: None,
            depth_target: None,
        };
        bundle.push_depth_from_camera_order(app, config, camera_order, snapshots);
        bundle
    }

    pub fn with_depth_from_camera_order(
        mut self,
        app: &mut App,
        config: CaptureConfig,
        camera_order: isize,
        snapshots: Vec<Box<dyn GpuCaptureHandler>>,
    ) -> Self {
        self.push_depth_from_camera_order(app, config, camera_order, snapshots);
        self
    }

    pub fn color_target(&self) -> Option<&Handle<Image>> {
        self.color_target.as_ref()
    }

    pub fn depth_target(&self) -> Option<&Handle<Image>> {
        self.depth_target.as_ref()
    }

    fn push_depth_from_camera_order(
        &mut self,
        app: &mut App,
        config: CaptureConfig,
        camera_order: isize,
        snapshots: Vec<Box<dyn GpuCaptureHandler>>,
    ) {
        let (view_copy, depth_target) =
            crate::capture::view_copy::ViewTextureCopyPlugin::new_depth_for_camera_order(
                app,
                config.width,
                config.height,
                camera_order,
            );
        let depth_capture =
            CameraCapturePlugin::from_existing_handle(config, depth_target.clone(), snapshots);

        self.plugins.push(CapturePluginNode::ViewCopy(view_copy));
        self.plugins.push(CapturePluginNode::Camera(depth_capture));
        self.depth_target = Some(depth_target);
    }
}

impl Plugin for CaptureBundle {
    fn is_unique(&self) -> bool {
        false
    }

    fn build(&self, app: &mut App) {
        for plugin in &self.plugins {
            match plugin {
                CapturePluginNode::Camera(plugin) => plugin.build(app),
                CapturePluginNode::ViewCopy(plugin) => plugin.build(app),
            }
        }
    }
}

type ToSyncSnapshot = Box<dyn GpuCaptureHandler>;
type DynSnapshotSync = Box<dyn SnapshotSync>;

#[derive(Resource, Default, Deref, DerefMut)]
struct ImageCopiers(Vec<ImageCopier>);

#[derive(Resource, Default)]
struct ImageCopyDriverInstalled(bool);

// Realtime consumers prefer a recent frame over a growing queue of stale work.
// Explicit dataset submissions must still deliver every requested channel.
const MAX_REALTIME_READBACKS: usize = 3;

struct ReadbackPermit(Arc<AtomicUsize>);

impl ReadbackPermit {
    fn acquire(count: &Arc<AtomicUsize>, realtime: bool) -> Option<Self> {
        count
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                if realtime && count >= MAX_REALTIME_READBACKS {
                    None
                } else {
                    Some(count + 1)
                }
            })
            .ok()?;
        Some(Self(count.clone()))
    }
}

impl Drop for ReadbackPermit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

struct ImageCopier {
    config: CaptureConfig,
    src_image: Handle<Image>,
    queue: Mutex<
        VecDeque<(
            Buffer,
            Vec<DynSnapshotSync>,
            u32,
            u32,
            TextureFormat,
            ReadbackPermit,
        )>,
    >,
    free_buffers: Arc<Mutex<Vec<Buffer>>>,
    /// Pool of frame-sized output buffers, so the per-frame conversion allocates nothing.
    free_frames: Arc<Mutex<Vec<Vec<u8>>>>,
    snapshots: Arc<Vec<ToSyncSnapshot>>,
    in_flight: Arc<AtomicUsize>,
}

impl ImageCopier {
    pub fn new(
        config: CaptureConfig,
        src_image: Handle<Image>,
        snapshots: Arc<Vec<ToSyncSnapshot>>,
    ) -> ImageCopier {
        ImageCopier {
            config,
            src_image,
            queue: Mutex::new(VecDeque::new()),
            free_buffers: Arc::new(Mutex::new(Vec::new())),
            free_frames: Arc::new(Mutex::new(Vec::new())),
            snapshots,
            in_flight: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn acquire_buffer(&self, render_device: &RenderDevice, size: u64) -> Buffer {
        if let Some(buf) = self.free_buffers.lock().unwrap().pop() {
            return buf;
        }
        render_device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

/// Byte order of a 4-byte color texel as it sits in the readback buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColorOrder {
    Rgba,
    Bgra,
}

/// How mapped readback bytes become the bytes handlers receive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameEncoding {
    /// 4-byte color texels packed down to tight RGB triples.
    Rgb8 { source: ColorOrder },
    /// Rows passed through verbatim, minus the row padding wgpu requires.
    Raw,
}

/// Everything needed to turn one mapped readback into a frame, resolved once per frame instead of
/// per pixel. Constructing one is also the check that a texture format can produce the requested
/// frame kind at all, which [`CameraCapturePlugin`] runs at startup.
#[derive(Clone, Copy, Debug)]
struct FrameLayout {
    encoding: FrameEncoding,
    /// Meaningful bytes per source row, ignoring copy alignment padding.
    row_bytes: usize,
    padded_row_bytes: usize,
    output_row_bytes: usize,
    height: u32,
}

impl FrameLayout {
    fn new(
        kind: CapturedFrameKind,
        format: TextureFormat,
        width: u32,
        height: u32,
    ) -> Option<Self> {
        let pixel_size = format.pixel_size().ok()?;
        let row_bytes = width as usize * pixel_size;
        let padded_row_bytes = RenderDevice::align_copy_bytes_per_row(row_bytes);

        let (encoding, output_row_bytes) = match kind {
            CapturedFrameKind::Rgb8 => {
                let source = match format {
                    TextureFormat::Rgba8UnormSrgb | TextureFormat::Rgba8Unorm => ColorOrder::Rgba,
                    TextureFormat::Bgra8UnormSrgb | TextureFormat::Bgra8Unorm => ColorOrder::Bgra,
                    _ => return None,
                };
                (FrameEncoding::Rgb8 { source }, width as usize * 3)
            }
            CapturedFrameKind::Depth32F | CapturedFrameKind::R32Uint => {
                (FrameEncoding::Raw, row_bytes)
            }
        };

        Some(Self {
            encoding,
            row_bytes,
            padded_row_bytes,
            output_row_bytes,
            height,
        })
    }

    fn output_len(&self) -> usize {
        self.output_row_bytes * self.height as usize
    }

    /// Writes straight from the mapped range into `out`, which must be [`Self::output_len`] long.
    /// The encoding is matched per row, never per pixel.
    fn write(&self, mapped: &[u8], out: &mut [u8]) {
        let rows = mapped
            .chunks(self.padded_row_bytes)
            .take(self.height as usize);

        for (row, out_row) in rows.zip(out.chunks_exact_mut(self.output_row_bytes)) {
            let row = &row[..self.row_bytes.min(row.len())];
            match self.encoding {
                FrameEncoding::Raw => {
                    let len = row.len().min(out_row.len());
                    out_row[..len].copy_from_slice(&row[..len]);
                }
                FrameEncoding::Rgb8 {
                    source: ColorOrder::Rgba,
                } => {
                    for (texel, pixel) in row.chunks_exact(4).zip(out_row.chunks_exact_mut(3)) {
                        pixel.copy_from_slice(&texel[..3]);
                    }
                }
                FrameEncoding::Rgb8 {
                    source: ColorOrder::Bgra,
                } => {
                    for (texel, pixel) in row.chunks_exact(4).zip(out_row.chunks_exact_mut(3)) {
                        pixel.copy_from_slice(&[texel[2], texel[1], texel[0]]);
                    }
                }
            }
        }
    }
}

/// Takes a pooled output buffer, sized without ever re-zeroing a buffer that already fits.
fn acquire_frame_buffer(pool: &Mutex<Vec<Vec<u8>>>, len: usize) -> Vec<u8> {
    let mut buffer = pool.lock().unwrap().pop().unwrap_or_default();
    if buffer.len() < len {
        buffer.resize(len, 0);
    } else {
        buffer.truncate(len);
    }
    buffer
}

fn capture_texture_aspect(format: TextureFormat) -> TextureAspect {
    if matches!(
        format,
        TextureFormat::Depth16Unorm
            | TextureFormat::Depth24Plus
            | TextureFormat::Depth24PlusStencil8
            | TextureFormat::Depth32Float
            | TextureFormat::Depth32FloatStencil8
    ) {
        TextureAspect::DepthOnly
    } else {
        TextureAspect::All
    }
}

pub trait SnapshotSync: Send {
    fn frame_id(&self) -> Option<CaptureFrameId> {
        None
    }

    fn captured(
        self: Box<Self>,
        world: &mut DeferredWorld,
        config: &CaptureConfig,
    ) -> Box<dyn SnapshotAsync>;
}

pub trait SnapshotAsync: Send {
    fn captured(&mut self, frame: CapturedFrame<'_>);
}

pub trait GpuCaptureHandler: Send + Sync + 'static {
    fn captured(
        &self,
        world: &World,
        frame_id: Option<CaptureFrameId>,
    ) -> Option<Box<dyn SnapshotSync>>;
}

fn image_copy_driver(world: &World, mut render_context: RenderContext) {
    let Some(copiers) = world.get_resource::<ImageCopiers>() else {
        return;
    };
    let Some(gpu_images) = world.get_resource::<RenderAssets<GpuImage>>() else {
        return;
    };

    let submission = world.get_resource::<CaptureFrameSubmission>();
    let active_frame_id = submission.and_then(CaptureFrameSubmission::active_id);

    for copier in copiers.iter() {
        let Some(src_image) = gpu_images.get(&copier.src_image) else {
            continue;
        };

        let snapshots: Vec<DynSnapshotSync> = copier
            .snapshots
            .iter()
            .filter_map(|handler| handler.captured(world, active_frame_id))
            .collect();
        if let Some(id) = active_frame_id {
            for snapshot in &snapshots {
                if let Some(snapshot_id) = snapshot.frame_id()
                    && snapshot_id != id
                {
                    panic!(
                        "capture snapshot frame mismatch: active={id:?}, snapshot={snapshot_id:?}"
                    );
                }
            }
            if snapshots
                .iter()
                .any(|snapshot| snapshot.frame_id() == Some(id))
            {
                submission
                    .unwrap()
                    .claim(id, copier.config.frame_kind)
                    .unwrap_or_else(|error| panic!("capture channel submission failed: {error:?}"));
            }
        }
        if snapshots.is_empty() {
            continue;
        }

        let size = src_image.texture_descriptor.size;
        let format = src_image.texture_descriptor.format;
        let block_dimensions = format.block_dimensions();
        let block_size = format.block_copy_size(None).unwrap();
        let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(
            (size.width as usize / block_dimensions.0 as usize) * block_size as usize,
        );
        let buffer_size = padded_bytes_per_row as u64 * size.height as u64;
        let Some(permit) = ReadbackPermit::acquire(&copier.in_flight, active_frame_id.is_none())
        else {
            continue;
        };
        let buffer = copier.acquire_buffer(render_context.render_device(), buffer_size);

        render_context.command_encoder().copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &src_image.texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: capture_texture_aspect(format),
            },
            TexelCopyBufferInfo {
                buffer: &buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        std::num::NonZero::<u32>::new(padded_bytes_per_row as u32)
                            .unwrap()
                            .into(),
                    ),
                    rows_per_image: None,
                },
            },
            size,
        );

        let mut queue = copier.queue.lock().unwrap();
        queue.push_back((buffer, snapshots, size.width, size.height, format, permit));
    }

    if let Some(id) = active_frame_id {
        submission
            .unwrap()
            .finish(id)
            .unwrap_or_else(|error| panic!("incomplete capture frame submission: {error:?}"));
    }
}

fn receive_image_from_buffer(mut world: DeferredWorld) {
    let copier_count = world.resource::<ImageCopiers>().len();
    if copier_count == 0 {
        return;
    }

    for idx in 0..copier_count {
        let next = {
            let copiers = world.resource::<ImageCopiers>();
            let Some(copier) = copiers.get(idx) else {
                continue;
            };
            let mut guard = copier.queue.lock().unwrap();
            guard.pop_front().map(
                |(buffer, snapshots, width, height, texture_format, permit)| {
                    (
                        buffer,
                        snapshots,
                        width,
                        height,
                        texture_format,
                        copier.free_buffers.clone(),
                        copier.free_frames.clone(),
                        copier.config.clone(),
                        permit,
                    )
                },
            )
        };

        let Some((
            buffer,
            snapshots,
            width,
            height,
            texture_format,
            free_buffers,
            free_frames,
            config,
            permit,
        )) = next
        else {
            continue;
        };

        // The callback only signals completion; the conversion happens on the async pool, reading
        // the mapped range in place. That keeps the heavy work off the polling thread without the
        // full-frame copy an intermediate `Vec` would cost.
        let (s, r) = futures::channel::oneshot::channel();
        buffer.slice(..).map_async(MapMode::Read, move |result| {
            let _ = s.send(result);
        });

        let snapshots: Vec<(Option<CaptureFrameId>, Box<dyn SnapshotAsync>)> = snapshots
            .into_iter()
            .map(|snapshot| {
                let frame_id = snapshot.frame_id();
                (frame_id, snapshot.captured(&mut world, &config))
            })
            .collect();
        let frame_kind = config.frame_kind;

        AsyncComputeTaskPool::get()
            .spawn(async move {
                // Keep the budget until conversion and every publisher finish.
                let _permit = permit;
                r.await
                    .expect("capture buffer map channel dropped")
                    .expect("Failed to map buffer");

                let layout = FrameLayout::new(frame_kind, texture_format, width, height)
                    .expect("Unsupported capture texture format");
                let mut frame_bytes = acquire_frame_buffer(&free_frames, layout.output_len());

                {
                    let mapped = buffer.slice(..).get_mapped_range();
                    layout.write(&mapped, &mut frame_bytes);
                }
                buffer.unmap();
                free_buffers.lock().unwrap().push(buffer);

                for (frame_id, mut snapshot) in snapshots {
                    snapshot.captured(CapturedFrame {
                        frame_id,
                        kind: frame_kind,
                        width,
                        height,
                        data: frame_bytes.as_slice(),
                    });
                }

                free_frames.lock().unwrap().push(frame_bytes);
            })
            .detach();
    }
}

pub struct CameraCapturePlugin {
    config: CaptureConfig,
    snapshots: Arc<Vec<ToSyncSnapshot>>,
    handle: Handle<Image>,
    expose_config_resource: bool,
}

impl CameraCapturePlugin {
    pub fn new(
        app: &mut App,
        config: CaptureConfig,
        snapshots: Vec<ToSyncSnapshot>,
    ) -> (Self, Handle<Image>) {
        let handle = create_capture_image_handle(
            app,
            config.width,
            config.height,
            config.texture_format,
            RenderAssetUsages::default(),
            TextureUsages::COPY_SRC,
        );

        (
            Self {
                config,
                snapshots: Arc::new(snapshots),
                handle: handle.clone(),
                expose_config_resource: true,
            },
            handle,
        )
    }

    pub fn from_existing_handle(
        config: CaptureConfig,
        handle: Handle<Image>,
        snapshots: Vec<ToSyncSnapshot>,
    ) -> Self {
        Self {
            config,
            snapshots: Arc::new(snapshots),
            handle,
            expose_config_resource: false,
        }
    }
}

impl Plugin for CameraCapturePlugin {
    fn is_unique(&self) -> bool {
        false
    }

    fn build(&self, app: &mut App) {
        assert!(
            FrameLayout::new(
                self.config.frame_kind,
                self.config.texture_format,
                self.config.width,
                self.config.height,
            )
            .is_some(),
            "capture texture format {:?} cannot produce {:?} frames",
            self.config.texture_format,
            self.config.frame_kind,
        );

        if self.expose_config_resource {
            app.insert_resource(self.config.clone());
        }

        let render_app = app.sub_app_mut(RenderApp);
        render_app.world_mut().init_resource::<ImageCopiers>();
        render_app
            .world_mut()
            .init_resource::<CaptureFrameSubmission>();
        render_app
            .world_mut()
            .init_resource::<ImageCopyDriverInstalled>();

        {
            let mut copiers = render_app.world_mut().resource_mut::<ImageCopiers>();
            copiers.push(ImageCopier::new(
                self.config.clone(),
                self.handle.clone(),
                self.snapshots.clone(),
            ));
        }

        let installed = render_app.world().resource::<ImageCopyDriverInstalled>().0;
        if !installed {
            render_app.add_systems(
                RenderGraph,
                image_copy_driver
                    .after(camera_driver)
                    .in_set(RenderGraphSystems::Render),
            );

            render_app
                .world_mut()
                .resource_mut::<ImageCopyDriverInstalled>()
                .0 = true;
            render_app.add_systems(
                Render,
                receive_image_from_buffer.after(RenderSystems::Render),
            );
        }

        if self.expose_config_resource {
            render_app.insert_resource(self.config.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realtime_readback_budget_covers_async_work_and_releases_on_drop() {
        let count = Arc::new(AtomicUsize::new(0));
        let mut permits: Vec<_> = (0..MAX_REALTIME_READBACKS)
            .map(|_| ReadbackPermit::acquire(&count, true).unwrap())
            .collect();
        assert!(ReadbackPermit::acquire(&count, true).is_none());
        drop(permits.pop());
        let replacement = ReadbackPermit::acquire(&count, true).unwrap();
        assert!(ReadbackPermit::acquire(&count, true).is_none());
        // Dataset capture may exceed the realtime limit instead of dropping a channel.
        let dataset = ReadbackPermit::acquire(&count, false).unwrap();
        assert_eq!(count.load(Ordering::Acquire), MAX_REALTIME_READBACKS + 1);
        drop((permits, replacement, dataset));
        assert_eq!(count.load(Ordering::Acquire), 0);
        assert!(ReadbackPermit::acquire(&count, true).is_some());
    }

    #[test]
    fn rgb8_layout_drops_row_padding_and_alpha() {
        // 2x2 BGRA, rows padded out to wgpu's copy alignment.
        let layout =
            FrameLayout::new(CapturedFrameKind::Rgb8, TextureFormat::Bgra8UnormSrgb, 2, 2).unwrap();
        assert_eq!(layout.output_len(), 2 * 2 * 3);

        let mut mapped = vec![0u8; layout.padded_row_bytes * 2];
        mapped[..8].copy_from_slice(&[1, 2, 3, 255, 4, 5, 6, 255]);
        mapped[layout.padded_row_bytes..][..8].copy_from_slice(&[7, 8, 9, 255, 10, 11, 12, 255]);

        let mut out = vec![0u8; layout.output_len()];
        layout.write(&mapped, &mut out);

        assert_eq!(out, vec![3, 2, 1, 6, 5, 4, 9, 8, 7, 12, 11, 10]);
    }

    #[test]
    fn rgb8_layout_rejects_formats_it_cannot_pack() {
        assert!(
            FrameLayout::new(CapturedFrameKind::Rgb8, TextureFormat::Rgba16Float, 4, 4).is_none()
        );
    }

    #[test]
    fn raw_layout_passes_rows_through() {
        let layout = FrameLayout::new(
            CapturedFrameKind::Depth32F,
            TextureFormat::Depth32Float,
            2,
            2,
        )
        .unwrap();
        assert_eq!(layout.output_len(), 2 * 2 * 4);

        let mut mapped = vec![0u8; layout.padded_row_bytes * 2];
        mapped[..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        mapped[layout.padded_row_bytes..][..8].copy_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);

        let mut out = vec![0u8; layout.output_len()];
        layout.write(&mapped, &mut out);

        assert_eq!(out, (1..=16).collect::<Vec<u8>>());
    }

    #[test]
    fn frame_buffer_pool_reuses_allocations() {
        let pool = Mutex::new(Vec::new());

        let buffer = acquire_frame_buffer(&pool, 64);
        assert_eq!(buffer.len(), 64);
        let capacity = buffer.capacity();
        pool.lock().unwrap().push(buffer);

        let reused = acquire_frame_buffer(&pool, 64);
        assert_eq!(reused.len(), 64);
        assert_eq!(reused.capacity(), capacity);
        assert!(pool.lock().unwrap().is_empty());
    }

    #[test]
    fn capture_submission_accepts_all_channels_in_any_order() {
        let submission = CaptureFrameSubmission::default();
        let id = CaptureFrameId::new(7);

        submission.begin(id, CaptureChannels::ALL).unwrap();
        submission.claim(id, CapturedFrameKind::Depth32F).unwrap();
        submission.claim(id, CapturedFrameKind::R32Uint).unwrap();
        submission.claim(id, CapturedFrameKind::Rgb8).unwrap();

        assert_eq!(submission.finish(id), Ok(()));
        assert_eq!(submission.active_id(), None);
    }

    #[test]
    fn capture_submission_rejects_missing_and_duplicate_channels() {
        let submission = CaptureFrameSubmission::default();
        let id = CaptureFrameId::new(11);

        submission.begin(id, CaptureChannels::ALL).unwrap();
        submission.claim(id, CapturedFrameKind::Rgb8).unwrap();
        assert!(matches!(
            submission.claim(id, CapturedFrameKind::Rgb8),
            Err(CaptureSubmissionError::DuplicateChannel { .. })
        ));
        assert!(matches!(
            submission.finish(id),
            Err(CaptureSubmissionError::MissingChannels { .. })
        ));
    }

    #[test]
    fn capture_submission_rejects_cross_frame_claims() {
        let submission = CaptureFrameSubmission::default();
        let active = CaptureFrameId::new(21);
        let wrong = CaptureFrameId::new(22);

        submission.begin(active, CaptureChannels::ALL).unwrap();

        assert!(matches!(
            submission.claim(wrong, CapturedFrameKind::Rgb8),
            Err(CaptureSubmissionError::WrongFrame {
                active: CaptureFrameId(21),
                received: CaptureFrameId(22),
            })
        ));
    }
}
