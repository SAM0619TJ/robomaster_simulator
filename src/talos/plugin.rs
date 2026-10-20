use crate::capture::driver::{CaptureConfig, CapturedFrameKind};
use crate::capture::{IMAGE_HEIGHT, IMAGE_WIDTH};
use crate::components::{
    Controlled, InfantryChassis, InfantryGimbal, InfantryLaunchOffset, SubscribeAutoAim,
};
use crate::config::SimulationConfig;
use crate::systems::{
    GimbalAimTarget, GimbalAimTracker, gimbal_pid_controls, projectile_launch,
    update_auto_aim_subscription,
};
use crate::talos::capture::{
    TalosCaptureContext, TalosCapturePlugin, TalosFrameStamp, advance_talos_frame_stamp,
    publish_talos_runtime_state_system,
};
use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;
use bevy::transform::TransformSystems;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use talos_ipc::*;

#[derive(Resource)]
pub struct ShmSubscriberRes(pub Arc<Mutex<ShmSubscriber>>);

#[derive(Resource, Deref, DerefMut)]
pub struct TalosEnabled(pub AtomicBool);

#[derive(Resource, Default)]
struct PendingTalosShot(AtomicU64);

// Keep the shot token across target loss and subscription toggles to avoid replay.
#[derive(Resource, Default)]
struct TalosCommandState {
    timestamp_ns: u64,
    fire_token: u64,
}

const MAX_COMMAND_AGE_NS: u64 = 100_000_000;

pub struct TalosPluginConfig {
    pub width: u32,
    pub height: u32,
    pub fov_y: f32,
    pub texture_format: TextureFormat,
}

impl Default for TalosPluginConfig {
    fn default() -> Self {
        let config = SimulationConfig::default();
        Self {
            width: IMAGE_WIDTH,
            height: IMAGE_HEIGHT,
            fov_y: config.camera.fov.to_radians(),
            texture_format: TextureFormat::Rgba8UnormSrgb,
        }
    }
}

#[derive(Default)]
pub struct TalosPlugin {
    pub config: TalosPluginConfig,
}

impl Plugin for TalosPlugin {
    fn build(&self, app: &mut App) {
        let publisher = match ShmPublisher::create() {
            Ok(p) => {
                info!("talos shm created");
                p
            }
            Err(e) => {
                error!("cannot create talos shm: {}", e);
                return;
            }
        };

        let publisher = Arc::new(Mutex::new(publisher));

        let capture_config = CaptureConfig {
            width: self.config.width,
            height: self.config.height,
            texture_format: self.config.texture_format,
            frame_kind: CapturedFrameKind::Rgb8,
        };

        let capture_context = TalosCaptureContext {
            publisher: publisher.clone(),
            fov_y: self.config.fov_y,
        };

        app.init_resource::<TalosFrameStamp>()
            .init_resource::<PendingTalosShot>()
            .init_resource::<TalosCommandState>();

        app.add_plugins(TalosCapturePlugin {
            config: capture_config,
            context: capture_context,
        });

        match ShmSubscriber::connect() {
            Ok(subscriber) => {
                info!("connected to talos-cpp");
                app.insert_resource(ShmSubscriberRes(Arc::new(Mutex::new(subscriber))));
            }
            Err(_) => {
                info!("could not connect to talos-cpp");
            }
        }

        app.insert_resource(TalosEnabled(AtomicBool::new(true)));
        app.add_systems(Last, (advance_talos_frame_stamp, heartbeat_system));
        app.add_systems(
            Last,
            publish_talos_runtime_state_system.after(advance_talos_frame_stamp),
        );
        app.add_systems(
            Last,
            crate::talos::ground_truth::publish_ground_truth_system
                .after(publish_talos_runtime_state_system),
        );
        app.add_systems(
            Update,
            process_subscription
                .after(update_auto_aim_subscription)
                .before(gimbal_pid_controls),
        );
        app.add_systems(
            PostUpdate,
            projectile_launch
                .after(TransformSystems::Propagate)
                .run_if(take_pending_shot),
        );
        if std::env::var("DAEDALUS_TALOS_OUTPOST_TEST").as_deref() == Ok("1") {
            app.add_plugins(super::outpost_test::TalosOutpostTestPlugin);
        }
    }
}

fn process_subscription(
    context: Option<Res<ShmSubscriberRes>>,
    enabled: Res<SubscribeAutoAim>,
    pending_shot: Res<PendingTalosShot>,
    mut state: ResMut<TalosCommandState>,
    mut commands: Commands,
    gimbal: Option<
        Single<
            (Entity, Option<&mut GimbalAimTracker>),
            (
                With<Controlled>,
                With<InfantryGimbal>,
                Without<InfantryChassis>,
                Without<InfantryLaunchOffset>,
            ),
        >,
    >,
) {
    // The request lives only until this frame's post-propagation launch system.
    pending_shot.0.store(0, Ordering::Release);
    let Some(gimbal) = gimbal else {
        return;
    };
    let (gimbal_entity, tracker) = gimbal.into_inner();
    if !enabled.load(Ordering::Acquire) {
        commands.entity(gimbal_entity).remove::<GimbalAimTracker>();
        state.timestamp_ns = 0;
        return;
    }

    let now_ns = unix_ns();
    let command = context.as_deref().and_then(recv_gimbal_cmd);
    let Some(cmd) = command.filter(|cmd| command_is_valid(cmd, now_ns)) else {
        // A paused/disconnected vision client must not leave the PID following
        // its last setpoint forever. Malformed packets cannot renew this lease.
        if !timestamp_is_fresh(state.timestamp_ns, now_ns) {
            commands.entity(gimbal_entity).remove::<GimbalAimTracker>();
            state.timestamp_ns = 0;
        }
        return;
    };
    state.timestamp_ns = cmd.timestamp_ns;
    if cmd.distance_m == -1.0 {
        commands.entity(gimbal_entity).remove::<GimbalAimTracker>();
        return;
    }
    if consume_fire_request(&cmd, now_ns, &mut state.fire_token) {
        let token = u64::from_le_bytes(cmd._pad[..8].try_into().unwrap());
        pending_shot.0.store(
            if token == 0 { cmd.timestamp_ns } else { token },
            Ordering::Release,
        );
    }

    let target = GimbalAimTarget::from_solver_degrees(cmd.yaw_deg, -cmd.pitch_deg);
    match tracker {
        Some(mut tracker) => tracker.retarget(target),
        None => {
            commands
                .entity(gimbal_entity)
                .insert(GimbalAimTracker::new(target));
        }
    }
}

fn unix_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(0)
}

fn timestamp_is_fresh(timestamp_ns: u64, now_ns: u64) -> bool {
    timestamp_ns != 0
        && now_ns
            .checked_sub(timestamp_ns)
            .is_some_and(|age| age <= MAX_COMMAND_AGE_NS)
}

fn command_is_valid(cmd: &GimbalCmd, now_ns: u64) -> bool {
    timestamp_is_fresh(cmd.timestamp_ns, now_ns)
        && cmd.yaw_deg.is_finite()
        && cmd.pitch_deg.is_finite()
        && cmd.distance_m.is_finite()
        && (cmd.distance_m >= 0.0 || cmd.distance_m == -1.0)
        && cmd.fire_advice <= 1
}

fn consume_fire_request(cmd: &GimbalCmd, now_ns: u64, last_token: &mut u64) -> bool {
    // C++ retains the original shot timestamp in v2 padding even when a newer
    // fire=false pose overwrites the fire=true packet in the latest-only mailbox.
    let token = u64::from_le_bytes(cmd._pad[..8].try_into().unwrap());
    if token == 0 {
        return cmd.fire_advice == 1;
    }
    if token <= *last_token || !timestamp_is_fresh(token, now_ns) {
        return false;
    }
    *last_token = token;
    true
}

fn take_pending_shot(pending_shot: Res<PendingTalosShot>, enabled: Res<SubscribeAutoAim>) -> bool {
    let timestamp_ns = pending_shot.0.swap(0, Ordering::AcqRel);
    enabled.load(Ordering::Acquire) && timestamp_is_fresh(timestamp_ns, unix_ns())
}

fn heartbeat_system(context: Option<Res<TalosCaptureContext>>) {
    if let Some(ctx) = context {
        if let Ok(mut publisher) = ctx.publisher.lock() {
            publisher.update_heartbeat();
        }
    }
}

pub fn publish_pose(
    context: &TalosCaptureContext,
    index: PoseIndex,
    position: [f32; 3],
    quaternion: [f32; 4],
    frame_seq: u64,
    timestamp_ns: u64,
) {
    if let Ok(mut publisher) = context.publisher.lock() {
        publisher.publish_pose(index, position, quaternion, frame_seq, timestamp_ns);
    }
}

pub fn recv_gimbal_cmd(subscriber: &ShmSubscriberRes) -> Option<GimbalCmd> {
    subscriber.0.lock().ok()?.recv_gimbal_cmd()
}

pub const M_ALIGN_MAT3: Mat3 = Mat3::from_cols(
    Vec3::new(0.0, -1.0, 0.0), // M[0,0], M[1,0], M[2,0]
    Vec3::new(0.0, 0.0, 1.0),  // M[0,1], M[1,1], M[2,1]
    Vec3::new(-1.0, 0.0, 0.0), // M[0,2], M[1,2], M[2,2]
);

#[inline]
pub fn to_ros(bevy_transform: Transform) -> Transform {
    let new_rotation = to_ros_quat(bevy_transform.rotation);
    let new_translation = to_ros_translation(bevy_transform.translation);
    Transform::from_translation(new_translation).with_rotation(new_rotation)
}

pub fn to_ros_translation(vec3: Vec3) -> Vec3 {
    let align_rot_mat = M_ALIGN_MAT3;
    let new_translation = align_rot_mat * vec3;
    new_translation
}

pub fn to_ros_quat(quat: Quat) -> Quat {
    let align_rot_mat = M_ALIGN_MAT3;
    let align_quat = Quat::from_mat3(&align_rot_mat);
    let new_rotation = align_quat * quat * align_quat.inverse();
    new_rotation
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command(timestamp_ns: u64) -> GimbalCmd {
        GimbalCmd {
            timestamp_ns,
            yaw_deg: 35.0,
            pitch_deg: -12.0,
            distance_m: 3.0,
            fire_advice: 1,
            ..default()
        }
    }

    fn token_command(now_ns: u64, token: u64, fire_advice: u8) -> GimbalCmd {
        let mut cmd = command(now_ns);
        cmd.fire_advice = fire_advice;
        cmd._pad[..8].copy_from_slice(&token.to_le_bytes());
        cmd
    }

    #[test]
    fn command_rejects_stale_future_and_malformed_inputs() {
        let now = 1_000_000_000;
        assert!(command_is_valid(&command(now), now));
        assert!(command_is_valid(&command(now - MAX_COMMAND_AGE_NS), now));
        for timestamp_ns in [0, now - MAX_COMMAND_AGE_NS - 1, now + 1] {
            assert!(!command_is_valid(&command(timestamp_ns), now));
        }
        for cmd in [
            GimbalCmd {
                yaw_deg: f32::NAN,
                ..command(now)
            },
            GimbalCmd {
                pitch_deg: f32::INFINITY,
                ..command(now)
            },
            GimbalCmd {
                distance_m: -2.0,
                ..command(now)
            },
            GimbalCmd {
                fire_advice: 2,
                ..command(now)
            },
        ] {
            assert!(!command_is_valid(&cmd, now));
        }
        // A fresh upstream no-solution message still clears the tracker.
        assert!(command_is_valid(
            &GimbalCmd {
                distance_m: -1.0,
                ..command(now)
            },
            now
        ));
    }

    #[test]
    fn overwritten_fire_pulse_survives_exactly_once() {
        let now = 1_000_000_000;
        let mut last_token = 0;
        let latest = token_command(now, now - 10_000_000, 0);
        assert!(consume_fire_request(&latest, now, &mut last_token));
        assert!(!consume_fire_request(&latest, now + 1, &mut last_token));
        let repeated = GimbalCmd {
            fire_advice: 1,
            ..latest
        };
        assert!(!consume_fire_request(&repeated, now + 2, &mut last_token));
        assert!(consume_fire_request(
            &token_command(now + 3, now + 3, 0),
            now + 3,
            &mut last_token
        ));
    }

    #[test]
    fn fresh_pose_cannot_renew_an_expired_or_future_shot() {
        let now = 1_000_000_000;
        let mut last_token = 0;
        assert!(!consume_fire_request(
            &token_command(now, now - MAX_COMMAND_AGE_NS - 1, 0),
            now,
            &mut last_token
        ));
        assert!(!consume_fire_request(
            &token_command(now, now + 1, 0),
            now,
            &mut last_token
        ));
        assert!(consume_fire_request(
            &token_command(now, now, 0),
            now,
            &mut last_token
        ));
    }

    #[test]
    fn zero_padding_preserves_legacy_fire_advice() {
        let now = 1_000_000_000;
        let mut last_token = 0;
        assert!(consume_fire_request(&command(now), now, &mut last_token));
        assert!(!consume_fire_request(
            &GimbalCmd {
                fire_advice: 0,
                ..command(now)
            },
            now,
            &mut last_token
        ));
    }

    #[test]
    fn cpp_degree_sign_convention_points_the_muzzle_in_ros_aim_direction() {
        for (yaw, pitch) in [(0.0_f32, 0.0_f32), (35.0, 12.0), (-135.0, -15.0)] {
            // C++ encodes ROS elevation with the opposite sign on the wire.
            let wire_pitch_deg = -pitch;
            let target = GimbalAimTarget::from_solver_degrees(yaw, -wire_pitch_deg);
            let muzzle = Quat::from_euler(EulerRot::YXZ, target.yaw, target.pitch, 0.0);
            let actual = to_ros_translation(muzzle * Vec3::Y);
            let (yaw, pitch) = (yaw.to_radians(), pitch.to_radians());
            let expected = Vec3::new(
                pitch.cos() * yaw.cos(),
                pitch.cos() * yaw.sin(),
                pitch.sin(),
            );
            assert!(
                actual.abs_diff_eq(expected, 1e-5),
                "{actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn disable_and_expired_lease_remove_pid_tracker() {
        let mut app = App::new();
        app.insert_resource(SubscribeAutoAim(AtomicBool::new(false)))
            .init_resource::<PendingTalosShot>()
            .init_resource::<TalosCommandState>()
            .add_systems(Update, process_subscription);
        let gimbal = app
            .world_mut()
            .spawn((
                Controlled,
                InfantryGimbal::default(),
                GimbalAimTracker::new(GimbalAimTarget {
                    yaw: 0.0,
                    pitch: 0.0,
                }),
            ))
            .id();
        app.update();
        assert!(app.world().get::<GimbalAimTracker>(gimbal).is_none());

        app.world()
            .resource::<SubscribeAutoAim>()
            .store(true, Ordering::Release);
        app.world_mut()
            .entity_mut(gimbal)
            .insert(GimbalAimTracker::new(GimbalAimTarget {
                yaw: 0.0,
                pitch: 0.0,
            }));
        app.world_mut()
            .resource_mut::<TalosCommandState>()
            .timestamp_ns = unix_ns();
        app.update();
        assert!(app.world().get::<GimbalAimTracker>(gimbal).is_some());
        app.world_mut()
            .resource_mut::<TalosCommandState>()
            .timestamp_ns = unix_ns() - MAX_COMMAND_AGE_NS - 1;
        app.update();
        assert!(app.world().get::<GimbalAimTracker>(gimbal).is_none());
    }
}
