//! Opt-in, fixed firing lane using real rendered observations and real projectiles.
//! Initial acquisition runs once after the asynchronously loaded robot is ready;
//! all subsequent aiming and firing must come from the external Talos client.
use crate::components::{
    CameraMode, Controlled, FollowingType, Infantry, InfantryGimbal, InfantryLaunchOffset,
    SubscribeAutoAim,
};
use crate::systems::{
    ControllerInput, ControllerState, GameplaySystems, sample_gamepad_controller,
    update_auto_aim_subscription,
};
use avian3d::prelude::{AngularVelocity, LinearVelocity, Position, RigidBody, Rotation};
use bevy::prelude::*;
use std::sync::atomic::Ordering;

// About 3 m from the blue outpost, along the unobstructed firing lane.
const ROBOT_POSITION: Vec3 = Vec3::new(-0.53, 0.29, 2.27);
const INITIAL_LOOK_AT: Vec3 = Vec3::new(-3.077369, 1.299048, 3.848423);

pub(super) struct TalosOutpostTestPlugin;

impl Plugin for TalosOutpostTestPlugin {
    fn build(&self, app: &mut App) {
        info!("DAEDALUS_TALOS_OUTPOST_TEST=1: fixed 3 m firing lane; manual controls disabled");
        app.add_systems(
            Update,
            lock_manual_controls
                .after(sample_gamepad_controller)
                .before(update_auto_aim_subscription)
                .in_set(GameplaySystems::Input),
        )
        .add_systems(
            Update,
            fix_robot_and_initial_view
                .after(GameplaySystems::Input)
                .before(GameplaySystems::Camera),
        );
    }
}

fn lock_manual_controls(
    mut controller: ResMut<ControllerState>,
    mut camera: ResMut<CameraMode>,
    following: Res<SubscribeAutoAim>,
) {
    controller.controlled = ControllerInput {
        auto_aim: true,
        ..default()
    };
    controller.remote = ControllerInput::default();
    camera.0 = FollowingType::Robot;
    following.store(true, Ordering::Release);
}

fn fix_robot_and_initial_view(
    mut commands: Commands,
    mut initialized: Local<bool>,
    mut infantry: Query<
        (
            Entity,
            &mut Transform,
            &RigidBody,
            Option<&mut Position>,
            Option<&mut Rotation>,
            Option<&mut LinearVelocity>,
            Option<&mut AngularVelocity>,
        ),
        (
            With<Infantry>,
            With<Controlled>,
            Without<InfantryGimbal>,
            Without<InfantryLaunchOffset>,
        ),
    >,
    mut gimbal: Query<
        (&mut Transform, &mut InfantryGimbal),
        (
            With<Controlled>,
            With<InfantryGimbal>,
            Without<Infantry>,
            Without<InfantryLaunchOffset>,
        ),
    >,
    muzzle: Query<
        &Transform,
        (
            With<Controlled>,
            With<InfantryLaunchOffset>,
            Without<Infantry>,
            Without<InfantryGimbal>,
        ),
    >,
) {
    let Ok((entity, mut root, rigid_body, position, rotation, velocity, angular_velocity)) =
        infantry.single_mut()
    else {
        return;
    };
    *root = Transform::from_translation(ROBOT_POSITION);
    if *rigid_body != RigidBody::Static {
        commands.entity(entity).insert(RigidBody::Static);
    }
    if let Some(mut position) = position {
        position.0 = ROBOT_POSITION;
    }
    if let Some(mut rotation) = rotation {
        rotation.0 = Quat::IDENTITY;
    }
    if let Some(mut velocity) = velocity {
        velocity.0 = Vec3::ZERO;
    }
    if let Some(mut velocity) = angular_velocity {
        velocity.0 = Vec3::ZERO;
    }

    if *initialized {
        return;
    }
    let (Ok((mut gimbal, mut data)), Ok(muzzle)) = (gimbal.single_mut(), muzzle.single()) else {
        // WorldAssetRoot resolves asynchronously in Bevy 0.19. Do not mark the
        // acquisition complete until setup_vehicle has tagged every required node.
        return;
    };
    // The vehicle GLB has an identity VEHICLE parent. Account for the rotating
    // muzzle offset iteratively instead of aiming from the chassis origin.
    let pivot = ROBOT_POSITION + gimbal.translation;
    let mut origin = pivot;
    for _ in 0..8 {
        let delta = INITIAL_LOOK_AT - origin;
        let yaw = (-delta.x).atan2(-delta.z);
        let pitch = delta.y.atan2(delta.x.hypot(delta.z)) - std::f32::consts::FRAC_PI_2;
        let muzzle_world = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
        gimbal.rotation = muzzle_world * muzzle.rotation.inverse();
        origin = pivot + gimbal.rotation * muzzle.translation;
    }
    (data.local_yaw, data.pitch, _) = gimbal.rotation.to_euler(EulerRot::YXZ);
    *initialized = true;
    info!("Outpost test camera acquired; subsequent aiming and firing require Talos commands");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robomaster::prelude::{INFANTRY_THREE_CONFIG, Team};
    use crate::systems::{GimbalAimTarget, GimbalAimTracker};
    use std::sync::atomic::AtomicBool;

    #[test]
    fn fixture_waits_for_scene_locks_controls_and_preserves_client_aim() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, TransformPlugin, TalosOutpostTestPlugin))
            .init_resource::<ControllerState>()
            .insert_resource(CameraMode(FollowingType::Free))
            .insert_resource(SubscribeAutoAim(AtomicBool::new(false)));
        // The startup frame precedes asynchronous scene/model completion.
        app.update();
        let root = app
            .world_mut()
            .spawn((
                Transform::from_xyz(9.0, 8.0, 7.0),
                GlobalTransform::default(),
                Infantry::new(Team::Red, INFANTRY_THREE_CONFIG),
                Controlled,
                RigidBody::Dynamic,
                Position(Vec3::ONE),
                Rotation(Quat::from_rotation_x(0.3)),
                LinearVelocity(Vec3::ONE),
                AngularVelocity(Vec3::ONE),
            ))
            .id();
        // A partially configured root must not finish acquisition either.
        app.update();
        let gimbal = app
            .world_mut()
            .spawn((
                Transform::from_xyz(0.0, 0.08, 0.0),
                GlobalTransform::default(),
                InfantryGimbal::default(),
                Controlled,
                ChildOf(root),
            ))
            .id();
        let muzzle = app
            .world_mut()
            .spawn((
                Transform::from_translation(Vec3::new(-0.002, 0.11, -0.01))
                    .with_rotation(Quat::from_rotation_x(-1.13)),
                GlobalTransform::default(),
                InfantryLaunchOffset,
                Controlled,
                ChildOf(gimbal),
            ))
            .id();
        {
            let mut controller = app.world_mut().resource_mut::<ControllerState>();
            controller.controlled.shoot = true;
            controller.controlled.movement = Vec2::ONE;
            controller.controlled.gimbal = Vec2::ONE;
        }
        app.update();
        assert_eq!(
            app.world().get::<Transform>(root).unwrap().translation,
            ROBOT_POSITION
        );
        assert_eq!(app.world().get::<Position>(root).unwrap().0, ROBOT_POSITION);
        assert_eq!(
            *app.world().get::<RigidBody>(root).unwrap(),
            RigidBody::Static
        );
        assert_eq!(
            app.world().get::<LinearVelocity>(root).unwrap().0,
            Vec3::ZERO
        );
        assert_eq!(
            app.world().get::<AngularVelocity>(root).unwrap().0,
            Vec3::ZERO
        );
        let pose = app.world().get::<GlobalTransform>(muzzle).unwrap();
        let expected = (INITIAL_LOOK_AT - pose.translation()).normalize();
        assert!((pose.rotation() * Vec3::Y - expected).length() < 1e-5);
        let controller = app.world().resource::<ControllerState>();
        assert!(!controller.controlled.shoot);
        assert_eq!(controller.controlled.movement, Vec2::ZERO);
        assert_eq!(controller.controlled.gimbal, Vec2::ZERO);
        assert!(controller.controlled.auto_aim);
        assert!(
            app.world()
                .resource::<SubscribeAutoAim>()
                .load(Ordering::Acquire)
        );
        assert!(app.world().resource::<CameraMode>().0 == FollowingType::Robot);

        let client_rotation = Quat::from_rotation_y(-0.9);
        let client_target = GimbalAimTarget {
            yaw: 0.4,
            pitch: -1.3,
        };
        app.world_mut()
            .get_mut::<Transform>(gimbal)
            .unwrap()
            .rotation = client_rotation;
        app.world_mut()
            .entity_mut(gimbal)
            .insert(GimbalAimTracker::new(client_target));
        app.world_mut().get_mut::<LinearVelocity>(root).unwrap().0 = Vec3::ONE;
        app.update();
        assert_eq!(
            app.world().get::<Transform>(gimbal).unwrap().rotation,
            client_rotation
        );
        assert_eq!(
            app.world()
                .get::<GimbalAimTracker>(gimbal)
                .unwrap()
                .target(),
            client_target
        );
        assert_eq!(
            app.world().get::<LinearVelocity>(root).unwrap().0,
            Vec3::ZERO
        );
    }
}
