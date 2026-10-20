use avian3d::prelude::*;
use bevy::input::gamepad::{GamepadRumbleIntensity, GamepadRumbleRequest};
use bevy::prelude::*;
use core::{f32::consts::PI, time::Duration};

use crate::components::{
    Controlled, DartLaunch, DartProjectile, DartSetting, GameLayer, Infantry, InfantryLaunchOffset,
    ProjectileCooldown, ProjectileLifetime, ProjectileSetting,
};
use crate::config::SimulationConfig;
use crate::robomaster::prelude::Projectile;
use crate::statistic::ProjectileStatistics;
use crate::systems::{ControllerState, request_controller_rumble};

// All launch sources share one clock, including frames with no fire request.
#[derive(Resource, Default)]
pub(crate) struct ProjectileCooldownClock(Duration);

fn try_consume_cooldown(
    timer: &mut Timer,
    clock: &mut ProjectileCooldownClock,
    now: Duration,
) -> bool {
    timer.tick(now.saturating_sub(clock.0));
    clock.0 = now;
    if !timer.is_finished() {
        return false;
    }
    timer.reset();
    true
}

pub fn setup_projectile(
    mut commands: Commands,
    config: Res<SimulationConfig>,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.insert_resource(ProjectileCooldownClock::default());
    commands.insert_resource(ProjectileSetting(
        meshes.add(Sphere::new(config.projectile.diameter / 2.0)),
        materials.add(StandardMaterial {
            base_color: Color::srgba(0.132866, 1.0, 0.132869, 0.85),
            emissive: LinearRgba::new(0.132866, 1.0, 0.132869, 0.85),
            emissive_exposure_weight: -1.0,
            alpha_mode: AlphaMode::Opaque,
            ..default()
        }),
    ));
    commands.insert_resource(DartSetting(
        asset_server.load(GltfAssetLabel::Scene(0).from_asset("DART.glb")),
    ));
}

pub fn projectile_launch(
    time: Res<Time>,
    mut cooldown: ResMut<ProjectileCooldown>,
    mut cooldown_clock: ResMut<ProjectileCooldownClock>,
    mut stats: ResMut<ProjectileStatistics>,
    config: Res<SimulationConfig>,
    _asset_server: Res<AssetServer>,
    mut commands: Commands,
    controller: Option<Res<ControllerState>>,
    mut rumble_requests: MessageWriter<GamepadRumbleRequest>,
    setting: Res<ProjectileSetting>,
    infantry: Single<(&LinearVelocity, &AngularVelocity), (With<Infantry>, With<Controlled>)>,
    launch_offset: Single<&GlobalTransform, (With<Controlled>, With<InfantryLaunchOffset>)>,
) {
    if !try_consume_cooldown(&mut cooldown, &mut cooldown_clock, time.elapsed()) {
        return;
    }

    let direction = launch_offset
        .rotation()
        .mul_vec3(Vec3::Y)
        .normalize_or_zero();
    if direction == Vec3::ZERO {
        return;
    }
    stats.increase_launch();
    let vel = infantry.0.0 + direction * config.projectile.speed;
    commands.spawn((
        RigidBody::Dynamic,
        Collider::sphere(config.projectile.diameter / 2.0),
        Mass(config.projectile.mass),
        Friction::new(config.projectile.friction),
        Restitution::new(0.3),
        LinearDamping(config.projectile.linear_damping),
        GameLayer::projectile_collision_layers(true),
        Mesh3d(setting.0.clone()),
        MeshMaterial3d(setting.1.clone()),
        LinearVelocity(vel),
        AngularVelocity(infantry.1.0),
        Transform::from_translation(launch_offset.translation()),
        ProjectileLifetime(Timer::from_seconds(
            config.projectile.lifetime,
            TimerMode::Once,
        )),
        Projectile,
    ));
    request_controller_rumble(
        controller.as_deref(),
        &mut rumble_requests,
        GamepadRumbleIntensity {
            strong_motor: 0.45,
            weak_motor: 0.2,
        },
        Duration::from_millis(80),
    );
}

pub fn projectile_aerodynamics(
    config: Res<SimulationConfig>,
    mut projectiles: Query<Forces, (With<Projectile>, Without<DartProjectile>)>,
) {
    let aero = &config.projectile.aerodynamics;
    if !aero.enabled {
        return;
    }

    let diameter = config.projectile.diameter;
    if diameter <= 0.0 {
        return;
    }
    let air_density = aero.air_density.max(0.0);
    let drag_coefficient = aero.drag_coefficient.max(0.0);
    if air_density == 0.0 || drag_coefficient == 0.0 {
        return;
    }

    let area = PI * (diameter * 0.5).powi(2);
    let wind = Vec3::new(aero.wind[0], aero.wind[1], aero.wind[2]);
    let k = 0.5 * air_density * drag_coefficient * area;

    for mut forces in projectiles.iter_mut() {
        let v_rel = forces.linear_velocity() - wind;
        let speed = v_rel.length();
        if speed <= 1e-3 {
            continue;
        }
        forces.apply_force(-k * speed * v_rel);
    }
}

pub fn dart_launch(
    mut commands: Commands,
    config: Res<SimulationConfig>,
    mut stats: ResMut<ProjectileStatistics>,
    controller: Option<Res<ControllerState>>,
    mut rumble_requests: MessageWriter<GamepadRumbleRequest>,
    setting: Res<DartSetting>,
    launchers: Query<&GlobalTransform, With<DartLaunch>>,
) {
    const DART_FORWARD: Vec3 = Vec3::Y;
    const DART_MODEL_FORWARD: Vec3 = Vec3::NEG_Y;
    const DART_SPEED_MPS: f32 = 17.0;
    const DART_MASS_KG: f32 = 0.25;
    const DART_COLLIDER_RADIUS_M: f32 = 0.001;
    const DART_COLLIDER_LENGTH_M: f32 = 0.001;
    const DART_SPAWN_OFFSET_M: f32 = 0.00;

    let Ok(launcher) = launchers.single() else {
        return;
    };

    let direction = launcher
        .rotation()
        .mul_vec3(DART_FORWARD)
        .normalize_or_zero();
    if direction == Vec3::ZERO {
        return;
    }

    stats.increase_launch();

    let transform =
        Transform::from_translation(launcher.translation() + direction * DART_SPAWN_OFFSET_M)
            .with_rotation(
                launcher.rotation() * Quat::from_rotation_arc(DART_MODEL_FORWARD, DART_FORWARD),
            );
    let voxel = |size| {
        ColliderConstructorHierarchy::new(ColliderConstructor::VoxelizedTrimeshFromMesh {
            voxel_size: size,
            fill_mode: FillMode::FloodFill {
                detect_cavities: true,
            },
        })
        .with_default_layers(GameLayer::projectile_collision_layers(true))
    };
    commands.spawn((
        RigidBody::Dynamic,
        voxel(0.005),
        Mass(DART_MASS_KG),
        Friction::new(config.projectile.friction),
        Restitution::new(0.55),
        LinearDamping(config.projectile.linear_damping),
        GameLayer::projectile_collision_layers(true),
        WorldAssetRoot(setting.0.clone()),
        transform,
        LinearVelocity(direction * DART_SPEED_MPS),
        ProjectileLifetime(Timer::from_seconds(
            config.projectile.lifetime,
            TimerMode::Once,
        )),
        Projectile,
        DartProjectile,
    ));
    request_controller_rumble(
        controller.as_deref(),
        &mut rumble_requests,
        GamepadRumbleIntensity {
            strong_motor: 0.65,
            weak_motor: 0.35,
        },
        Duration::from_millis(140),
    );
}

pub fn cleanup_projectiles(
    time: Res<Time>,
    mut commands: Commands,
    mut projectiles: Query<(Entity, &mut ProjectileLifetime)>,
) {
    for (entity, mut lifetime) in &mut projectiles {
        lifetime.tick(time.delta());
        if lifetime.is_finished() {
            commands.entity(entity).despawn();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::InfantryGimbal;
    use crate::robomaster::prelude::{INFANTRY_THREE_CONFIG, Team};
    use bevy::transform::TransformSystems;

    #[test]
    fn launch_uses_current_propagated_muzzle_position_and_direction() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, TransformPlugin, AssetPlugin::default()))
            .add_message::<GamepadRumbleRequest>()
            .insert_resource(SimulationConfig::default())
            .init_resource::<ProjectileStatistics>()
            .insert_resource(ProjectileCooldown(Timer::from_seconds(
                0.0,
                TimerMode::Once,
            )))
            .init_resource::<ProjectileCooldownClock>()
            .insert_resource(ProjectileSetting(Handle::default(), Handle::default()))
            .add_systems(
                Update,
                |mut gimbal: Single<&mut Transform, With<InfantryGimbal>>| {
                    gimbal.rotation = Quat::from_euler(EulerRot::YXZ, 0.6, -0.2, 0.0);
                },
            )
            .add_systems(
                PostUpdate,
                projectile_launch.after(TransformSystems::Propagate),
            );
        let chassis_transform =
            Transform::from_xyz(3.0, 1.0, -2.0).with_rotation(Quat::from_rotation_y(0.7));
        let chassis_velocity = Vec3::new(0.2, 0.0, -0.1);
        let chassis = app
            .world_mut()
            .spawn((
                chassis_transform,
                Infantry::new(Team::Red, INFANTRY_THREE_CONFIG),
                Controlled,
                LinearVelocity(chassis_velocity),
                AngularVelocity::ZERO,
            ))
            .id();
        let gimbal_transform = Transform::from_xyz(0.04, 0.3, -0.06);
        let gimbal = app
            .world_mut()
            .spawn((
                gimbal_transform,
                InfantryGimbal::default(),
                ChildOf(chassis),
            ))
            .id();
        let muzzle_transform =
            Transform::from_xyz(-0.002, 0.11, -0.01).with_rotation(Quat::from_rotation_x(-1.13));
        app.world_mut().spawn((
            muzzle_transform,
            InfantryLaunchOffset,
            Controlled,
            ChildOf(gimbal),
        ));
        app.update();

        let current_gimbal = Transform {
            rotation: Quat::from_euler(EulerRot::YXZ, 0.6, -0.2, 0.0),
            ..gimbal_transform
        };
        let expected = GlobalTransform::from(chassis_transform)
            .mul_transform(current_gimbal)
            .mul_transform(muzzle_transform);
        let speed = app.world().resource::<SimulationConfig>().projectile.speed;
        let mut projectiles = app
            .world_mut()
            .query_filtered::<(&Transform, &LinearVelocity), With<Projectile>>();
        let (projectile, velocity) = projectiles.single(app.world()).unwrap();
        assert!(
            projectile
                .translation
                .abs_diff_eq(expected.translation(), 1e-5)
        );
        assert!(velocity.0.abs_diff_eq(
            chassis_velocity + expected.rotation() * Vec3::Y * speed,
            1e-5
        ));
        assert_eq!(
            app.world().resource::<ProjectileStatistics>().launch_count,
            1
        );
    }

    #[test]
    fn sparse_fire_pulses_include_time_without_requests() {
        let mut timer = Timer::new(Duration::from_millis(50), TimerMode::Once);
        let mut clock = ProjectileCooldownClock::default();
        assert!(try_consume_cooldown(
            &mut timer,
            &mut clock,
            Duration::from_millis(100)
        ));
        assert!(!try_consume_cooldown(
            &mut timer,
            &mut clock,
            Duration::from_millis(102)
        ));
        assert!(try_consume_cooldown(
            &mut timer,
            &mut clock,
            Duration::from_millis(200)
        ));
        // An idle interval allows one shot without banking several shots.
        assert!(!try_consume_cooldown(
            &mut timer,
            &mut clock,
            Duration::from_millis(201)
        ));
    }

    #[test]
    fn manual_and_talos_requests_share_one_cooldown() {
        let mut timer = Timer::new(Duration::from_millis(50), TimerMode::Once);
        let mut clock = ProjectileCooldownClock::default();
        let mut count = 0;
        for ms in 0..=200 {
            let now = Duration::from_millis(ms);
            count += usize::from(try_consume_cooldown(&mut timer, &mut clock, now));
            assert!(!try_consume_cooldown(&mut timer, &mut clock, now));
        }
        assert_eq!(count, 4);
    }
}
