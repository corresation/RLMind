// C functions over RocketSim v3, used by Arena.cpp. The structs here match the Rs* structs there.
use std::ffi::{CStr, c_char};
use std::path::Path;
use std::time::Instant;

use rocketsim::*;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RsVec {
    x: f32,
    y: f32,
    z: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RsPhys {
    pos: RsVec,
    forward: RsVec,
    right: RsVec,
    up: RsVec,
    vel: RsVec,
    ang_vel: RsVec,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RsControls {
    throttle: f32,
    steer: f32,
    pitch: f32,
    yaw: f32,
    roll: f32,
    jump: bool,
    boost: bool,
    handbrake: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RsCar {
    phys: RsPhys,
    prev_controls: RsControls,
    is_on_ground: bool,
    wheels_with_contact: [bool; 4],
    has_jumped: bool,
    has_double_jumped: bool,
    has_flipped: bool,
    flip_rel_torque: RsVec,
    jump_time: f32,
    flip_time: f32,
    is_flipping: bool,
    is_jumping: bool,
    air_time: f32,
    air_time_since_jump: f32,
    boost: f32,
    boosting_time: f32,
    is_supersonic: bool,
    supersonic_grace_timer: f32,
    handbrake_val: f32,
    is_auto_flipping: bool,
    auto_flip_timer: f32,
    auto_flip_torque_scale: f32,
    has_world_contact: bool,
    world_contact_normal: RsVec,
    bump_victim: i32, // -1 = none
    bump_cooldown_timer: f32,
    is_demoed: bool,
    demo_respawn_timer: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RsPad {
    pos: RsVec,
    is_big: bool,
    cooldown: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RsMutators {
    gravity: RsVec,
    ball_radius: f32,
    goal_base_threshold_y: f32,
}

pub const EVENT_CAR_HIT_BALL: u8 = 1;
pub const EVENT_CAR_HIT_CAR: u8 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RsEvent {
    kind: u8,
    is_demo: bool,
    car_a: u32, // Hitter or bumper
    car_b: u32, // Victim
    tick: u64,
    contact_point: RsVec,
    extra_hit_vel: RsVec,
    ball_pos: RsVec,
}

fn to_rs(v: Vec3A) -> RsVec {
    RsVec { x: v.x, y: v.y, z: v.z }
}

fn from_rs(v: RsVec) -> Vec3A {
    Vec3A::new(v.x, v.y, v.z)
}

fn phys_to_rs(p: &PhysState) -> RsPhys {
    RsPhys {
        pos: to_rs(p.pos),
        forward: to_rs(p.rot_mat.x_axis),
        right: to_rs(p.rot_mat.y_axis),
        up: to_rs(p.rot_mat.z_axis),
        vel: to_rs(p.vel),
        ang_vel: to_rs(p.ang_vel),
    }
}

fn phys_from_rs(p: &RsPhys) -> PhysState {
    PhysState {
        pos: from_rs(p.pos),
        rot_mat: Mat3A::from_cols(from_rs(p.forward), from_rs(p.right), from_rs(p.up)),
        vel: from_rs(p.vel),
        ang_vel: from_rs(p.ang_vel),
    }
}

fn controls_to_rs(c: &CarControls) -> RsControls {
    RsControls {
        throttle: c.throttle,
        steer: c.steer,
        pitch: c.pitch,
        yaw: c.yaw,
        roll: c.roll,
        jump: c.jump,
        boost: c.boost,
        handbrake: c.handbrake,
    }
}

fn controls_from_rs(c: &RsControls) -> CarControls {
    CarControls {
        throttle: c.throttle,
        steer: c.steer,
        pitch: c.pitch,
        yaw: c.yaw,
        roll: c.roll,
        jump: c.jump,
        boost: c.boost,
        handbrake: c.handbrake,
    }
}

fn car_to_rs(s: &CarState) -> RsCar {
    RsCar {
        phys: phys_to_rs(&s.phys),
        prev_controls: controls_to_rs(&s.prev_controls),
        is_on_ground: s.is_on_ground,
        wheels_with_contact: s.wheels_with_contact.map(|w| w.is_some()),
        has_jumped: s.has_jumped,
        has_double_jumped: s.has_double_jumped,
        has_flipped: s.has_flipped,
        flip_rel_torque: to_rs(s.flip_rel_torque),
        jump_time: s.jump_time(),
        flip_time: s.flip_time,
        is_flipping: s.is_flipping,
        is_jumping: s.is_jumping,
        air_time: s.air_time,
        air_time_since_jump: s.air_time_since_jump,
        boost: s.boost,
        boosting_time: s.boosting_time,
        is_supersonic: s.is_supersonic,
        supersonic_grace_timer: s.supersonic_grace_timer,
        handbrake_val: s.handbrake_val,
        is_auto_flipping: s.is_auto_flipping,
        auto_flip_timer: s.auto_flip_timer,
        auto_flip_torque_scale: s.auto_flip_torque_scale,
        has_world_contact: s.world_contact_normal.is_some(),
        world_contact_normal: to_rs(s.world_contact_normal.unwrap_or(Vec3A::ZERO)),
        bump_victim: s.bump_cooldown_victim.map_or(-1, |v| v as i32),
        bump_cooldown_timer: s.bump_cooldown_timer,
        is_demoed: s.is_demoed,
        demo_respawn_timer: s.demo_respawn_timer,
    }
}

// Fields RocketSim 2 doesn't have keep their v3 defaults
fn car_from_rs(c: &RsCar) -> CarState {
    let mut s = CarState::DEFAULT;
    s.phys = phys_from_rs(&c.phys);
    s.prev_controls = controls_from_rs(&c.prev_controls);
    s.is_on_ground = c.is_on_ground;
    s.has_jumped = c.has_jumped;
    s.has_double_jumped = c.has_double_jumped;
    s.has_flipped = c.has_flipped;
    s.flip_rel_torque = from_rs(c.flip_rel_torque);
    s.jump_ticks = (c.jump_time * consts::TICK_RATE).round() as u32;
    s.flip_time = c.flip_time;
    s.is_flipping = c.is_flipping;
    s.is_jumping = c.is_jumping;
    s.air_time = c.air_time;
    s.air_time_since_jump = c.air_time_since_jump;
    s.boost = c.boost;
    s.boosting_time = c.boosting_time;
    s.is_supersonic = c.is_supersonic;
    s.supersonic_grace_timer = c.supersonic_grace_timer;
    s.handbrake_val = c.handbrake_val;
    s.is_auto_flipping = c.is_auto_flipping;
    s.auto_flip_timer = c.auto_flip_timer;
    s.auto_flip_torque_scale = c.auto_flip_torque_scale;
    s.world_contact_normal = c.has_world_contact.then(|| from_rs(c.world_contact_normal));
    s.bump_cooldown_victim = (c.bump_victim >= 0).then_some(c.bump_victim as usize);
    s.bump_cooldown_timer = c.bump_cooldown_timer;
    s.is_demoed = c.is_demoed;
    s.demo_respawn_timer = c.demo_respawn_timer;
    s
}

const GAME_MODES: [GameMode; 6] = [
    GameMode::Soccar,
    GameMode::Hoops,
    GameMode::Heatseeker,
    GameMode::Snowday,
    GameMode::Dropshot,
    GameMode::TheVoid,
];

unsafe fn arena_ref<'a>(arena: *mut Arena) -> &'a mut Arena {
    unsafe { &mut *arena }
}

// Loads the meshes ourselves so we can print what got loaded, like RocketSim 2 did
fn load_meshes(folder: &Path, silent: bool) -> std::io::Result<()> {
    let start = Instant::now();
    if !silent {
        println!("Initializing RocketSim v3 (Rust), created by ZealanL, ported by VirxEC...");
    }
    if !folder.is_dir() {
        return Err(std::io::Error::new(std::io::ErrorKind::NotFound, format!("no collision meshes folder at {}", folder.display())));
    }

    let mut meshes = Vec::new();
    let mut counts = Vec::new();
    for game_mode in [GameMode::Soccar, GameMode::Hoops, GameMode::Dropshot] {
        let mode_folder = folder.join(game_mode.name());
        if !mode_folder.is_dir() {
            continue;
        }
        if !silent {
            println!("Loading arena meshes for {}...", game_mode.name());
        }
        let mut paths: Vec<_> = std::fs::read_dir(&mode_folder)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "cmf"))
            .collect();
        paths.sort();

        let mut files = Vec::new();
        for path in paths {
            let bytes = std::fs::read(&path)?;
            let mesh = CollisionMeshFile::read_from_bytes(&bytes)?;
            if !silent {
                println!(
                    "   > Loaded {} verts and {} tris, hash: {:#x}",
                    mesh.get_vertices().len(),
                    mesh.get_indices().len() / 3,
                    mesh.get_hash()
                );
            }
            files.push(bytes);
        }
        counts.push((game_mode, files.len()));
        meshes.push((game_mode, files));
    }

    rocketsim::init_from_mem(meshes, true)?;
    if !silent {
        println!("RocketSim::Init(): Finished loading arena collision meshes:");
        for (game_mode, count) in counts {
            println!(" > {}: {count}", game_mode.name());
        }
        println!("Finished initializing RocketSim in {:.3}s!", start.elapsed().as_secs_f32());
    }
    Ok(())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_init(meshes_path: *const c_char, silent: bool) -> bool {
    let path = unsafe { CStr::from_ptr(meshes_path) }.to_string_lossy().into_owned();
    match load_meshes(Path::new(&path), silent) {
        Ok(()) => true,
        Err(e) => {
            eprintln!("RocketSim init failed: {e}");
            false
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rs_arena_new(game_mode: u8) -> *mut Arena {
    Box::into_raw(Box::new(Arena::new(GAME_MODES[game_mode as usize])))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_arena_free(arena: *mut Arena) {
    drop(unsafe { Box::from_raw(arena) });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_arena_add_car(arena: *mut Arena, team: u8) -> u32 {
    let team = if team == 0 { Team::Blue } else { Team::Orange };
    unsafe { arena_ref(arena) }.add_car(team, CarBodyConfig::OCTANE) as u32
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_arena_mutators(arena: *mut Arena, out: *mut RsMutators) {
    let m = unsafe { arena_ref(arena) }.mutator_config();
    unsafe {
        *out = RsMutators {
            gravity: to_rs(m.gravity),
            ball_radius: m.ball_radius,
            goal_base_threshold_y: m.goal_base_threshold_y,
        }
    };
}

// Steps with the given controls (one per car), writing car-ball and car-car hits to events
// Returns the number of events, which can be more than events_cap (the extra ones aren't written)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_arena_step(
    arena: *mut Arena, ticks: u32, controls: *const RsControls, events: *mut RsEvent, events_cap: u32,
) -> u32 {
    let arena = unsafe { arena_ref(arena) };
    let num_cars = arena.num_cars();
    let controls = unsafe { std::slice::from_raw_parts(controls, num_cars) };
    for (i, c) in controls.iter().enumerate() {
        arena.set_car_controls(i, controls_from_rs(c));
    }

    let mut num_events = 0u32;
    for _ in 0..ticks {
        let tick = arena.tick_count();
        let mut hits = arrayvec::ArrayVec::<RsEvent, 16>::new();
        for event in arena.step_tick() {
            let ev = match event {
                ArenaEvent::CarHitBall(e) => RsEvent {
                    kind: EVENT_CAR_HIT_BALL,
                    is_demo: false,
                    car_a: e.car_idx as u32,
                    car_b: 0,
                    tick,
                    contact_point: to_rs(e.contact_point),
                    extra_hit_vel: to_rs(e.extra_hit_vel),
                    ball_pos: RsVec { x: 0., y: 0., z: 0. },
                },
                ArenaEvent::CarHitCar(e) => RsEvent {
                    kind: EVENT_CAR_HIT_CAR,
                    is_demo: e.is_demo,
                    car_a: e.bumper_car_idx as u32,
                    car_b: e.victim_car_idx as u32,
                    tick,
                    contact_point: to_rs(e.contact_point),
                    extra_hit_vel: RsVec { x: 0., y: 0., z: 0. },
                    ball_pos: RsVec { x: 0., y: 0., z: 0. },
                },
                _ => continue,
            };
            let _ = hits.try_push(ev);
        }

        let ball_pos = to_rs(arena.get_ball_state().phys.pos);
        for mut ev in hits {
            ev.ball_pos = ball_pos;
            if num_events < events_cap {
                unsafe { *events.add(num_events as usize) = ev };
            }
            num_events += 1;
        }
    }
    num_events
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_arena_tick_count(arena: *mut Arena) -> u64 {
    unsafe { arena_ref(arena) }.tick_count()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_arena_num_pads(arena: *mut Arena) -> u32 {
    unsafe { arena_ref(arena) }.num_boost_pads() as u32
}

// Fills cars (num_cars), ball and pads (num_pads)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_arena_get(arena: *mut Arena, cars: *mut RsCar, ball: *mut RsPhys, pads: *mut RsPad) {
    let arena = unsafe { arena_ref(arena) };
    for i in 0..arena.num_cars() {
        unsafe { *cars.add(i) = car_to_rs(arena.get_car_state(i)) };
    }
    unsafe { *ball = phys_to_rs(&arena.get_ball_state().phys) };
    for i in 0..arena.num_boost_pads() {
        let config = arena.get_boost_pad_config(i);
        let state = arena.get_boost_pad_state(i);
        unsafe {
            *pads.add(i) = RsPad {
                pos: to_rs(config.pos),
                is_big: config.is_big,
                cooldown: state.cooldown,
            }
        };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_arena_set_car(arena: *mut Arena, idx: u32, car: *const RsCar) {
    unsafe { arena_ref(arena) }.set_car_state(idx as usize, car_from_rs(unsafe { &*car }));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_arena_set_ball(arena: *mut Arena, ball: *const RsPhys) {
    let arena = unsafe { arena_ref(arena) };
    let mut state = BallState::DEFAULT;
    state.phys = phys_from_rs(unsafe { &*ball });
    arena.set_ball_state(state);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_arena_set_pad(arena: *mut Arena, idx: u32, cooldown: f32) {
    let arena = unsafe { arena_ref(arena) };
    let mut state = arena.get_boost_pad_state(idx as usize);
    state.cooldown = cooldown;
    arena.set_boost_pad_state(idx as usize, state);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_arena_reset_kickoff(arena: *mut Arena, seed: i64) {
    unsafe { arena_ref(arena) }.reset_to_random_kickoff((seed >= 0).then_some(seed as u64));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_arena_is_ball_scored(arena: *mut Arena) -> bool {
    unsafe { arena_ref(arena) }.is_ball_scored()
}
