import os

import numpy as np
import pytest

from bowlrl.envs.bowling_env import EnvConfig, FastBowlingEnv, N_JOINTS
from bowlrl.imitation.pd_controller import PDGains, ReferencePDController, rollout
from bowlrl.imitation.reference import ReferenceMotion
from bowlrl.sim.aero import AeroConfig, simulate_flight
from bowlrl.sim.metrics import JOINT_NAMES
from bowlrl.sim.scaling import scale_model


@pytest.fixture(scope="module")
def env():
    return FastBowlingEnv(EnvConfig(seed=0, reference="synthetic"))


# ------------------------------------------------------------------ physics / model
def test_model_compiles_and_mass(env):
    assert env.model.nu == N_JOINTS
    assert abs(env.body_weight / 9.81 - 85.0) < 1.0


def test_scaling_changes_mass_and_length():
    import mujoco
    xml = scale_model(height_m=2.0, mass_kg=100.0, strength_scale=1.2)
    m = mujoco.MjModel.from_xml_string(xml)
    assert abs(mujoco.mj_getTotalmass(m) - 100.0) < 1.5
    base = mujoco.MjModel.from_xml_string(scale_model())
    assert m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")][2] > \
        base.body_pos[mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, "pelvis")][2]
    assert np.all(m.actuator_gear[:, 0] > base.actuator_gear[:, 0])


def test_ball_attached_at_reset(env):
    env.reset(seed=1)
    ball = env.data.xpos[env.idx.ball_body]
    site = env.data.site_xpos[env.idx.ball_site]
    assert np.linalg.norm(ball - site) < 1e-3
    assert env.ball_held and env.data.eq_active[env.idx.weld_eq] == 1


def test_back_foot_on_ground_at_reset(env):
    env.reset(seed=2)
    z = min(env.data.site_xpos[env.idx.heel_r_site][2], env.data.site_xpos[env.idx.toe_r_site][2])
    assert -0.01 < z < 0.03


# ------------------------------------------------------------------ gym API
def test_spaces_and_step(env):
    obs, info = env.reset(seed=3)
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape and np.isfinite(r)
    assert "peak_grf_front_bw" in info and "reward_terms" in info


def test_episode_terminates(env):
    env.reset(seed=4)
    for _ in range(200):
        _, _, term, trunc, info = env.step(np.zeros(N_JOINTS + 1))
        if term or trunc:
            break
    assert term or trunc
    assert env.t <= env.cfg.max_time + 1e-6


def test_gym_registration():
    import gymnasium as gym
    e = gym.make("FastBowling-v0")
    e.reset(seed=0)
    e.step(e.action_space.sample())


# ------------------------------------------------------------------ release / flight / legality
def test_release_deactivates_weld_and_gives_flight(env):
    env.reset(seed=5)
    env.cfg.release_mode = "policy"
    # force a release once the hand is above the shoulder using the PD controller to raise the arm
    ctl = ReferencePDController(env, PDGains(kp=6, kd=0.1))
    env.cfg.root_assist, env.assist = 1.0, 1.0
    released = False
    for _ in range(80):
        a = ctl.act()
        a[-1] = 1.0
        _, _, term, trunc, info = env.step(a)
        if info["released"]:
            released = True
            break
        if term or trunc:
            break
    env.cfg.root_assist, env.assist = 0.0, 0.0
    assert released
    assert env.data.eq_active[env.idx.weld_eq] == 0
    assert env.flight is not None and info["release_speed_kmh"] > 0
    assert "landing_error_m" in info and "no_ball" in info and "illegal_elbow" in info


def test_flight_model_physics():
    cfg = AeroConfig()
    res = simulate_flight(np.array([0.0, 0.0, 2.2]), np.array([38.0, 0.0, -2.0]), np.zeros(3), cfg)
    assert res.landed and 5.0 < res.landing_pos[0] < 20.0
    assert res.landing_speed < 38.5  # drag slows it
    slow = simulate_flight(np.array([0.0, 0.0, 2.2]), np.array([20.0, 0.0, -2.0]), np.zeros(3), cfg)
    assert slow.landing_pos[0] < res.landing_pos[0]
    # back-spin (omega about -y for +x flight) produces lift -> longer carry
    spin = simulate_flight(np.array([0.0, 0.0, 2.2]), np.array([30.0, 0.0, -2.0]), np.array([0.0, -150.0, 0.0]), cfg)
    res = simulate_flight(np.array([0.0, 0.0, 2.2]), np.array([30.0, 0.0, -2.0]), np.zeros(3), cfg)
    assert spin.landing_pos[0] > res.landing_pos[0]


def test_no_ball_rule(env):
    """A release with the front heel beyond the popping crease is a no-ball."""
    env.reset(seed=6)
    env.data.qpos[env.idx.root_qpos] += 1.5   # move the whole bowler 1.5 m past the crease
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    env._release()
    assert env.no_ball


def test_elbow_rule_logic(env):
    env.reset(seed=7)
    env.elbow_at_horizontal = 40.0
    env.data.qpos[env.idx.elbow_r_flex_qpos] = np.radians(10.0)
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    env._release()
    assert env.illegal_elbow and env.elbow_extension_deg > env.cfg.elbow_limit_deg


# ------------------------------------------------------------------ reference / imitation
def test_synthetic_reference_consistency(env):
    ref = env.ref
    assert ref.joints_deg.shape[1] == len(JOINT_NAMES)
    s = ref.sample(0.14)
    assert abs(np.degrees(s["joints"][JOINT_NAMES.index("knee_l_flex")]) - 8.0) < 3.0
    assert 0.8 < s["root_pos"][2] < 1.2


def test_reference_csv_roundtrip(tmp_path, env):
    p = tmp_path / "ref.csv"
    env.ref.to_csv(str(p))
    r2 = ReferenceMotion.from_csv(str(p))
    a, b = env.ref.sample(0.2), r2.sample(0.2)
    assert np.allclose(a["joints"], b["joints"], atol=1e-3)
    assert np.allclose(a["root_pos"], b["root_pos"], atol=1e-3)


def test_pd_baseline_with_assist_completes_delivery():
    e = FastBowlingEnv(EnvConfig(seed=0, init_noise_pos=0, init_noise_vel=0, release_mode="auto", root_assist=1.0))
    ctl = ReferencePDController(e, PDGains(kp=6, kd=0.1))
    R, info, _ = rollout(e, ctl, seed=0)
    assert info["released"] and info["release_speed_kmh"] > 20


def test_curriculum_anneals(env):
    env.set_curriculum(0.0)
    assert env.track_weight == env.cfg.reward.track
    env.set_curriculum(1.0)
    assert abs(env.track_weight - env.cfg.reward.track_end) < 1e-9


# ------------------------------------------------------------------ CEM parametrisation
def test_cem_knots():
    from bowlrl.optim.cem import CEMConfig, knots_to_ctrl, release_time
    cfg = CEMConfig(knots=4, horizon=0.3)
    params = np.concatenate([np.linspace(-1, 1, N_JOINTS * 4), [0.25]])
    c0, c1 = knots_to_ctrl(params, cfg, 0.0), knots_to_ctrl(params, cfg, 0.3)
    assert c0.shape == (N_JOINTS,) and np.all(np.abs(c0) <= 1) and not np.allclose(c0, c1)
    assert release_time(params, cfg) == 0.25


# ------------------------------------------------------------------ capture importer
def test_opensim_mot_import(tmp_path):
    from bowlrl.capture.opensim_to_reference import convert
    from bowlrl.imitation.reference import Events
    cols = ["time", "pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz",
            "hip_flexion_l", "knee_angle_l", "ankle_angle_l", "arm_flex_r", "elbow_flex_r", "lumbar_extension"]
    t = np.arange(0, 1.0, 0.01)
    data = np.column_stack([t, 5 * np.sin(t), np.zeros_like(t), 30 * np.ones_like(t), 4 * t, 0.95 + 0 * t, 0 * t,
                            30 * np.sin(3 * t), 10 + 0 * t, -5 + 0 * t, -100 * t, 10 + 0 * t, -20 + 0 * t])
    mot = tmp_path / "trial.mot"
    with open(mot, "w") as f:
        f.write("Coordinates\nversion=1\nnRows=100\nnColumns=13\ninDegrees=yes\nendheader\n")
        f.write("\t".join(cols) + "\n")
        np.savetxt(f, data, delimiter="\t", fmt="%.5f")
    out = tmp_path / "ref.csv"
    ev = convert(str(mot), str(out), Events(0.3, 0.45, 0.6, 0.8))
    ref = ReferenceMotion.from_csv(str(out))
    assert abs(ev.t_ffc - 0.15) < 1e-6 and ref.duration > 0.3
    s = ref.sample(0.0)
    assert abs(np.degrees(s["joints"][JOINT_NAMES.index("lumbar_flex")]) - 20.0) < 1.0   # sign flipped
    assert abs(np.degrees(s["joints"][JOINT_NAMES.index("ankle_l_pf")]) - 5.0) < 1.0
