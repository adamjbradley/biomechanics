"""FastBowlingEnv — Gymnasium environment for optimising a fast-bowling delivery.

Episode = one delivery stride, from back-foot contact (BFC) to just after release.
The agent commands joint torques (23) plus a release signal; the ball is welded to the
bowling hand until release, then an analytic flight model (bowlrl.sim.aero) gives the
pitching point and the error to the target.

Reward = speed + accuracy (terminal, at release)
       + imitation tracking of a reference action (dense, annealed by curriculum)
       - effort - lumbar overload - legality penalties (no-ball, illegal elbow extension).

See configs/base.yaml for every tunable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from bowlrl.imitation.reference import ReferenceMotion
from bowlrl.sim import metrics as M
from bowlrl.sim.aero import AeroConfig, FlightResult, landing_error, simulate_flight
from bowlrl.sim.scaling import MODEL_PATH, scale_model

N_JOINTS = len(M.JOINT_NAMES)


def _quat_diff_rotvec(q_ref: np.ndarray, q_now: np.ndarray) -> np.ndarray:
    """Rotation vector (world frame) taking q_now to q_ref; quaternions are wxyz."""
    q_inv = np.zeros(4)
    q_err = np.zeros(4)
    mujoco.mju_negQuat(q_inv, q_now)
    mujoco.mju_mulQuat(q_err, q_ref, q_inv)
    rv = np.zeros(3)
    mujoco.mju_quat2Vel(rv, q_err, 1.0)
    return rv


def _quat_angle(q_ref: np.ndarray, q_now: np.ndarray) -> float:
    return float(np.linalg.norm(_quat_diff_rotvec(q_ref, q_now)))


def _resolve_auto_reference(cfg: "EnvConfig") -> str:
    """Use data/reference/cem_reference.csv when it exists and its optimisation produced a
    legal delivery faster than cfg.min_cem_speed_kmh; otherwise fall back to synthetic."""
    import json
    d = os.path.normpath(os.path.join(os.path.dirname(cfg.model_path), "..", "..", "data", "reference"))
    csv_path, json_path = os.path.join(d, "cem_reference.csv"), os.path.join(d, "cem_result.json")
    if not os.path.exists(csv_path):
        return "synthetic"
    try:
        info = json.load(open(json_path))["info"]
        ok = (info.get("released") and not info.get("no_ball") and not info.get("illegal_elbow")
              and float(info.get("release_speed_kmh", 0)) >= cfg.min_cem_speed_kmh)
    except Exception:
        ok = False
    return csv_path if ok else "synthetic"


def _omega_max_per_joint() -> np.ndarray:
    """Maximum concentric angular velocity (rad/s) of each torque generator (Felton 2015
    torque-generator fits, King/Yeadon conventions)."""
    table = {"lumbar": 12.0, "shoulder": 45.0, "elbow": 45.0, "wrist": 55.0,
             "hip": 22.0, "knee": 28.0, "ankle": 30.0}
    out = []
    for n in M.JOINT_NAMES:
        key = n.split("_")[0]
        out.append(table[key])
    return np.array(out)


@dataclass
class RewardWeights:
    speed: float = 10.0          # x (v_release / speed_norm)
    accuracy: float = 6.0        # x exp(-(err/sigma)^2)
    track: float = 2.0           # imitation (annealed: track_start -> track_end)
    track_end: float = 0.2
    effort: float = 0.02
    lumbar: float = 1.0
    upright: float = 0.1
    legal: float = 5.0           # penalty for no-ball / illegal action / no release
    full_toss: float = 3.0
    speed_norm: float = 40.0     # m/s (144 km/h) -> reward 1 per unit weight
    accuracy_sigma: float = 0.6  # m


@dataclass
class EnvConfig:
    model_path: str = MODEL_PATH
    subject_height_m: float = 1.85
    subject_mass_kg: float = 85.0
    strength_scale: float = 1.0
    joint_torques: Optional[Dict[str, float]] = None
    control_hz: int = 100
    max_time: float = 0.60
    post_release_time: float = 0.10
    release_mode: str = "policy"          # "policy" | "auto"
    target_xy: Tuple[float, float] = (12.5, 0.10)
    target_random_box: Tuple[float, float] = (0.0, 0.0)   # +/- x, +/- y randomisation
    reference: str = "synthetic"          # "synthetic" | path to CSV
    run_up_speed: float = 5.0
    init_noise_pos: float = 0.03          # rad
    init_noise_vel: float = 0.1           # rad/s
    obs_noise: float = 0.0
    action_noise: float = 0.0
    elbow_limit_deg: float = 15.0
    lumbar_comp_limit_bw: float = 8.0     # body-weights before penalty
    lumbar_shear_limit_bw: float = 2.5
    lumbar_penalty_cap: float = 4.0       # per control step, keeps contact spikes from dominating
    terminate_on_fall: bool = True
    torque_velocity: bool = True          # Hill-type torque-velocity limit on the torque generators
    eccentric_gain: float = 1.3
    min_cem_speed_kmh: float = 80.0       # quality gate for reference: auto (CEM result must beat this)
    root_assist: float = 0.0              # 0..1 virtual spring-damper pulling the pelvis to the reference
    root_assist_end: float = 0.0          # value after curriculum anneal
    assist_kp: float = 3000.0             # N/m
    assist_kd: float = 250.0              # N s/m
    assist_kr: float = 600.0              # N m/rad
    assist_kdr: float = 40.0              # N m s/rad
    seed: Optional[int] = None
    reward: RewardWeights = field(default_factory=RewardWeights)
    aero: AeroConfig = field(default_factory=AeroConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnvConfig":
        d = dict(d)
        rw = d.pop("reward", {}) or {}
        ae = d.pop("aero", {}) or {}
        if "target_xy" in d:
            d["target_xy"] = tuple(d["target_xy"])
        if "target_random_box" in d:
            d["target_random_box"] = tuple(d["target_random_box"])
        return cls(reward=RewardWeights(**rw), aero=AeroConfig(**ae), **d)


class FastBowlingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        xml = scale_model(self.cfg.subject_height_m, self.cfg.subject_mass_kg,
                          self.cfg.strength_scale, self.cfg.joint_torques, self.cfg.model_path)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.idx = M.ModelIndex.build(self.model)
        self.frame_skip = max(1, int(round(1.0 / (self.model.opt.timestep * self.cfg.control_hz))))
        self.dt = self.model.opt.timestep * self.frame_skip
        self.body_weight = float(mujoco.mj_getTotalmass(self.model) * 9.81)
        self.omega_max = _omega_max_per_joint()

        ref = self.cfg.reference
        if ref == "auto":
            ref = _resolve_auto_reference(self.cfg)
        if ref == "synthetic":
            self.ref = ReferenceMotion.synthetic(self.cfg.run_up_speed).make_ground_consistent(self.model, self.idx)
        else:
            self.ref = ReferenceMotion.from_csv(ref)
        self.reference_source = ref
        self.track_weight = self.cfg.reward.track
        self.assist = self.cfg.root_assist
        self._reset_episode_state()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_JOINTS + 1,), dtype=np.float32)
        obs_dim = self._observe().shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._renderer = None
        self.np_random, _ = gym.utils.seeding.np_random(self.cfg.seed)
        self._reset_episode_state()

    # ------------------------------------------------------------------ curriculum
    def set_curriculum(self, progress: float) -> None:
        """progress in [0,1]: anneal tracking weight from reward.track to reward.track_end."""
        p = float(np.clip(progress, 0.0, 1.0))
        self.track_weight = (1 - p) * self.cfg.reward.track + p * self.cfg.reward.track_end
        self.assist = (1 - p) * self.cfg.root_assist + p * self.cfg.root_assist_end

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._reset_episode_state()

        # target
        tx, ty = self.cfg.target_xy
        bx, by = self.cfg.target_random_box
        if bx > 0:
            tx += self.np_random.uniform(-bx, bx)
        if by > 0:
            ty += self.np_random.uniform(-by, by)
        self.target_xy = np.array([tx, ty])
        self.model.body_pos[self.idx.target_body][:2] = self.target_xy

        # pose from reference at BFC + noise
        s = self.ref.sample(0.0)
        q = self.data.qpos
        v = self.data.qvel
        q[self.idx.root_qpos:self.idx.root_qpos + 3] = s["root_pos"]
        q[self.idx.root_qpos + 3:self.idx.root_qpos + 7] = s["root_quat"]
        q[self.idx.joint_qpos] = s["joints"] + self.np_random.normal(0, self.cfg.init_noise_pos, N_JOINTS)
        q[self.idx.joint_qpos] = np.clip(q[self.idx.joint_qpos], self.idx.joint_range[:, 0], self.idx.joint_range[:, 1])
        v[self.idx.root_dof:self.idx.root_dof + 3] = s["root_vel"]
        v[self.idx.root_dof + 3:self.idx.root_dof + 6] = s["root_angvel"]
        v[self.idx.joint_dof] = s["joint_vel"] + self.np_random.normal(0, self.cfg.init_noise_vel, N_JOINTS)
        mujoco.mj_forward(self.model, self.data)

        # drop the body so the lowest point of the back foot touches the ground
        sole = min(self.data.site_xpos[self.idx.heel_r_site][2], self.data.site_xpos[self.idx.toe_r_site][2],
                   self.data.site_xpos[self.idx.heel_l_site][2], self.data.site_xpos[self.idx.toe_l_site][2])
        q[self.idx.root_qpos + 2] -= (sole - 0.003)
        mujoco.mj_forward(self.model, self.data)
        self._zero_stance_foot_velocity()
        mujoco.mj_forward(self.model, self.data)
        self._attach_ball()
        self.data.eq_active[self.idx.weld_eq] = 1
        mujoco.mj_forward(self.model, self.data)
        return self._observe(), self._info()

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        torque_cmd = a[:N_JOINTS]
        if self.cfg.action_noise > 0:
            torque_cmd = np.clip(torque_cmd + self.np_random.normal(0, self.cfg.action_noise, N_JOINTS), -1, 1)
        self._apply_action(torque_cmd)

        reward = 0.0
        # release decision happens before the physics substeps of this control step
        if self.ball_held:
            want = (a[N_JOINTS] > 0.0) if self.cfg.release_mode == "policy" else self._auto_release_condition()
            if want and self._release_gate():
                self._release()

        peak_comp = 0.0
        peak_shear = 0.0
        self._apply_root_assist()
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_rnePostConstraint(self.model, self.data)
            # external (ground) force on each foot: com-based cfrc_ext (torque[3], force[3])
            grf_f = float(np.linalg.norm(self.data.cfrc_ext[self.idx.foot_l_body][3:]))
            grf_b = float(np.linalg.norm(self.data.cfrc_ext[self.idx.foot_r_body][3:]))
            self.peak_grf_front = max(self.peak_grf_front, grf_f)
            self.peak_grf_back = max(self.peak_grf_back, grf_b)
            self._track_events(grf_f)
            comp, shear = M.lumbar_load_fast(self.data, self.idx)
            peak_comp = max(peak_comp, comp)
            peak_shear = max(peak_shear, shear)
        self.t += self.dt
        self.steps += 1
        self.peak_lumbar_comp = max(self.peak_lumbar_comp, peak_comp)
        self.peak_lumbar_shear = max(self.peak_lumbar_shear, peak_shear)
        if self.ball_held:
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.idx.ball_site, vel6, 0)
            self.max_hand_speed = max(self.max_hand_speed, float(np.linalg.norm(vel6[3:])))

        # ---- dense rewards
        rw = self.cfg.reward
        r_track = self.track_weight * self._tracking_reward()
        r_effort = -rw.effort * self._effort(torque_cmd)
        comp_bw = peak_comp / self.body_weight
        shear_bw = peak_shear / self.body_weight
        r_lumbar = -rw.lumbar * min(self.cfg.lumbar_penalty_cap,
                                    max(0.0, comp_bw - self.cfg.lumbar_comp_limit_bw) ** 2
                                    + max(0.0, shear_bw - self.cfg.lumbar_shear_limit_bw) ** 2)
        torso_up = self.data.xmat[self.idx.torso_body].reshape(3, 3)[2, 2]
        r_upright = rw.upright * float(np.clip(torso_up, 0, 1)) if self.ball_held else 0.0
        reward += r_track + r_effort + r_lumbar + r_upright
        self.ep_reward_terms["track"] += r_track
        self.ep_reward_terms["effort"] += r_effort
        self.ep_reward_terms["lumbar"] += r_lumbar
        self.ep_reward_terms["upright"] += r_upright

        # ---- terminal / release rewards
        terminated = False
        truncated = False
        if self.released and not self.release_rewarded:
            reward += self._release_reward()
            self.release_rewarded = True
        if self.released and (self.t - self.t_release) >= self.cfg.post_release_time:
            terminated = True
        fell = self._fallen()
        if fell and self.cfg.terminate_on_fall:
            terminated = True
            if not self.released:
                reward -= rw.legal
                self.ep_reward_terms["legal"] -= rw.legal
                self.fail_reason = "fell"
        if self.t >= self.cfg.max_time - 1e-9 and not terminated:
            truncated = True
            if not self.released:
                reward -= rw.legal
                self.ep_reward_terms["legal"] -= rw.legal
                self.fail_reason = "no_release"

        obs = self._observe()
        if self.cfg.obs_noise > 0:
            obs = obs + self.np_random.normal(0, self.cfg.obs_noise, obs.shape).astype(np.float32)
        return obs, float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------------ internals
    def _apply_action(self, torque_cmd: np.ndarray) -> None:
        if self.cfg.torque_velocity:
            torque_cmd = self._torque_velocity_scale(torque_cmd)
        self.data.ctrl[:] = torque_cmd

    def _effort(self, torque_cmd: np.ndarray) -> float:
        return float(np.mean(self.data.ctrl ** 2))

    def _torque_velocity_scale(self, cmd: np.ndarray) -> np.ndarray:
        """Hill-type torque generator: concentric torque falls linearly to 0 at omega_max,
        eccentric torque plateaus at eccentric_gain x isometric."""
        qd = self.data.qvel[self.idx.joint_dof]
        rel = np.sign(cmd) * qd / self.omega_max        # >0 concentric (shortening)
        factor = np.where(rel >= 0, np.clip(1.0 - rel, 0.0, 1.0),
                          np.clip(1.0 - rel * (self.cfg.eccentric_gain - 1.0) * 2.0, 1.0, self.cfg.eccentric_gain))
        return cmd * factor

    def _reset_episode_state(self):
        self.t = 0.0
        self.steps = 0
        self.ball_held = True
        self.released = False
        self.release_rewarded = False
        self.t_release = None
        self.t_ffc = None
        self.t_bfc = 0.0
        self.t_arm_horizontal = None
        self.elbow_at_horizontal = None
        self.elbow_at_release = None
        self.ffc_metrics: Dict[str, float] = {}
        self.release_metrics: Dict[str, float] = {}
        self.flight: Optional[FlightResult] = None
        self.peak_grf_front = 0.0
        self.peak_grf_back = 0.0
        self.max_hand_speed = 0.0
        self.peak_lumbar_comp = 0.0
        self.peak_lumbar_shear = 0.0
        self.no_ball = False
        self.illegal_elbow = False
        self.fail_reason = ""
        self.target_xy = np.array(self.cfg.target_xy, dtype=float)
        self.ep_reward_terms = {k: 0.0 for k in ("track", "effort", "lumbar", "upright", "speed", "accuracy", "legal", "full_toss")}
        self._front_foot_was_down = False

    def _zero_stance_foot_velocity(self):
        """Solve for stance-leg joint velocities that make the grounded foot stationary.

        A hand-authored reference is rarely consistent with the run-up velocity; a foot that
        touches down while sliding produces an unphysical braking impulse. We least-squares
        adjust the hip (3), knee and ankle velocities of whichever foot is lowest so that the
        world velocity of its heel and toe is ~0.
        """
        m, d, idx = self.model, self.data, self.idx
        zl = min(d.site_xpos[idx.heel_l_site][2], d.site_xpos[idx.toe_l_site][2])
        zr = min(d.site_xpos[idx.heel_r_site][2], d.site_xpos[idx.toe_r_site][2])
        if zl < zr:
            leg = ["hip_l_flex", "hip_l_abd", "hip_l_rot", "knee_l_flex", "ankle_l_pf"]
            sites = [idx.heel_l_site, idx.toe_l_site]
        else:
            leg = ["hip_r_flex", "hip_r_abd", "hip_r_rot", "knee_r_flex", "ankle_r_pf"]
            sites = [idx.heel_r_site, idx.toe_r_site]
        dofs = [idx.joint_dof[M.JOINT_NAMES.index(n)] for n in leg]
        J = np.zeros((6, m.nv))
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        for k, sid in enumerate(sites):
            mujoco.mj_jacSite(m, d, jp, jr, sid)
            J[3 * k:3 * k + 3] = jp
        v_now = J @ d.qvel                          # current foot velocities (6,)
        A = J[:, dofs]
        dq, *_ = np.linalg.lstsq(A, -v_now, rcond=None)
        dq = np.clip(dq, -15.0, 15.0)
        d.qvel[dofs] += dq

    def _apply_root_assist(self):
        """Virtual spring-damper on the pelvis towards the reference root state (curriculum aid)."""
        d, idx = self.data, self.idx
        d.xfrc_applied[idx.pelvis_body][:] = 0.0
        if self.assist <= 0.0 or self.track_weight <= 0.0:
            return
        s = self.ref.sample(self.t)
        p = d.qpos[idx.root_qpos:idx.root_qpos + 3]
        v = d.qvel[idx.root_dof:idx.root_dof + 3]
        f = self.cfg.assist_kp * (s["root_pos"] - p) + self.cfg.assist_kd * (s["root_vel"] - v)
        q_now = d.qpos[idx.root_qpos + 3:idx.root_qpos + 7]
        rotvec = _quat_diff_rotvec(s["root_quat"], q_now)
        w = d.qvel[idx.root_dof + 3:idx.root_dof + 6]
        w_world = np.zeros(3)
        mujoco.mju_rotVecQuat(w_world, w, q_now)
        tau = self.cfg.assist_kr * rotvec + self.cfg.assist_kdr * (s["root_angvel"] - w_world)
        cap_f = 4.0 * self.body_weight
        f = np.clip(f, -cap_f, cap_f)
        tau = np.clip(tau, -600.0, 600.0)
        d.xfrc_applied[idx.pelvis_body][:3] = self.assist * f
        d.xfrc_applied[idx.pelvis_body][3:] = self.assist * tau

    def _attach_ball(self):
        """Place the ball at the hand's ball_site with the hand's velocity so the weld is stress-free."""
        d, idx = self.data, self.idx
        q, v = d.qpos, d.qvel
        q[idx.ball_qpos:idx.ball_qpos + 3] = d.site_xpos[idx.ball_site]
        q[idx.ball_qpos + 3:idx.ball_qpos + 7] = d.xquat[idx.hand_r_body]
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_SITE, idx.ball_site, vel6, 0)
        v[idx.ball_dof:idx.ball_dof + 3] = vel6[3:]
        v[idx.ball_dof + 3:idx.ball_dof + 6] = vel6[:3]

    def _release_gate(self) -> bool:
        """Release is only possible with the hand above the shoulder (a legal overarm release)."""
        d, idx = self.data, self.idx
        return bool(d.site_xpos[idx.ball_site][2] > d.site_xpos[idx.shoulder_r_site][2])

    def _auto_release_condition(self) -> bool:
        """Release once the upper arm has passed vertical and the hand is moving forward."""
        ua = M.upper_arm_vector(self.data, self.idx)
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.idx.ball_site, vel6, 0)
        return bool(ua[2] > 0.70 and ua[0] > 0.05 and vel6[3] > 0.0)

    def _release(self):
        d, idx = self.data, self.idx
        self.ball_held = False
        self.released = True
        self.t_release = self.t
        self.data.eq_active[idx.weld_eq] = 0
        mujoco.mj_forward(self.model, d)
        pos, vel, omega = M.ball_state(d, idx)
        self.release_pos, self.release_vel, self.release_omega = pos.copy(), vel.copy(), omega.copy()
        self.release_speed = float(np.linalg.norm(vel))
        # legality
        heel_x = d.site_xpos[idx.heel_l_site][0]
        toe_x = d.site_xpos[idx.toe_l_site][0]
        foot_y = d.site_xpos[idx.heel_l_site][1]
        self.no_ball = bool(min(heel_x, toe_x) > 0.0 or abs(foot_y) > 1.32)
        self.elbow_at_release = float(np.degrees(d.qpos[idx.elbow_r_flex_qpos]))
        if self.elbow_at_horizontal is not None:
            ext = self.elbow_at_horizontal - self.elbow_at_release
            self.elbow_extension_deg = float(ext)
            self.illegal_elbow = bool(ext > self.cfg.elbow_limit_deg)
        else:
            self.elbow_extension_deg = 0.0
            self.illegal_elbow = False
        # flight
        self.flight = simulate_flight(pos, vel, omega, self.cfg.aero)
        self.landing_err = landing_error(self.flight, self.target_xy) if self.flight.landed else 5.0
        ang = M.joint_angles_deg(d, idx)
        self.release_metrics = {
            "release_speed_ms": self.release_speed,
            "release_speed_kmh": self.release_speed * 3.6,
            "release_height_m": float(pos[2]),
            "release_x_m": float(pos[0]),
            "release_angle_deg": float(np.degrees(np.arctan2(vel[2], np.hypot(vel[0], vel[1])))),
            "front_knee_release_deg": ang["knee_l_flex"],
            "trunk_flexion_release_deg": M.trunk_flexion_deg(d, idx),
            "trunk_lat_flexion_release_deg": M.trunk_lateral_flexion_deg(d, idx),
            "shoulder_alignment_release_deg": M.shoulder_alignment_deg(d, idx),
            "elbow_at_release_deg": self.elbow_at_release,
            "elbow_extension_deg": self.elbow_extension_deg,
            "landing_x_m": float(self.flight.landing_pos[0]),
            "landing_y_m": float(self.flight.landing_pos[1]),
            "landing_error_m": float(self.landing_err),
            "full_toss": float(self.flight.full_toss),
            "stumps_height_m": float(self.flight.stumps_height) if self.flight.stumps_height is not None else float("nan"),
            "no_ball": float(self.no_ball),
            "illegal_elbow": float(self.illegal_elbow),
            "t_release_s": float(self.t),
            "ffc_to_release_s": float(self.t - self.t_ffc) if self.t_ffc is not None else float("nan"),
        }

    def _release_reward(self) -> float:
        rw = self.cfg.reward
        legal = not (self.no_ball or self.illegal_elbow)
        legal_factor = 1.0 if legal else 0.2
        r_speed = rw.speed * (self.release_speed / rw.speed_norm) * legal_factor
        r_acc = rw.accuracy * float(np.exp(-(self.landing_err / rw.accuracy_sigma) ** 2)) * legal_factor
        r_ft = -rw.full_toss if (self.flight is not None and self.flight.full_toss) else 0.0
        r_legal = 0.0 if legal else -rw.legal
        self.ep_reward_terms["speed"] += r_speed
        self.ep_reward_terms["accuracy"] += r_acc
        self.ep_reward_terms["full_toss"] += r_ft
        self.ep_reward_terms["legal"] += r_legal
        return r_speed + r_acc + r_ft + r_legal

    def _track_events(self, grf_front: float):
        d, idx = self.data, self.idx
        # front-foot contact
        if self.t_ffc is None:
            if grf_front > 50.0 and self.data.time > 0.03:
                self.t_ffc = self.data.time
                ang = M.joint_angles_deg(d, idx)
                ua = M.upper_arm_vector(d, idx)
                self.ffc_metrics = {
                    "t_ffc_s": self.t_ffc,
                    "front_knee_ffc_deg": ang["knee_l_flex"],
                    "front_hip_ffc_deg": ang["hip_l_flex"],
                    "front_ankle_ffc_deg": ang["ankle_l_pf"],
                    "trunk_flexion_ffc_deg": M.trunk_flexion_deg(d, idx),
                    "trunk_lat_flexion_ffc_deg": M.trunk_lateral_flexion_deg(d, idx),
                    "shoulder_alignment_ffc_deg": M.shoulder_alignment_deg(d, idx),
                    "hip_alignment_ffc_deg": M.hip_alignment_deg(d, idx),
                    "hip_shoulder_separation_ffc_deg": M.shoulder_alignment_deg(d, idx) - M.hip_alignment_deg(d, idx),
                    "bowling_arm_elevation_ffc_deg": float(np.degrees(np.arcsin(np.clip(ua[2], -1, 1)))),
                    "front_heel_x_ffc_m": float(d.site_xpos[idx.heel_l_site][0]),
                    "pelvis_speed_ffc_ms": float(np.linalg.norm(d.qvel[idx.root_dof:idx.root_dof + 2])),
                    "stride_length_m": float(d.site_xpos[idx.heel_l_site][0] - d.site_xpos[idx.toe_r_site][0]),
                }
        # bowling upper arm reaches horizontal on the way up (ICC elbow-rule start point)
        if self.t_arm_horizontal is None and self.ball_held:
            ua = M.upper_arm_vector(d, idx)
            if ua[2] >= 0.0:
                self.t_arm_horizontal = self.data.time
                self.elbow_at_horizontal = float(np.degrees(d.qpos[idx.elbow_r_flex_qpos]))

    def _tracking_reward(self) -> float:
        if self.track_weight <= 0:
            return 0.0
        s = self.ref.sample(self.t)
        d, idx = self.data, self.idx
        dq = d.qpos[idx.joint_qpos] - s["joints"]
        dq = (dq + np.pi) % (2 * np.pi) - np.pi
        pose = np.exp(-2.0 * float(np.mean(dq ** 2)))
        dv = d.qvel[idx.root_dof:idx.root_dof + 3] - s["root_vel"]
        vel = np.exp(-0.5 * float(np.dot(dv, dv)))
        dz = d.qpos[idx.root_qpos + 2] - s["root_pos"][2]
        height = np.exp(-20.0 * dz * dz)
        ang = _quat_angle(s["root_quat"], d.qpos[idx.root_qpos + 3:idx.root_qpos + 7])
        orient = np.exp(-4.0 * ang * ang)
        return 0.55 * pose + 0.10 * vel + 0.15 * height + 0.20 * orient

    def _fallen(self) -> bool:
        d, idx = self.data, self.idx
        z = d.qpos[idx.root_qpos + 2]
        head_z = d.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")][2]
        return bool(z < 0.55 or head_z < z + 0.1 or abs(d.qpos[idx.root_qpos + 1]) > 2.0)

    def _observe(self) -> np.ndarray:
        d, idx = self.data, self.idx
        root_q = d.qpos[idx.root_qpos + 3:idx.root_qpos + 7]
        root_v = np.zeros(3)
        q_inv = np.zeros(4)
        mujoco.mju_negQuat(q_inv, root_q)
        mujoco.mju_rotVecQuat(root_v, d.qvel[idx.root_dof:idx.root_dof + 3], q_inv)
        root_w = d.qvel[idx.root_dof + 3:idx.root_dof + 6]
        qj = d.qpos[idx.joint_qpos]
        vj = d.qvel[idx.joint_dof] * 0.1
        pelvis = d.xpos[idx.pelvis_body]
        hand_rel = d.site_xpos[idx.ball_site] - pelvis
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_SITE, idx.ball_site, vel6, 0)
        heel_x = d.site_xpos[idx.heel_l_site][0]
        foot_down = 1.0 if self.t_ffc is not None else 0.0
        ref = self.ref.sample(self.t + self.dt)["joints"] if self.track_weight > 0 else np.zeros(N_JOINTS)
        obs = np.concatenate([
            [d.qpos[idx.root_qpos + 2]], root_q, root_v, root_w * 0.2,
            qj, vj,
            [float(self.ball_held), self.t, self.ref.phase(self.t)],
            (self.target_xy - pelvis[:2]) * 0.1,
            hand_rel, vel6[3:] * 0.05,
            [foot_down, heel_x],
            ref,
        ]).astype(np.float32)
        return obs

    def _info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "t": self.t,
            "released": self.released,
            "peak_grf_front_bw": self.peak_grf_front / self.body_weight,
            "peak_grf_back_bw": self.peak_grf_back / self.body_weight,
            "peak_lumbar_comp_bw": self.peak_lumbar_comp / self.body_weight,
            "peak_lumbar_shear_bw": self.peak_lumbar_shear / self.body_weight,
            "max_hand_speed_ms": self.max_hand_speed,
            "fail_reason": self.fail_reason,
            "target_x": float(self.target_xy[0]), "target_y": float(self.target_xy[1]),
        }
        info.update(self.ffc_metrics)
        info.update(self.release_metrics)
        info["reward_terms"] = dict(self.ep_reward_terms)
        return info

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=854)
            self._cam = mujoco.MjvCamera()
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._cam.lookat[:] = [0.0, 0.0, 1.0]
            self._cam.distance = 4.5
            self._cam.azimuth = 90
            self._cam.elevation = -15
        self._cam.lookat[0] = self.data.xpos[self.idx.pelvis_body][0]
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(config: Optional[EnvConfig] = None, render_mode: Optional[str] = None, **kwargs) -> FastBowlingEnv:
    return FastBowlingEnv(config, render_mode=render_mode)


gym.register(id="FastBowling-v0", entry_point="bowlrl.envs.bowling_env:FastBowlingEnv")
