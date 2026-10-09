"""Reference-tracking PD controller with optional computed-torque feed-forward.

Uses:
  * baseline: how fast does the *reference* action bowl in this physics model?
  * behaviour-cloning labels (scripts/pretrain_bc.py)
  * sanity check that the env + model can produce a legal delivery before RL

Feed-forward torques come from MuJoCo inverse dynamics (mj_inverse) evaluated on the
reference kinematics, so the PD loop only has to correct contact and modelling error.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from bowlrl.envs.bowling_env import FastBowlingEnv, N_JOINTS


@dataclass
class PDGains:
    kp: float = 4.0          # x gear  (N m per rad of error): full torque at ~14 deg error
    kd: float = 0.08         # x gear  (N m per rad/s)
    kp_scale: dict = None    # per-joint multipliers, e.g. {"knee_l_flex": 2.0}
    feedforward: bool = False
    release_time: float | None = None   # None -> use env release_mode / reference event


class ReferencePDController:
    def __init__(self, env: FastBowlingEnv, gains: PDGains | None = None):
        self.env = env
        self.g = gains or PDGains()
        self.kp = self.g.kp * env.idx.actuator_gear.astype(float)
        self.kd = self.g.kd * env.idx.actuator_gear.astype(float)
        from bowlrl.sim.metrics import JOINT_NAMES
        if self.g.kp_scale:
            for n, k in self.g.kp_scale.items():
                self.kp[JOINT_NAMES.index(n)] *= k
                self.kd[JOINT_NAMES.index(n)] *= np.sqrt(k)

    def feedforward_torque(self, t: float) -> np.ndarray:
        """Inverse-dynamics joint torques for the reference at time t (N m)."""
        env = self.env
        m, d = env.model, env.data
        s0 = env.ref.sample(t)
        h = 2e-3
        s1 = env.ref.sample(t + h)
        sm = env.ref.sample(max(t - h, 0.0))
        qacc = np.zeros(m.nv)
        qacc[env.idx.joint_dof] = (s1["joints"] - 2 * s0["joints"] + sm["joints"]) / (h * h)
        # evaluate inverse dynamics on a scratch copy so we don't disturb the live state
        scratch = getattr(self, "_scratch", None)
        if scratch is None:
            scratch = self._scratch = mujoco.MjData(m)
        scratch.qpos[:] = d.qpos
        scratch.qvel[:] = d.qvel
        scratch.qpos[env.idx.joint_qpos] = s0["joints"]
        scratch.qvel[env.idx.joint_dof] = s0["joint_vel"]
        scratch.qacc[:] = qacc
        mujoco.mj_inverse(m, scratch)
        return scratch.qfrc_inverse[env.idx.joint_dof].copy()

    def act(self, t: float | None = None) -> np.ndarray:
        env = self.env
        t = env.t if t is None else t
        s = env.ref.sample(t + env.dt)
        q = env.data.qpos[env.idx.joint_qpos]
        qd = env.data.qvel[env.idx.joint_dof]
        tau = self.kp * (s["joints"] - q) + self.kd * (s["joint_vel"] - qd)
        if self.g.feedforward:
            tau = tau + self.feedforward_torque(t)
        ctrl = np.clip(tau / env.idx.actuator_gear, -1.0, 1.0)
        release = -1.0
        t_rel = self.g.release_time if self.g.release_time is not None else env.ref.events.t_release
        if env.cfg.release_mode == "policy" and t >= t_rel - 1e-9:
            release = 1.0
        return np.concatenate([ctrl, [release]]).astype(np.float32)


def rollout(env: FastBowlingEnv, controller: ReferencePDController, seed: int | None = None):
    obs, info = env.reset(seed=seed)
    total = 0.0
    done = False
    frames = []
    while not done:
        a = controller.act()
        obs, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
        if env.render_mode == "rgb_array":
            frames.append(env.render())
    return total, info, frames
