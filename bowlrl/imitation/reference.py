"""Reference (imitation) motion for the delivery stride: BFC -> FFC -> release -> follow-through.

Two sources:
  * `ReferenceMotion.synthetic()` - a literature-informed keyframe action (Portus 2004,
    Worthington 2013, Felton 2023) interpolated with monotone cubic splines. Use this to
    bootstrap training before you have capture data.
  * `ReferenceMotion.from_csv()` - a trajectory exported by bowlrl.capture (OpenCap /
    Pose2Sim -> OpenSim IK -> our joint set).

The CSV format is: time + root (x,y,z,qw,qx,qy,qz) + the 23 joint angles in degrees, in
bowlrl.sim.metrics.JOINT_NAMES order, plus optional event columns in the header comment.

Angles here are DEGREES; the env converts to radians.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R

from bowlrl.sim.metrics import JOINT_NAMES

ROOT_COLS = ["root_x", "root_y", "root_z", "root_qw", "root_qx", "root_qy", "root_qz"]


def _quat_wxyz(yaw_deg: np.ndarray, pitch_deg: np.ndarray) -> np.ndarray:
    """Pelvis orientation from heading (yaw about world z) then anterior tilt (pitch about body y)."""
    eul = np.column_stack([np.asarray(yaw_deg), np.asarray(pitch_deg)])
    q = R.from_euler("ZY", eul, degrees=True).as_quat()  # xyzw
    return np.roll(q, 1, axis=1)


@dataclass
class Events:
    t_bfc: float = 0.0
    t_ffc: float = 0.14
    t_release: float = 0.26
    t_end: float = 0.45


# ------------------------------------------------------------------ synthetic keyframes
# Each keyframe: time, root (x, y, z, yaw_deg), joint angles (deg). Unlisted joints = 0.
# Right-arm bowler, semi-open/side-on action. Yaw is the pelvis heading relative to +x
# (0 = facing the batter; positive = rotated so the left side leads, i.e. side-on).
def _synthetic_keyframes(run_up_speed: float = 5.0) -> List[dict]:
    """root = (x, y, z, yaw_deg, pitch_deg); pitch > 0 = pelvis tilted forward (anterior)."""
    kf = []
    # BFC: back (right) foot lands, body side-on, bowling arm down/behind, front arm up,
    # front knee lifted, slight backward lean.
    kf.append(dict(t=0.00, root=(-1.15, 0.00, 1.00, 45.0, -8.0), joints=dict(
        lumbar_flex=-5, lumbar_lat=0, lumbar_rot=15,
        shoulder_l_flex=140, shoulder_l_abd=25, elbow_l_flex=40,
        shoulder_r_flex=-45, shoulder_r_abd=15, elbow_r_flex=25, wrist_r_flex=-20,
        hip_l_flex=70, hip_l_abd=0, knee_l_flex=75, ankle_l_pf=10,
        hip_r_flex=20, knee_r_flex=25, ankle_r_pf=-5)))
    # mid-stride (back-foot drive / flight)
    kf.append(dict(t=0.07, root=(-0.82, 0.00, 1.02, 38.0, 0.0), joints=dict(
        lumbar_flex=-3, lumbar_lat=-5, lumbar_rot=10,
        shoulder_l_flex=155, shoulder_l_abd=20, elbow_l_flex=30,
        shoulder_r_flex=-65, shoulder_r_abd=25, elbow_r_flex=20, wrist_r_flex=-25,
        hip_l_flex=50, knee_l_flex=30, ankle_l_pf=0,
        hip_r_flex=-5, knee_r_flex=35, ankle_r_pf=15)))
    # FFC: front (left) foot lands heel just behind the crease, knee near straight (braced),
    # trunk fairly upright, bowling arm delayed (still back), front arm high.
    kf.append(dict(t=0.14, root=(-0.52, 0.02, 1.00, 30.0, 8.0), joints=dict(
        lumbar_flex=5, lumbar_lat=-12, lumbar_rot=5,
        shoulder_l_flex=120, shoulder_l_abd=10, elbow_l_flex=35,
        shoulder_r_flex=-85, shoulder_r_abd=35, elbow_r_flex=15, wrist_r_flex=-30,
        hip_l_flex=42, knee_l_flex=8, ankle_l_pf=-10,
        hip_r_flex=-5, knee_r_flex=40, ankle_r_pf=25)))
    # mid delivery: pelvis and trunk fold over the braced front leg, arm circumducts over the
    # top, front arm pulls down and through
    kf.append(dict(t=0.20, root=(-0.34, 0.03, 0.98, 12.0, 15.0), joints=dict(
        lumbar_flex=12, lumbar_lat=-22, lumbar_rot=-10,
        shoulder_l_flex=40, shoulder_l_abd=5, elbow_l_flex=60,
        shoulder_r_flex=-120, shoulder_r_abd=25, elbow_r_flex=12, wrist_r_flex=-25,
        hip_l_flex=45, knee_l_flex=4, ankle_l_pf=-12,
        hip_r_flex=5, knee_r_flex=55, ankle_r_pf=35)))
    # release: arm vertical / slightly forward, wrist flicks, pelvis ~40 deg forward,
    # trunk a further ~25 deg on the pelvis, front knee straight.
    kf.append(dict(t=0.26, root=(-0.20, 0.04, 0.96, 0.0, 20.0), joints=dict(
        lumbar_flex=20, lumbar_lat=-25, lumbar_rot=-25,
        shoulder_l_flex=-30, shoulder_l_abd=0, elbow_l_flex=70,
        shoulder_r_flex=-158, shoulder_r_abd=15, elbow_r_flex=8, wrist_r_flex=20,
        hip_l_flex=58, knee_l_flex=3, ankle_l_pf=-15,
        hip_r_flex=15, knee_r_flex=60, ankle_r_pf=40)))
    # follow-through
    kf.append(dict(t=0.45, root=(0.25, 0.05, 0.92, -15.0, 30.0), joints=dict(
        lumbar_flex=30, lumbar_lat=-20, lumbar_rot=-35,
        shoulder_l_flex=-45, elbow_l_flex=60,
        shoulder_r_flex=-235, shoulder_r_abd=5, elbow_r_flex=30, wrist_r_flex=40,
        hip_l_flex=60, knee_l_flex=20, ankle_l_pf=0,
        hip_r_flex=60, knee_r_flex=70, ankle_r_pf=20)))
    return kf


@dataclass
class ReferenceMotion:
    t: np.ndarray                      # (N,)
    root_pos: np.ndarray               # (N, 3)
    root_quat: np.ndarray              # (N, 4) wxyz
    joints_deg: np.ndarray             # (N, 23)
    events: Events = field(default_factory=Events)
    source: str = "synthetic"
    _pos_spline: object = field(default=None, repr=False)
    _joint_spline: object = field(default=None, repr=False)
    _yaw_spline: object = field(default=None, repr=False)
    _pitch_spline: object = field(default=None, repr=False)
    _table: dict = field(default=None, repr=False)
    _table_dt: float = field(default=0.0, repr=False)

    # ----------------------------------------------------------------- constructors
    @classmethod
    def synthetic(cls, run_up_speed: float = 5.0, dt: float = 0.002) -> "ReferenceMotion":
        kf = _synthetic_keyframes(run_up_speed)
        tk = np.array([k["t"] for k in kf])
        roots = np.array([k["root"] for k in kf])  # x y z yaw pitch
        J = np.zeros((len(kf), len(JOINT_NAMES)))
        for i, k in enumerate(kf):
            for n, v in k["joints"].items():
                J[i, JOINT_NAMES.index(n)] = v
        # run-up carries momentum: enforce initial forward speed by adjusting x spline slope
        t = np.arange(0.0, tk[-1] + 1e-9, dt)
        pos_spl = PchipInterpolator(tk, roots[:, :3], axis=0)
        yaw_spl = PchipInterpolator(tk, roots[:, 3])
        pitch_spl = PchipInterpolator(tk, roots[:, 4])
        j_spl = PchipInterpolator(tk, J, axis=0)
        pos = pos_spl(t)
        quat = _quat_wxyz(yaw_spl(t), pitch_spl(t))
        ref = cls(t=t, root_pos=pos, root_quat=quat, joints_deg=j_spl(t),
                  events=Events(0.0, 0.14, 0.26, tk[-1]), source="synthetic")
        ref._pos_spline, ref._joint_spline = pos_spl, j_spl
        ref._yaw_spline, ref._pitch_spline = yaw_spl, pitch_spl
        return ref.tabulate()

    @classmethod
    def from_csv(cls, path: str) -> "ReferenceMotion":
        events = Events()
        with open(path) as f:
            first = f.readline()
            if first.startswith("#"):
                try:
                    meta = json.loads(first[1:])
                    events = Events(**{k: v for k, v in meta.get("events", {}).items() if k in Events.__dataclass_fields__})
                except json.JSONDecodeError:
                    pass
                header = f.readline().strip().split(",")
            else:
                header = first.strip().split(",")
            data = np.loadtxt(f, delimiter=",")
        col = {n: i for i, n in enumerate(header)}
        t = data[:, col["time"]]
        root_pos = data[:, [col[c] for c in ROOT_COLS[:3]]]
        root_quat = data[:, [col[c] for c in ROOT_COLS[3:]]]
        joints = np.zeros((len(t), len(JOINT_NAMES)))
        for j, n in enumerate(JOINT_NAMES):
            if n in col:
                joints[:, j] = data[:, col[n]]
        ref = cls(t=t, root_pos=root_pos, root_quat=root_quat, joints_deg=joints, events=events, source=path)
        ref._pos_spline = PchipInterpolator(t, root_pos, axis=0)
        ref._joint_spline = PchipInterpolator(t, joints, axis=0)
        eul = R.from_quat(np.roll(root_quat, -1, axis=1)).as_euler("ZYX", degrees=True)
        ref._yaw_spline = PchipInterpolator(t, np.degrees(np.unwrap(np.radians(eul[:, 0]))))
        ref._pitch_spline = PchipInterpolator(t, np.degrees(np.unwrap(np.radians(eul[:, 1]))))
        return ref.tabulate()

    def to_csv(self, path: str) -> None:
        header = ["time"] + ROOT_COLS + JOINT_NAMES
        data = np.column_stack([self.t, self.root_pos, self.root_quat, self.joints_deg])
        with open(path, "w") as f:
            f.write("#" + json.dumps({"events": self.events.__dict__, "source": self.source}) + "\n")
            f.write(",".join(header) + "\n")
            np.savetxt(f, data, delimiter=",", fmt="%.6f")

    # ----------------------------------------------------------------- queries
    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def tabulate(self, dt: float = 0.001) -> "ReferenceMotion":
        """Pre-evaluate the splines on a fixed grid so sample() becomes an array lookup
        (the env calls it several times per control step)."""
        n = int(round(self.duration / dt)) + 1
        grid = np.arange(n) * dt
        keys = ("root_pos", "root_vel", "root_quat", "root_angvel", "joints", "joint_vel")
        tab = {k: [] for k in keys}
        self._table = None
        for tg in grid:
            s = self._sample_spline(float(tg))
            for k in keys:
                tab[k].append(s[k])
        self._table = {k: np.array(v) for k, v in tab.items()}
        self._table_dt = dt
        return self

    def sample(self, time: float) -> Dict[str, np.ndarray]:
        """Pose at `time` (clamped): root pos (3), root quat wxyz (4), joints rad (23),
        plus velocities by central difference."""
        if self._table is not None:
            i = int(np.clip(round(time / self._table_dt), 0, len(self._table["joints"]) - 1))
            return {k: v[i] for k, v in self._table.items()}
        return self._sample_spline(time)

    def _sample_spline(self, time: float) -> Dict[str, np.ndarray]:
        tc = float(np.clip(time, self.t[0], self.t[-1]))
        h = 1e-3
        pos = self._pos_spline(tc)
        vel = (self._pos_spline(min(tc + h, self.t[-1])) - self._pos_spline(max(tc - h, self.t[0]))) / (2 * h)
        q = np.radians(self._joint_spline(tc))
        qd = np.radians(self._joint_spline(min(tc + h, self.t[-1])) - self._joint_spline(max(tc - h, self.t[0]))) / (2 * h)
        yaw = float(self._yaw_spline(tc))
        pitch = float(self._pitch_spline(tc))
        yaw_rate = float(self._yaw_spline(min(tc + h, self.t[-1])) - self._yaw_spline(max(tc - h, self.t[0]))) / (2 * h)
        pitch_rate = float(self._pitch_spline(min(tc + h, self.t[-1])) - self._pitch_spline(max(tc - h, self.t[0]))) / (2 * h)
        quat = _quat_wxyz(np.array([yaw]), np.array([pitch]))[0]
        # angular velocity in the world frame: yaw about world z, pitch about the yawed y axis
        cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        angvel = np.array([0.0, 0.0, np.radians(yaw_rate)]) + np.radians(pitch_rate) * np.array([-sy, cy, 0.0])
        return dict(root_pos=pos, root_vel=vel, root_quat=quat, root_angvel=angvel,
                    joints=q, joint_vel=qd)

    def phase(self, time: float) -> float:
        return float(np.clip(time / self.duration, 0.0, 1.0))

    # ----------------------------------------------------------------- physical consistency
    def make_ground_consistent(self, model, idx, stance=((0.0, 0.10, "r"), (None, None, "l")),
                               clearance: float = 0.004) -> "ReferenceMotion":
        """Recompute root height so the stance foot touches the ground.

        `stance` is a list of (t_start, t_end, foot) windows; None means the FFC event /
        the end of the motion. Between windows (flight) the height is interpolated.
        Requires the MuJoCo model the reference will be tracked with (for its leg lengths).
        """
        import mujoco
        d = mujoco.MjData(model)
        sites = {"r": (idx.heel_r_site, idx.toe_r_site), "l": (idx.heel_l_site, idx.toe_l_site)}
        z_new = self.root_pos[:, 2].copy()
        in_stance = np.zeros(len(self.t), dtype=bool)
        for (t0, t1, foot) in stance:
            t0 = self.events.t_ffc if t0 is None else t0
            t1 = self.duration if t1 is None else t1
            for i, ti in enumerate(self.t):
                if t0 - 1e-9 <= ti <= t1 + 1e-9:
                    d.qpos[idx.root_qpos:idx.root_qpos + 3] = self.root_pos[i]
                    d.qpos[idx.root_qpos + 3:idx.root_qpos + 7] = self.root_quat[i]
                    d.qpos[idx.joint_qpos] = np.radians(self.joints_deg[i])
                    mujoco.mj_kinematics(model, d)
                    sole = min(d.site_xpos[s][2] for s in sites[foot])
                    z_new[i] = self.root_pos[i, 2] - (sole - clearance)
                    in_stance[i] = True
        # flight phases: interpolate between the neighbouring stance values
        if (~in_stance).any():
            z_new[~in_stance] = np.interp(self.t[~in_stance], self.t[in_stance], z_new[in_stance])
        # light smoothing to avoid kinks at the window edges
        k = 5
        zs = np.convolve(np.pad(z_new, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
        self.root_pos[:, 2] = zs
        self._pos_spline = PchipInterpolator(self.t, self.root_pos, axis=0)
        return self.tabulate()
