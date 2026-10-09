"""Biomechanical measurements taken from a live MuJoCo state.

All functions take (model, data) and return plain floats / arrays so they can be used
inside the env step, in evaluation, and in tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import mujoco
import numpy as np

JOINT_NAMES = [
    "lumbar_flex", "lumbar_lat", "lumbar_rot",
    "shoulder_l_flex", "shoulder_l_abd", "shoulder_l_rot", "elbow_l_flex", "wrist_l_flex",
    "shoulder_r_flex", "shoulder_r_abd", "shoulder_r_rot", "elbow_r_flex", "wrist_r_flex",
    "hip_l_flex", "hip_l_abd", "hip_l_rot", "knee_l_flex", "ankle_l_pf",
    "hip_r_flex", "hip_r_abd", "hip_r_rot", "knee_r_flex", "ankle_r_pf",
]


@dataclass
class ModelIndex:
    """Cached ids for everything the env needs each step."""
    joint_qpos: np.ndarray
    joint_dof: np.ndarray
    root_qpos: int
    root_dof: int
    ball_qpos: int
    ball_dof: int
    ball_body: int
    ball_geom: int
    hand_r_body: int
    torso_body: int
    pelvis_body: int
    upper_arm_r_body: int
    forearm_r_body: int
    foot_l_geoms: List[int]
    foot_r_geoms: List[int]
    foot_l_body: int
    foot_r_body: int
    ground_geom: int
    ball_site: int
    wrist_r_site: int
    shoulder_r_site: int
    elbow_r_site: int
    heel_l_site: int
    toe_l_site: int
    heel_r_site: int
    toe_r_site: int
    target_body: int
    target_mocap: int
    weld_eq: int
    elbow_r_flex_qpos: int
    knee_l_flex_qpos: int
    knee_r_flex_qpos: int
    hip_l_flex_qpos: int
    actuator_gear: np.ndarray
    joint_range: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    @classmethod
    def build(cls, m: mujoco.MjModel) -> "ModelIndex":
        def jid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        def bid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        def gid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        def sid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
        jids = [jid(n) for n in JOINT_NAMES]
        for n, j in zip(JOINT_NAMES, jids):
            if j < 0:
                raise ValueError(f"joint {n} missing from model")
        root = jid("root")
        ball = jid("ball_free")
        return cls(
            joint_qpos=np.array([m.jnt_qposadr[j] for j in jids]),
            joint_dof=np.array([m.jnt_dofadr[j] for j in jids]),
            root_qpos=int(m.jnt_qposadr[root]), root_dof=int(m.jnt_dofadr[root]),
            ball_qpos=int(m.jnt_qposadr[ball]), ball_dof=int(m.jnt_dofadr[ball]),
            ball_body=bid("ball"), ball_geom=gid("ball_geom"),
            hand_r_body=bid("hand_r"), torso_body=bid("torso"), pelvis_body=bid("pelvis"),
            upper_arm_r_body=bid("upper_arm_r"), forearm_r_body=bid("forearm_r"),
            foot_l_geoms=[gid("foot_l_geom")], foot_r_geoms=[gid("foot_r_geom")],
            foot_l_body=bid("foot_l"), foot_r_body=bid("foot_r"),
            ground_geom=gid("ground"),
            ball_site=sid("ball_site"), wrist_r_site=sid("wrist_r_site"),
            shoulder_r_site=sid("shoulder_r_site"), elbow_r_site=sid("elbow_r_site"),
            heel_l_site=sid("heel_l_site"), toe_l_site=sid("toe_l_site"),
            heel_r_site=sid("heel_r_site"), toe_r_site=sid("toe_r_site"),
            target_body=bid("target"), target_mocap=-1,
            weld_eq=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "ball_grip"),
            elbow_r_flex_qpos=int(m.jnt_qposadr[jid("elbow_r_flex")]),
            knee_l_flex_qpos=int(m.jnt_qposadr[jid("knee_l_flex")]),
            knee_r_flex_qpos=int(m.jnt_qposadr[jid("knee_r_flex")]),
            hip_l_flex_qpos=int(m.jnt_qposadr[jid("hip_l_flex")]),
            actuator_gear=m.actuator_gear[:, 0].copy(),
            joint_range=np.array([m.jnt_range[j] for j in jids]),
        )


def ground_reaction_force(m: mujoco.MjModel, d: mujoco.MjData, geom_ids: List[int], ground_geom: int) -> np.ndarray:
    """Sum of contact forces (world frame, N) between the given geoms and the ground."""
    total = np.zeros(3)
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        pair = (c.geom1, c.geom2)
        if ground_geom not in pair:
            continue
        other = c.geom2 if c.geom1 == ground_geom else c.geom1
        if other not in geom_ids:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        # contact frame -> world: frame rows are the axes
        frame = c.frame.reshape(3, 3)
        f_world = frame.T @ f6[:3]
        # force on the non-ground body is +normal when geom1 is the body; sign convention:
        if c.geom1 == ground_geom:
            f_world = -f_world
        total += f_world
    return total


def lumbar_load(m: mujoco.MjModel, d: mujoco.MjData, idx: ModelIndex) -> Dict[str, float]:
    """Interaction load between pelvis and torso across the lumbar joint.

    Uses MuJoCo's com-based interaction force cfrc_int (torque[3], force[3], world frame)
    on the torso body, i.e. the net load transmitted from the pelvis through the lumbar
    joint. We project onto the torso's long axis to get compression (negative = compressive)
    and the remainder as shear. Values in N and N m.
    """
    mujoco.mj_rnePostConstraint(m, d)
    cf = d.cfrc_int[idx.torso_body]
    torque_w, force_w = cf[:3], cf[3:]
    R = d.xmat[idx.torso_body].reshape(3, 3)
    axis = R[:, 2]
    f_local = R.T @ force_w
    t_local = R.T @ torque_w
    compression = -float(np.dot(force_w, axis))
    shear = float(np.linalg.norm(force_w - np.dot(force_w, axis) * axis))
    return {
        "lumbar_compression_N": compression,
        "lumbar_shear_N": shear,
        "lumbar_ap_shear_N": float(f_local[0]),
        "lumbar_ml_shear_N": float(f_local[1]),
        "lumbar_flex_moment_Nm": float(t_local[1]),
        "lumbar_lat_moment_Nm": float(t_local[0]),
        "lumbar_rot_moment_Nm": float(t_local[2]),
    }


def lumbar_load_fast(d: mujoco.MjData, idx: ModelIndex):
    """(compression N, shear N) across the lumbar joint; assumes mj_rnePostConstraint was called."""
    force_w = d.cfrc_int[idx.torso_body][3:]
    axis = d.xmat[idx.torso_body].reshape(3, 3)[:, 2]
    along = float(np.dot(force_w, axis))
    shear_vec = force_w - along * axis
    return -along, float(np.sqrt(np.dot(shear_vec, shear_vec)))


def joint_angles_deg(d: mujoco.MjData, idx: ModelIndex) -> Dict[str, float]:
    return {n: float(np.degrees(d.qpos[q])) for n, q in zip(JOINT_NAMES, idx.joint_qpos)}


def upper_arm_vector(d: mujoco.MjData, idx: ModelIndex) -> np.ndarray:
    """Shoulder -> elbow unit vector of the bowling arm, world frame."""
    v = d.site_xpos[idx.elbow_r_site] - d.site_xpos[idx.shoulder_r_site]
    return v / max(np.linalg.norm(v), 1e-9)


def shoulder_alignment_deg(d: mujoco.MjData, idx: ModelIndex) -> float:
    """Angle of the shoulder line (L->R) relative to the delivery direction in the ground plane.
    0 = fully front-on (shoulders square to the batter), 90 = fully side-on."""
    R = d.xmat[idx.torso_body].reshape(3, 3)
    lr = -R[:, 1]  # body +y is left; shoulder line L->R
    lr[2] = 0
    lr /= max(np.linalg.norm(lr), 1e-9)
    return float(np.degrees(np.arctan2(abs(lr[1]), abs(lr[0]))))


def hip_alignment_deg(d: mujoco.MjData, idx: ModelIndex) -> float:
    R = d.xmat[idx.pelvis_body].reshape(3, 3)
    lr = -R[:, 1]
    lr[2] = 0
    lr /= max(np.linalg.norm(lr), 1e-9)
    return float(np.degrees(np.arctan2(abs(lr[1]), abs(lr[0]))))


def trunk_lateral_flexion_deg(d: mujoco.MjData, idx: ModelIndex) -> float:
    """Lateral lean of the torso long axis away from vertical, towards the bowling-arm side (+)."""
    R = d.xmat[idx.torso_body].reshape(3, 3)
    axis = R[:, 2]
    return float(np.degrees(np.arctan2(-axis[1], axis[2])))


def trunk_flexion_deg(d: mujoco.MjData, idx: ModelIndex) -> float:
    R = d.xmat[idx.torso_body].reshape(3, 3)
    axis = R[:, 2]
    return float(np.degrees(np.arctan2(axis[0], axis[2])))


def ball_state(d: mujoco.MjData, idx: ModelIndex):
    pos = d.xpos[idx.ball_body].copy()
    vel = d.qvel[idx.ball_dof:idx.ball_dof + 3].copy()
    omega = d.qvel[idx.ball_dof + 3:idx.ball_dof + 6].copy()
    return pos, vel, omega
