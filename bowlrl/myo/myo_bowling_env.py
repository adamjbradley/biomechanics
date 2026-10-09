"""Phase 3 — muscle-driven bowler on a MyoSuite / MyoConverter musculoskeletal model.

The torque-driven bowler (Phase 2) tells you the optimal *joint* technique. This env swaps
the torque generators for Hill-type muscles so that strength and muscle fatigue limits are
physiological and subject-specific: the action is muscle excitation in [0,1] for every
muscle actuator in the model (plus torque motors for any residual non-muscle actuators).

The same reward, legality checks, ball flight and injury proxies are reused unchanged.

Model sources (all open):
  * MyoSuite's `myolegs_with_torso.xml` (80 leg muscles + rigid torso) — lower-body muscles,
    use torque actuators on the arm (add them with `arm_torque_joints`).
  * A full-body subject-specific model: scale OpenCap's OpenSim model to the bowler, then
    `myoconverter` (https://github.com/MyoHub/myoconverter) -> MuJoCo XML with muscles.
  * MyoSuite's MyoSkeleton + MyoArm/MyoLegs assets combined (see bowlrl/myo/README.md).

Usage:
    from bowlrl.myo.myo_bowling_env import MyoBowlingEnv, MyoMapping
    env = MyoBowlingEnv(myo_xml="path/to/model.xml", mapping=MyoMapping(...), config=EnvConfig(...))

The mapping tells the env which bodies/joints in the foreign model play the roles that
bowler.xml names explicitly (hand, feet, torso, pelvis, root joint, and the 23 joint
coordinates). Unmapped coordinates are simply not tracked/measured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from bowlrl.envs.bowling_env import EnvConfig, FastBowlingEnv, N_JOINTS
from bowlrl.imitation.reference import ReferenceMotion
from bowlrl.sim import metrics as M


@dataclass
class MyoMapping:
    """Names in the foreign (MyoSuite/MyoConverter) model."""
    root_joint: str = "root"
    pelvis_body: str = "pelvis"
    torso_body: str = "torso"
    head_body: str = "head"
    hand_body: str = "hand_r"
    upper_arm_body: str = "humerus_r"
    forearm_body: str = "ulna_r"
    foot_l_body: str = "calcn_l"
    foot_r_body: str = "calcn_r"
    toes_l_body: str = "toes_l"
    toes_r_body: str = "toes_r"
    # bowlrl joint name -> (foreign joint name, sign). Unlisted joints are not tracked.
    joints: Dict[str, Tuple[str, float]] = field(default_factory=lambda: {
        "lumbar_flex": ("lumbar_extension", -1.0), "lumbar_lat": ("lumbar_bending", -1.0),
        "lumbar_rot": ("lumbar_rotation", 1.0),
        "shoulder_r_flex": ("arm_flex_r", 1.0), "shoulder_r_abd": ("arm_add_r", -1.0),
        "shoulder_r_rot": ("arm_rot_r", 1.0), "elbow_r_flex": ("elbow_flex_r", 1.0),
        "wrist_r_flex": ("wrist_flex_r", 1.0),
        "shoulder_l_flex": ("arm_flex_l", 1.0), "shoulder_l_abd": ("arm_add_l", -1.0),
        "shoulder_l_rot": ("arm_rot_l", 1.0), "elbow_l_flex": ("elbow_flex_l", 1.0),
        "wrist_l_flex": ("wrist_flex_l", 1.0),
        "hip_l_flex": ("hip_flexion_l", 1.0), "hip_l_abd": ("hip_adduction_l", -1.0),
        "hip_l_rot": ("hip_rotation_l", 1.0), "knee_l_flex": ("knee_angle_l", 1.0),
        "ankle_l_pf": ("ankle_angle_l", -1.0),
        "hip_r_flex": ("hip_flexion_r", 1.0), "hip_r_abd": ("hip_adduction_r", -1.0),
        "hip_r_rot": ("hip_rotation_r", 1.0), "knee_r_flex": ("knee_angle_r", 1.0),
        "ankle_r_pf": ("ankle_angle_r", -1.0),
    })
    # joints to drive with torque motors because the model has no muscles for them
    arm_torque_joints: Dict[str, float] = field(default_factory=dict)   # foreign joint -> gear (N m)
    ball_offset_in_hand: Tuple[float, float, float] = (0.03, 0.0, -0.08)
    heel_offset: Tuple[float, float, float] = (-0.02, 0.0, -0.02)
    toe_offset: Tuple[float, float, float] = (0.05, 0.0, -0.02)


def build_merged_spec(myo_xml: str, mp: MyoMapping, add_scene: bool = True) -> mujoco.MjSpec:
    """Load the foreign model and add the ball (welded to the hand), pitch markers, target,
    hand/heel/toe sites and optional torque motors."""
    spec = mujoco.MjSpec.from_file(myo_xml)
    names = {b.name for b in spec.bodies}
    for need in (mp.hand_body, mp.pelvis_body, mp.torso_body, mp.foot_l_body, mp.foot_r_body):
        if need not in names:
            raise ValueError(f"body '{need}' not found in {myo_xml}; adjust MyoMapping")

    def body(n):
        return next(b for b in spec.bodies if b.name == n)

    site_names = {s.name for s in spec.sites}

    def add_site(b, name, pos):
        if name not in site_names:   # keep sites the model already defines (e.g. bowler.xml)
            b.add_site(name=name, pos=list(pos), size=[0.01, 0, 0])

    hand = body(mp.hand_body)
    add_site(hand, "ball_site", mp.ball_offset_in_hand)
    add_site(hand, "wrist_r_site", [0, 0, 0])
    add_site(body(mp.upper_arm_body), "shoulder_r_site", [0, 0, 0])
    add_site(body(mp.forearm_body), "elbow_r_site", [0, 0, 0])
    for side, fb, tb in (("l", mp.foot_l_body, mp.toes_l_body), ("r", mp.foot_r_body, mp.toes_r_body)):
        add_site(body(fb), f"heel_{side}_site", mp.heel_offset)
        add_site(body(tb) if tb in names else body(fb), f"toe_{side}_site", mp.toe_offset)
    add_site(body(mp.pelvis_body), "pelvis_site", [0, 0, 0])

    if "ball" in names:
        return spec  # model already carries the ball/target/creases (bowler.xml)
    ball = spec.worldbody.add_body(name="ball", pos=[0, 0, 1.0])
    ball.add_freejoint(name="ball_free")
    ball.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.036, 0, 0], mass=0.156,
                  rgba=[0.55, 0.05, 0.05, 1], contype=2, conaffinity=1)
    ball.add_site(name="ball_centre", pos=[0, 0, 0], size=[0.005, 0, 0])
    spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name="ball_grip", objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1=mp.hand_body, name2="ball")
    for b in (mp.hand_body, mp.forearm_body, mp.upper_arm_body, mp.torso_body):
        spec.add_exclude(bodyname1=b, bodyname2="ball")

    tgt = spec.worldbody.add_body(name="target", pos=[12.5, 0.1, 0.0])
    tgt.add_geom(name="target_disc", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.15, 0.002, 0],
                 rgba=[1, 0.2, 0.1, 0.6], contype=0, conaffinity=0)
    if add_scene:
        has_ground = any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.worldbody.geoms)
        if not has_ground:
            spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[40, 20, 0.1],
                                    friction=[1.2, 0.01, 0.0005])
        spec.worldbody.add_geom(name="bowler_popping_crease", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, 0.002],
                                size=[0.01, 1.32, 0.001], contype=0, conaffinity=0)
    for jn, gear in mp.arm_torque_joints.items():
        spec.add_actuator(name=f"a_{jn}", trntype=mujoco.mjtTrn.mjTRN_JOINT, target=jn, gear=[gear, 0, 0, 0, 0, 0],
                          ctrlrange=[-1, 1], ctrllimited=True)
    return spec


def build_index(m: mujoco.MjModel, mp: MyoMapping) -> M.ModelIndex:
    def jid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    def bid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    def gid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    def sid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
    jq, jd, rng = [], [], []
    for n in M.JOINT_NAMES:
        fj = mp.joints.get(n)
        j = jid(fj[0]) if fj else -1
        jq.append(int(m.jnt_qposadr[j]) if j >= 0 else -1)
        jd.append(int(m.jnt_dofadr[j]) if j >= 0 else -1)
        rng.append(m.jnt_range[j] if j >= 0 else np.array([-np.pi, np.pi]))
    root = jid(mp.root_joint)
    ball = jid("ball_free")
    ground = gid("ground")
    if ground < 0:  # first plane geom
        ground = int(np.where(m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)[0][0])
    foot_l, foot_r = bid(mp.foot_l_body), bid(mp.foot_r_body)
    return M.ModelIndex(
        joint_qpos=np.array(jq), joint_dof=np.array(jd),
        root_qpos=int(m.jnt_qposadr[root]), root_dof=int(m.jnt_dofadr[root]),
        ball_qpos=int(m.jnt_qposadr[ball]), ball_dof=int(m.jnt_dofadr[ball]),
        ball_body=bid("ball"), ball_geom=gid("ball_geom"),
        hand_r_body=bid(mp.hand_body), torso_body=bid(mp.torso_body), pelvis_body=bid(mp.pelvis_body),
        upper_arm_r_body=bid(mp.upper_arm_body), forearm_r_body=bid(mp.forearm_body),
        foot_l_geoms=[int(g) for g in np.where(m.geom_bodyid == foot_l)[0]],
        foot_r_geoms=[int(g) for g in np.where(m.geom_bodyid == foot_r)[0]],
        foot_l_body=foot_l, foot_r_body=foot_r, ground_geom=ground,
        ball_site=sid("ball_site"), wrist_r_site=sid("wrist_r_site"), shoulder_r_site=sid("shoulder_r_site"),
        elbow_r_site=sid("elbow_r_site"), heel_l_site=sid("heel_l_site"), toe_l_site=sid("toe_l_site"),
        heel_r_site=sid("heel_r_site"), toe_r_site=sid("toe_r_site"),
        target_body=bid("target"), target_mocap=-1,
        weld_eq=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "ball_grip"),
        elbow_r_flex_qpos=jq[M.JOINT_NAMES.index("elbow_r_flex")],
        knee_l_flex_qpos=jq[M.JOINT_NAMES.index("knee_l_flex")],
        knee_r_flex_qpos=jq[M.JOINT_NAMES.index("knee_r_flex")],
        hip_l_flex_qpos=jq[M.JOINT_NAMES.index("hip_l_flex")],
        actuator_gear=m.actuator_gear[:, 0].copy(), joint_range=np.array(rng),
    )


class MyoBowlingEnv(FastBowlingEnv):
    """Muscle-driven variant. Action = [muscle excitations in 0..1 | torque motors in -1..1 | release]."""

    def __init__(self, myo_xml: str, mapping: Optional[MyoMapping] = None, config: Optional[EnvConfig] = None,
                 render_mode: Optional[str] = None):
        gym.Env.__init__(self)
        self.cfg = config or EnvConfig()
        self.cfg.torque_velocity = False     # muscles have their own force-velocity relation
        self.render_mode = render_mode
        self.mp = mapping or MyoMapping()
        self.spec = build_merged_spec(myo_xml, self.mp)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.idx = build_index(self.model, self.mp)
        self.frame_skip = max(1, int(round(1.0 / (self.model.opt.timestep * self.cfg.control_hz))))
        self.dt = self.model.opt.timestep * self.frame_skip
        self.body_weight = float(mujoco.mj_getTotalmass(self.model) * 9.81)
        self.omega_max = np.full(N_JOINTS, 1e9)
        self.is_muscle = np.array([self.model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_MUSCLE
                                   for i in range(self.model.nu)])
        self.tracked = self.idx.joint_qpos >= 0
        ref = self.cfg.reference
        self.ref = (ReferenceMotion.synthetic(self.cfg.run_up_speed) if ref in ("synthetic", "auto")
                    else ReferenceMotion.from_csv(ref))
        self.reference_source = ref
        self.track_weight = self.cfg.reward.track
        self.assist = self.cfg.root_assist
        self._reset_episode_state()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu + 1,), dtype=np.float32)
        obs_dim = self._observe().shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self._renderer = None
        self.np_random, _ = gym.utils.seeding.np_random(self.cfg.seed)

    # muscles take excitation in [0,1]; map the policy's [-1,1] output
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        ctrl = a[:-1].copy()
        ctrl[self.is_muscle] = 0.5 * (ctrl[self.is_muscle] + 1.0)
        self._pending_ctrl = ctrl
        return FastBowlingEnv.step(self, np.concatenate([np.zeros(N_JOINTS), [a[-1]]]))

    def _apply_action(self, torque_cmd):
        self.data.ctrl[:] = self._pending_ctrl

    def _effort(self, torque_cmd):
        return float(np.mean(self.data.ctrl ** 2))

    def _observe(self) -> np.ndarray:
        base = FastBowlingEnv._observe(self)
        d = self.data
        muscle = np.concatenate([d.act if self.model.na else np.zeros(0),
                                 d.actuator_length, d.actuator_velocity * 0.1]).astype(np.float32)
        return np.concatenate([base, muscle])

    def _tracking_reward(self) -> float:
        if self.track_weight <= 0 or not self.tracked.any():
            return 0.0
        s = self.ref.sample(self.t)
        d, idx = self.data, self.idx
        q = np.array([d.qpos[a] * self.mp.joints[n][1] if a >= 0 else 0.0
                      for n, a in zip(M.JOINT_NAMES, idx.joint_qpos)])
        dq = (q - s["joints"])[self.tracked]
        dq = (dq + np.pi) % (2 * np.pi) - np.pi
        return 0.7 * float(np.exp(-2.0 * np.mean(dq ** 2))) + 0.3 * float(
            np.exp(-20.0 * (d.qpos[idx.root_qpos + 2] - s["root_pos"][2]) ** 2))


