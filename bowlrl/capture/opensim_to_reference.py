"""Convert an OpenSim inverse-kinematics result (.mot) of a real delivery into a bowlrl
reference motion CSV.

Works with any pipeline that ends in OpenSim IK on a full-body model with the standard
coordinate names (Rajagopal 2016 / gait2392 / OpenCap's LaiUhlrich2022 model):

    OpenCap  (2+ phones)      -> https://app.opencap.ai  -> download *_ik.mot
    Pose2Sim (2+ cameras)     -> Pose2Sim.kinematics()   -> *.mot
    Marker-based (Vicon etc.) -> OpenSim IK tool         -> *.mot

    python -m bowlrl.capture.opensim_to_reference session_ik.mot \
        --out data/reference/subject01_reference.csv --bfc 1.42 --ffc 1.56 --release 1.68

Events (seconds in the .mot time base) can be given explicitly or auto-detected from the
foot trajectories (--auto-events). The output is trimmed to [BFC - 0.02 s, release + 0.2 s]
and re-timed so BFC = 0.

Frame conventions
    OpenSim: x forward, y up, z right        bowlrl/MuJoCo: x forward, y left, z up
    Angles: OpenSim degrees; sign conventions mapped per joint below.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from bowlrl.imitation.reference import ROOT_COLS, Events
from bowlrl.sim.metrics import JOINT_NAMES

# (opensim_coordinate, sign) for each bowlrl joint. Sign converts OpenSim's positive
# direction to bowlrl's (see bowler.xml header). Missing coordinates are zero-filled.
COORD_MAP: Dict[str, Tuple[str, float]] = {
    "lumbar_flex": ("lumbar_extension", -1.0),      # OpenSim: extension +ve
    "lumbar_lat": ("lumbar_bending", -1.0),         # OpenSim: bending to the right +ve? (model dependent)
    "lumbar_rot": ("lumbar_rotation", 1.0),
    "shoulder_l_flex": ("arm_flex_l", 1.0),
    "shoulder_l_abd": ("arm_add_l", -1.0),          # OpenSim: adduction +ve
    "shoulder_l_rot": ("arm_rot_l", 1.0),
    "elbow_l_flex": ("elbow_flex_l", 1.0),
    "wrist_l_flex": ("wrist_flex_l", 1.0),
    "shoulder_r_flex": ("arm_flex_r", 1.0),
    "shoulder_r_abd": ("arm_add_r", -1.0),
    "shoulder_r_rot": ("arm_rot_r", 1.0),
    "elbow_r_flex": ("elbow_flex_r", 1.0),
    "wrist_r_flex": ("wrist_flex_r", 1.0),
    "hip_l_flex": ("hip_flexion_l", 1.0),
    "hip_l_abd": ("hip_adduction_l", -1.0),
    "hip_l_rot": ("hip_rotation_l", 1.0),
    "knee_l_flex": ("knee_angle_l", 1.0),
    "ankle_l_pf": ("ankle_angle_l", -1.0),          # OpenSim: dorsiflexion +ve
    "hip_r_flex": ("hip_flexion_r", 1.0),
    "hip_r_abd": ("hip_adduction_r", -1.0),
    "hip_r_rot": ("hip_rotation_r", 1.0),
    "knee_r_flex": ("knee_angle_r", 1.0),
    "ankle_r_pf": ("ankle_angle_r", -1.0),
}


def read_mot(path: str) -> Tuple[List[str], np.ndarray, bool]:
    """Return (column names, data (N, C), in_degrees)."""
    in_degrees = True
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines) and not lines[i].strip().lower().startswith("endheader"):
        if lines[i].lower().startswith("indegrees"):
            in_degrees = lines[i].split("=")[1].strip().lower() == "yes"
        i += 1
    header = lines[i + 1].strip().split()
    data = np.array([[float(v) for v in l.split()] for l in lines[i + 2:] if l.strip()])
    return header, data, in_degrees


def auto_events(t: np.ndarray, cols: Dict[str, int], data: np.ndarray, foot_cols=("pelvis_tx",)) -> Events:
    """Crude event detection from the ankle-angle/knee patterns is unreliable; we use the
    pelvis forward velocity minima (braking at BFC and FFC) and the bowling-arm flexion
    peak for release. Inspect and override with explicit --bfc/--ffc/--release if needed."""
    tx = data[:, cols["pelvis_tx"]]
    vx = np.gradient(tx, t)
    ax = np.gradient(vx, t)
    # the two largest decelerations in the last 0.6 s are BFC (first) and FFC (second)
    n = len(t)
    win = t > (t[-1] - 0.8)
    idxs = np.where(win)[0]
    order = idxs[np.argsort(ax[idxs])][:6]
    order = np.sort(order)
    # merge neighbours
    ev = []
    for i in order:
        if not ev or t[i] - t[ev[-1]] > 0.06:
            ev.append(i)
    if len(ev) < 2:
        raise ValueError("could not auto-detect BFC/FFC; pass --bfc/--ffc")
    bfc, ffc = t[ev[-2]], t[ev[-1]]
    arm = data[:, cols["arm_flex_r"]] if "arm_flex_r" in cols else data[:, cols["arm_flex_l"]]
    after = t > ffc
    rel = t[after][np.argmax(np.abs(np.gradient(arm, t))[after])]
    return Events(t_bfc=float(bfc), t_ffc=float(ffc), t_release=float(rel), t_end=float(min(rel + 0.2, t[-1])))


def convert(mot_path: str, out_path: str, events: Optional[Events] = None, pre: float = 0.02, post: float = 0.20,
            unwrap_arm: bool = True) -> Events:
    header, data, in_deg = read_mot(mot_path)
    cols = {n: i for i, n in enumerate(header)}
    t = data[:, cols["time"]]
    if not in_deg:
        for n in header:
            if n not in ("time", "pelvis_tx", "pelvis_ty", "pelvis_tz"):
                data[:, cols[n]] = np.degrees(data[:, cols[n]])
    if events is None:
        events = auto_events(t, cols, data)
    sel = (t >= events.t_bfc - pre) & (t <= events.t_release + post)
    d = data[sel]
    tt = t[sel] - events.t_bfc

    # root: OpenSim pelvis_tx/ty/tz (x fwd, y up, z right) -> bowlrl (x fwd, y left, z up)
    root_pos = np.column_stack([d[:, cols["pelvis_tx"]], -d[:, cols["pelvis_tz"]], d[:, cols["pelvis_ty"]]])
    # pelvis orientation: OpenSim rotation order ZXY (tilt about z, list about x, rotation about y)
    tilt = d[:, cols["pelvis_tilt"]]        # +ve anterior tilt (about OpenSim z)
    lst = d[:, cols["pelvis_list"]]         # about OpenSim x
    rot = d[:, cols["pelvis_rotation"]]     # about OpenSim y (vertical)
    quats = []
    for a, b, c in zip(tilt, lst, rot):
        r_os = R.from_euler("ZXY", [a, b, c], degrees=True)
        # change of basis OpenSim -> bowlrl: columns map x->x, y->z, z->-y
        C = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        m = C @ r_os.as_matrix() @ C.T
        q = R.from_matrix(m).as_quat()  # xyzw
        quats.append(np.roll(q, 1))
    root_quat = np.array(quats)
    root_pos[:, :2] -= root_pos[0, :2]      # BFC pelvis at x=y=0 of the capture; shift to the crease below
    root_pos[:, 0] += -1.15                  # assume BFC pelvis 1.15 m behind the popping crease (edit if measured)

    joints = np.zeros((len(tt), len(JOINT_NAMES)))
    missing = []
    for j, n in enumerate(JOINT_NAMES):
        osn, sgn = COORD_MAP[n]
        if osn in cols:
            joints[:, j] = sgn * d[:, cols[osn]]
        else:
            missing.append(osn)
    if unwrap_arm:
        for n in ("shoulder_r_flex", "shoulder_l_flex"):
            j = JOINT_NAMES.index(n)
            joints[:, j] = np.degrees(np.unwrap(np.radians(joints[:, j])))
    ev = Events(0.0, events.t_ffc - events.t_bfc, events.t_release - events.t_bfc, float(tt[-1]))
    hdr = ["time"] + ROOT_COLS + JOINT_NAMES
    with open(out_path, "w") as f:
        f.write("#" + json.dumps({"events": ev.__dict__, "source": mot_path, "missing_coords": missing}) + "\n")
        f.write(",".join(hdr) + "\n")
        np.savetxt(f, np.column_stack([tt, root_pos, root_quat, joints]), delimiter=",", fmt="%.6f")
    if missing:
        print(f"[capture] zero-filled coordinates not in the .mot: {missing}")
    print(f"[capture] wrote {out_path}: {len(tt)} frames, events {ev}")
    return ev


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bfc", type=float)
    ap.add_argument("--ffc", type=float)
    ap.add_argument("--release", type=float)
    ap.add_argument("--auto-events", action="store_true")
    args = ap.parse_args()
    ev = None
    if not args.auto_events:
        if None in (args.bfc, args.ffc, args.release):
            ap.error("pass --bfc --ffc --release (seconds in the .mot) or --auto-events")
        ev = Events(args.bfc, args.ffc, args.release, args.release + 0.2)
    convert(args.mot, args.out, ev)


if __name__ == "__main__":
    main()
