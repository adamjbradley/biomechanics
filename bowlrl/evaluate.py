"""Evaluate a trained policy (or the PD/CEM baselines) and extract coaching cues.

    python -m bowlrl.evaluate --run runs/ppo_base --episodes 50 --video runs/ppo_base/delivery.mp4
    python -m bowlrl.evaluate --baseline pd --episodes 5          # reference tracking baseline
    python -m bowlrl.evaluate --baseline cem --episodes 1         # CEM open-loop optimum

Writes <out>/episodes.csv (one row per delivery), <out>/summary.json (mean/std/CI of every
metric, legal-delivery rate, target hit rate) and <out>/cues.md (technique cues at FFC and
release, compared with the reference action the policy was trained from, in the language of
the Loughborough/ECB findings).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np
import yaml

from bowlrl.envs.bowling_env import EnvConfig, FastBowlingEnv
from bowlrl.train import METRIC_KEYS, load_config

CUE_KEYS = [
    ("front_knee_ffc_deg", "Front knee flexion at FFC (deg)", "lower = straighter, braced front leg (Felton 2023: more extended front knee)"),
    ("front_knee_release_deg", "Front knee flexion at release (deg)", "near 0 = leg stays braced through release"),
    ("bowling_arm_elevation_ffc_deg", "Bowling upper-arm elevation at FFC (deg above horizontal)", "lower/negative = delayed bowling arm (Felton: delay arm circumduction)"),
    ("hip_shoulder_separation_ffc_deg", "Hip-shoulder separation at FFC (deg)", ">30 with mixed alignment is the lumbar-risk pattern (Portus 2004)"),
    ("shoulder_alignment_ffc_deg", "Shoulder alignment at FFC (deg, 0 front-on, 90 side-on)", "classifies front-on / semi-open / side-on"),
    ("trunk_flexion_release_deg", "Trunk flexion at release (deg)", "more flexion = more trunk contribution to release speed"),
    ("trunk_lat_flexion_release_deg", "Trunk lateral flexion at release (deg, +ve = towards bowling arm)", "contralateral lateral flexion loads the non-bowling-side pars"),
    ("elbow_extension_deg", "Elbow extension from upper-arm-horizontal to release (deg)", "ICC legal limit 15"),
    ("release_height_m", "Release height (m)", ""),
    ("release_angle_deg", "Release angle (deg, +ve up)", ""),
    ("ffc_to_release_s", "FFC to release (s)", "shorter = faster ground-contact phase"),
    ("stride_length_m", "Delivery stride length (m)", ""),
    ("peak_grf_front_bw", "Peak front-foot GRF (BW)", "elite 5-9 BW (Hurrion 2000, Worthington 2013)"),
    ("peak_lumbar_comp_bw", "Peak lumbar compression (BW)", "model-based proxy; compare across techniques, not absolute"),
    ("peak_lumbar_shear_bw", "Peak lumbar shear (BW)", ""),
]


def run_policy_episodes(env: FastBowlingEnv, act_fn, episodes: int, targets: Optional[List[List[float]]] = None,
                        render: bool = False) -> (List[dict], list):
    rows, frames = [], []
    for ep in range(episodes):
        if targets:
            env.cfg.target_xy = tuple(targets[ep % len(targets)])
        obs, info = env.reset(seed=1000 + ep)
        done, total = False, 0.0
        while not done:
            a = act_fn(obs, env)
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
            if render:
                frames.append(env.render())
        row = {k: info.get(k) for k in METRIC_KEYS}
        row.update({"episode": ep, "return": total, "released": info.get("released"),
                    "fail_reason": info.get("fail_reason"), "target_x": info["target_x"], "target_y": info["target_y"]})
        rows.append(row)
    return rows, frames


def summarise(rows: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    n = len(rows)
    released = [r for r in rows if r.get("released")]
    legal = [r for r in released if not r.get("no_ball") and not r.get("illegal_elbow")]
    out["n_episodes"] = n
    out["release_rate"] = len(released) / max(n, 1)
    out["legal_rate"] = len(legal) / max(n, 1)
    out["hit_rate_0p5m"] = float(np.mean([r["landing_error_m"] <= 0.5 for r in legal])) if legal else 0.0
    out["hit_rate_1m"] = float(np.mean([r["landing_error_m"] <= 1.0 for r in legal])) if legal else 0.0
    for k in METRIC_KEYS + ["return"]:
        vals = np.array([r[k] for r in rows if r.get(k) is not None and not (isinstance(r[k], float) and np.isnan(r[k]))], dtype=float)
        if len(vals):
            out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                      "ci95": float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                      "min": float(vals.min()), "max": float(vals.max()), "n": int(len(vals))}
    return out


def write_cues(summary: dict, ref_summary: Optional[dict], path: str, title: str):
    lines = [f"# Technique cues — {title}", ""]
    lines.append(f"Legal deliveries: {100 * summary['legal_rate']:.0f}%  |  "
                 f"release speed {summary.get('release_speed_kmh', {}).get('mean', float('nan')):.1f} ± "
                 f"{summary.get('release_speed_kmh', {}).get('std', float('nan')):.1f} km/h  |  "
                 f"landing error {summary.get('landing_error_m', {}).get('mean', float('nan')):.2f} m  |  "
                 f"within 0.5 m: {100 * summary['hit_rate_0p5m']:.0f}%")
    lines.append("")
    lines.append("| Cue | Policy | Reference | Delta | Reading |")
    lines.append("|---|---:|---:|---:|---|")
    for key, label, note in CUE_KEYS:
        p = summary.get(key, {}).get("mean")
        r = ref_summary.get(key, {}).get("mean") if ref_summary else None
        if p is None:
            continue
        d = (p - r) if (r is not None) else None
        lines.append(f"| {label} | {p:.1f} | {'' if r is None else f'{r:.1f}'} | {'' if d is None else f'{d:+.1f}'} | {note} |")
    lines.append("")
    lines.append("Interpretation: a delta is a *direction of travel* for this bowler's technique, as predicted by the "
                 "simulation. Validate one cue at a time against radar speed before coaching it (see README, Phase 4).")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def make_video(frames, path: str, fps: int = 50):
    try:
        import imageio
        imageio.mimsave(path, frames, fps=fps, macro_block_size=None)
        print(f"[eval] video -> {path}")
    except Exception as e:  # rendering needs EGL/OSMesa; fail softly
        print(f"[eval] video skipped: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="training run dir (model.zip + vecnormalize.pkl)")
    ap.add_argument("--model", default=None, help="explicit model path (default <run>/best/best_model.zip or <run>/model.zip)")
    ap.add_argument("--baseline", default=None, choices=[None, "pd", "cem"], help="evaluate a baseline instead of a policy")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--override", default=None)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--video", default=None, help="mp4 path (needs an offscreen GL backend, e.g. MUJOCO_GL=egl)")
    ap.add_argument("--targets", default="12.5,0.10;11.0,0.10;13.5,0.10;12.5,-0.15;12.5,0.35",
                    help="semicolon-separated target x,y list cycled through the episodes")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    env_cfg = EnvConfig.from_dict(cfg["env"])
    env_cfg.init_noise_pos = 0.0
    env_cfg.init_noise_vel = 0.0
    env_cfg.obs_noise = 0.0
    env_cfg.action_noise = 0.0
    env_cfg.root_assist = 0.0
    env_cfg.target_random_box = (0.0, 0.0)
    targets = [[float(v) for v in t.split(",")] for t in args.targets.split(";") if t.strip()]
    render = args.video is not None
    env = FastBowlingEnv(env_cfg, render_mode="rgb_array" if render else None)
    env.set_curriculum(1.0)

    out = args.out or (os.path.join(args.run, "eval") if args.run else f"runs/eval_{args.baseline or 'policy'}")
    os.makedirs(out, exist_ok=True)

    # ---- reference (PD tracking) summary for the cue comparison
    from bowlrl.imitation.pd_controller import PDGains, ReferencePDController
    ref_env = FastBowlingEnv(env_cfg)
    ref_env.cfg.root_assist = 1.0
    ref_env.assist = 1.0
    ref_env.cfg.release_mode = "auto"
    pd = ReferencePDController(ref_env, PDGains(kp=6.0, kd=0.1))
    ref_rows, _ = run_policy_episodes(ref_env, lambda o, e: pd.act(), 3, targets)
    ref_summary = summarise(ref_rows)

    if args.baseline == "pd":
        env.cfg.root_assist = 1.0
        env.assist = 1.0
        env.cfg.release_mode = "auto"
        pd2 = ReferencePDController(env, PDGains(kp=6.0, kd=0.1))
        rows, frames = run_policy_episodes(env, lambda o, e: pd2.act(), args.episodes, targets, render)
        title = "PD reference tracking (with root assist)"
    elif args.baseline == "cem":
        from bowlrl.optim.cem import CEMConfig, knots_to_ctrl, release_time
        npz = np.load("data/reference/cem_best.npz")
        cem_cfg = CEMConfig(knots=int(npz["knots"]), horizon=float(npz["horizon"]))
        params = npz["params"]
        env.cfg.release_mode = "policy"
        t_rel = release_time(params, cem_cfg)
        def act(o, e):
            return np.concatenate([knots_to_ctrl(params, cem_cfg, e.t), [1.0 if e.t >= t_rel - 1e-9 else -1.0]])
        rows, frames = run_policy_episodes(env, act, args.episodes, targets, render)
        title = "CEM open-loop optimum"
    else:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        model_path = args.model or (os.path.join(args.run, "best", "best_model.zip")
                                    if os.path.exists(os.path.join(args.run, "best", "best_model.zip"))
                                    else os.path.join(args.run, "model.zip"))
        vn_path = os.path.join(args.run, "vecnormalize.pkl")
        model = PPO.load(model_path, device="cpu")
        venv = DummyVecEnv([lambda: env])
        if os.path.exists(vn_path):
            venv = VecNormalize.load(vn_path, venv)
            venv.training = False
            venv.norm_reward = False
        def act(o, e):
            o_n = venv.normalize_obs(o[None, :]) if isinstance(venv, VecNormalize) else o[None, :]
            a, _ = model.predict(o_n, deterministic=True)
            return a[0]
        rows, frames = run_policy_episodes(env, act, args.episodes, targets, render)
        title = f"PPO policy {model_path}"

    with open(os.path.join(out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows)
    summary["reference"] = ref_summary
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    write_cues(summary, ref_summary, os.path.join(out, "cues.md"), title)
    if render and frames:
        make_video(frames, args.video)

    sp = summary.get("release_speed_kmh", {})
    er = summary.get("landing_error_m", {})
    print(f"[eval] {title}\n  episodes {summary['n_episodes']}  released {100*summary['release_rate']:.0f}%  "
          f"legal {100*summary['legal_rate']:.0f}%\n  speed {sp.get('mean', float('nan')):.1f} ± {sp.get('std', float('nan')):.1f} km/h  "
          f"landing error {er.get('mean', float('nan')):.2f} m  within 0.5 m {100*summary['hit_rate_0p5m']:.0f}%\n  -> {out}/")


if __name__ == "__main__":
    main()
