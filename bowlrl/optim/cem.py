"""Evolutionary technique optimisation (cross-entropy method) in the simulator.

This is the forward-dynamics optimisation approach of the Loughborough/ECB fast-bowling
work (Felton 2015-2025: torque-driven model + GA over activation timings), re-done with
CEM on our MuJoCo bowler. Each candidate is an open-loop activation profile for the 23
torque generators (K knots, linearly interpolated) plus a release time. The objective is
the environment return with imitation switched off, i.e. release speed + accuracy
- legality - lumbar overload.

The best rollout is written out as a *dynamically consistent* reference motion
(data/reference/cem_reference.csv) that PPO then imitates and refines, and the activation
knots seed the behaviour-cloning warm start.

Usage:
    python -m bowlrl.optim.cem --iters 60 --pop 64 --workers 8 --out data/reference
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from bowlrl.envs.bowling_env import EnvConfig, FastBowlingEnv, N_JOINTS
from bowlrl.imitation.pd_controller import PDGains, ReferencePDController
from bowlrl.imitation.reference import Events, ReferenceMotion
from bowlrl.sim.metrics import JOINT_NAMES


@dataclass
class CEMConfig:
    knots: int = 8
    horizon: float = 0.42
    pop: int = 64
    elites: int = 8
    iters: int = 60
    sigma0: float = 0.35
    sigma_min: float = 0.03
    alpha: float = 0.7           # smoothing of mean/std updates
    t_release_range: Tuple[float, float] = (0.18, 0.40)
    seed: int = 0
    workers: int = 1
    init_from_pd: bool = True
    objective: str = "technique"   # "technique" (speed-first, Felton-style) | "return" (env reward)
    accuracy_weight: float = 0.5   # m^-1 in the technique objective
    dense_weight: float = 0.1      # credit per m/s of peak hand speed (keeps un-released rollouts informative)
    arm_drive_seed: float = 0.0    # 0 = seed from PD tracking only; 1 = add the full arm/trunk drive heuristic


def technique_objective(info: dict, cem_cfg: "CEMConfig") -> float:
    """Release speed (m/s) subject to legality / safety, as penalties.

    Scale: 40 m/s = 40 points, so a 1 m/s gain is worth a 1 m landing-error change at
    accuracy_weight 1.0, and no-ball / illegal elbow / falling cost ~10 m/s each.
    """
    dense = cem_cfg.dense_weight * float(info.get("max_hand_speed_ms", 0.0))   # small credit for hand speed
    if not info.get("released"):
        return -30.0 + dense + (5.0 if info.get("t_ffc_s") else 0.0)
    v = float(info.get("release_speed_ms", 0.0)) + dense
    err = float(info.get("landing_error_m", 5.0))
    j = v - cem_cfg.accuracy_weight * err
    j -= 10.0 * float(info.get("no_ball", 0.0))
    j -= 10.0 * float(info.get("illegal_elbow", 0.0))
    j -= 3.0 * float(info.get("full_toss", 0.0))
    j -= 5.0 * (info.get("fail_reason") == "fell")
    j -= 1.0 * max(0.0, float(info.get("peak_lumbar_comp_bw", 0.0)) - 8.0)
    j -= 2.0 * max(0.0, float(info.get("peak_lumbar_shear_bw", 0.0)) - 2.5)
    return j


def knots_to_ctrl(params: np.ndarray, cfg: CEMConfig, t: float) -> np.ndarray:
    """params: (N_JOINTS*K + 1,) -> ctrl (N_JOINTS,) at time t."""
    K = cfg.knots
    knots = params[:N_JOINTS * K].reshape(N_JOINTS, K)
    grid = np.linspace(0.0, cfg.horizon, K)
    tc = np.clip(t, 0.0, cfg.horizon)
    j = int(np.searchsorted(grid, tc, side="right") - 1)
    j = min(max(j, 0), K - 2)
    w = (tc - grid[j]) / (grid[j + 1] - grid[j])
    return np.clip(knots[:, j] * (1 - w) + knots[:, j + 1] * w, -1.0, 1.0)


def release_time(params: np.ndarray, cfg: CEMConfig) -> float:
    lo, hi = cfg.t_release_range
    return float(np.clip(params[-1], lo, hi))


_ENV: Optional[FastBowlingEnv] = None
_ENV_CFG: Optional[EnvConfig] = None


def _get_env(env_cfg: EnvConfig) -> FastBowlingEnv:
    global _ENV, _ENV_CFG
    if _ENV is None or _ENV_CFG is not env_cfg:
        cfg = EnvConfig(**{**env_cfg.__dict__})
        cfg.release_mode = "policy"
        cfg.init_noise_pos = 0.0
        cfg.init_noise_vel = 0.0
        cfg.reward.track = 0.0
        cfg.reward.track_end = 0.0
        cfg.root_assist = 0.0
        _ENV = FastBowlingEnv(cfg)
        _ENV_CFG = env_cfg
    return _ENV


def rollout_params(params: np.ndarray, cem_cfg: CEMConfig, env_cfg: EnvConfig, record: bool = False):
    env = _get_env(env_cfg)
    obs, info = env.reset(seed=cem_cfg.seed)
    t_rel = release_time(params, cem_cfg)
    total = 0.0
    done = False
    log = []
    while not done:
        ctrl = knots_to_ctrl(params, cem_cfg, env.t)
        rel = 1.0 if env.t >= t_rel - 1e-9 else -1.0
        a = np.concatenate([ctrl, [rel]])
        if record:
            log.append((env.t, env.data.qpos.copy(), a.copy()))
        obs, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
    if record:
        log.append((env.t, env.data.qpos.copy(), a.copy()))
    return total, info, log


def _worker(args):
    params, cem_cfg, env_cfg = args
    total, info, _ = rollout_params(params, cem_cfg, env_cfg)
    score = technique_objective(info, cem_cfg) if cem_cfg.objective == "technique" else total
    return score, float(info.get("release_speed_kmh", 0.0) or 0.0), bool(info.get("released", False))


def pd_initial_mean(cem_cfg: CEMConfig, env_cfg: EnvConfig, arm_drive_from: float = 0.12) -> np.ndarray:
    """Seed the search with the PD controller's actions on the reference, plus a maximal
    over-the-top drive of the bowling arm and trunk from `arm_drive_from` (a crude but fast
    starting technique, ~60-70 km/h)."""
    env = _get_env(env_cfg)
    ctl = ReferencePDController(env, PDGains(kp=4.0, kd=0.08))
    env.reset(seed=cem_cfg.seed)
    ts, acts = [], []
    done = False
    J = {n: i for i, n in enumerate(JOINT_NAMES)}
    while not done and env.t < cem_cfg.horizon + 0.05:
        a = ctl.act()
        if env.t >= arm_drive_from and cem_cfg.arm_drive_seed > 0:
            k = cem_cfg.arm_drive_seed
            drive = {"shoulder_r_flex": -1.0, "shoulder_r_abd": -0.3, "elbow_r_flex": -0.5,
                     "wrist_r_flex": 1.0 if env.t > arm_drive_from + 0.08 else -0.5,
                     "lumbar_flex": 1.0, "lumbar_lat": -0.6}
            for n, v in drive.items():
                a[J[n]] = (1 - k) * a[J[n]] + k * v
        ts.append(env.t)
        acts.append(a[:N_JOINTS])
        _, _, term, trunc, _ = env.step(a)
        done = term or trunc
    ts, acts = np.array(ts), np.array(acts)
    grid = np.linspace(0.0, cem_cfg.horizon, cem_cfg.knots)
    knots = np.zeros((N_JOINTS, cem_cfg.knots))
    for j in range(N_JOINTS):
        knots[j] = np.interp(grid, ts, acts[:, j]) if len(ts) > 1 else 0.0
    return np.concatenate([knots.ravel(), [env.ref.events.t_release]])


def run_cem(cem_cfg: CEMConfig, env_cfg: EnvConfig, out_dir: str, log_every: int = 1):
    rng = np.random.default_rng(cem_cfg.seed)
    dim = N_JOINTS * cem_cfg.knots + 1
    mean = pd_initial_mean(cem_cfg, env_cfg) if cem_cfg.init_from_pd else np.zeros(dim)
    mean[-1] = 0.5 * sum(cem_cfg.t_release_range)
    std = np.full(dim, cem_cfg.sigma0)
    std[-1] = 0.05
    best = (-np.inf, None)
    history = []

    pool = None
    if cem_cfg.workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(cem_cfg.workers)

    t_start = time.time()
    for it in range(cem_cfg.iters):
        pop = mean + std * rng.standard_normal((cem_cfg.pop, dim))
        pop[:, :-1] = np.clip(pop[:, :-1], -1.0, 1.0)
        pop[0] = mean  # always evaluate the mean
        jobs = [(p, cem_cfg, env_cfg) for p in pop]
        results = pool.map(_worker, jobs) if pool else [_worker(j) for j in jobs]
        scores = np.array([r[0] for r in results])
        speeds = np.array([r[1] for r in results])
        order = np.argsort(-scores)
        elite = pop[order[:cem_cfg.elites]]
        new_mean = elite.mean(axis=0)
        new_std = elite.std(axis=0) + 1e-6
        mean = cem_cfg.alpha * new_mean + (1 - cem_cfg.alpha) * mean
        std = np.maximum(cem_cfg.alpha * new_std + (1 - cem_cfg.alpha) * std, cem_cfg.sigma_min)
        if scores[order[0]] > best[0]:
            best = (float(scores[order[0]]), pop[order[0]].copy())
        rec = dict(iter=it, best=float(scores[order[0]]), elite_mean=float(scores[order[:cem_cfg.elites]].mean()),
                   mean_score=float(scores.mean()), best_speed_kmh=float(speeds[order[0]]),
                   best_ever=best[0], elapsed_s=time.time() - t_start)
        history.append(rec)
        if it % log_every == 0:
            print(f"[cem] it {it:3d}  best {rec['best']:7.2f}  elite {rec['elite_mean']:7.2f}  "
                  f"mean {rec['mean_score']:7.2f}  speed {rec['best_speed_kmh']:6.1f} km/h  "
                  f"best_ever {best[0]:7.2f}  ({rec['elapsed_s']:.0f}s)", flush=True)
    if pool:
        pool.close()

    # final best rollout, recorded -> reference motion + metrics
    total, info, log = rollout_params(best[1], cem_cfg, env_cfg, record=True)
    score = technique_objective(info, cem_cfg) if cem_cfg.objective == "technique" else total
    env = _get_env(env_cfg)
    ts = np.array([l[0] for l in log])
    qpos = np.array([l[1] for l in log])
    acts = np.array([l[2] for l in log])
    idx = env.idx
    ref = ReferenceMotion(
        t=ts,
        root_pos=qpos[:, idx.root_qpos:idx.root_qpos + 3],
        root_quat=qpos[:, idx.root_qpos + 3:idx.root_qpos + 7],
        joints_deg=np.degrees(qpos[:, idx.joint_qpos]),
        events=Events(t_bfc=0.0, t_ffc=float(info.get("t_ffc_s") or 0.0),
                      t_release=float(info.get("t_release_s") or release_time(best[1], cem_cfg)),
                      t_end=float(ts[-1])),
        source="cem",
    )
    os.makedirs(out_dir, exist_ok=True)
    ref.to_csv(os.path.join(out_dir, "cem_reference.csv"))
    np.savez(os.path.join(out_dir, "cem_best.npz"), params=best[1], actions=acts, times=ts,
             knots=cem_cfg.knots, horizon=cem_cfg.horizon)
    clean_info = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in info.items() if k != "reward_terms"}
    clean_info["reward_terms"] = info.get("reward_terms", {})
    with open(os.path.join(out_dir, "cem_result.json"), "w") as f:
        json.dump({"score": score, "env_return": total, "info": clean_info, "history": history,
                   "cem_config": cem_cfg.__dict__}, f, indent=2, default=float)
    print(f"[cem] done. objective {score:.2f}  env return {total:.2f}  speed {clean_info.get('release_speed_kmh')} km/h  "
          f"landing err {clean_info.get('landing_error_m')} m  no_ball {clean_info.get('no_ball')}  "
          f"illegal_elbow {clean_info.get('illegal_elbow')}  -> {out_dir}/cem_reference.csv")
    return best, ref, clean_info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="env YAML (configs/base.yaml)")
    ap.add_argument("--out", default="data/reference")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--elites", type=int, default=8)
    ap.add_argument("--knots", type=int, default=8)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objective", default="technique", choices=["technique", "return"])
    ap.add_argument("--arm-drive-seed", type=float, default=0.0)
    args = ap.parse_args()
    env_cfg = EnvConfig()
    if args.config:
        import yaml
        with open(args.config) as f:
            env_cfg = EnvConfig.from_dict(yaml.safe_load(f)["env"])
    cem_cfg = CEMConfig(knots=args.knots, pop=args.pop, elites=args.elites, iters=args.iters,
                        workers=args.workers, seed=args.seed, objective=args.objective,
                        arm_drive_seed=args.arm_drive_seed)
    run_cem(cem_cfg, env_cfg, args.out)


if __name__ == "__main__":
    main()
