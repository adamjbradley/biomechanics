"""PPO training for FastBowlingEnv with imitation + root-assist curriculum.

    python -m bowlrl.train --config configs/base.yaml --run runs/ppo_base
    python -m bowlrl.train --config configs/base.yaml --override configs/subject_example.yaml \
        --run runs/subject01 --resume runs/ppo_base/checkpoints/latest.zip

Outputs (in --run):
    model.zip, vecnormalize.pkl, checkpoints/, tb/ (tensorboard), eval/ (metrics CSV)
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import yaml

from bowlrl.envs.bowling_env import EnvConfig, FastBowlingEnv

METRIC_KEYS = [
    "release_speed_kmh", "landing_error_m", "landing_x_m", "landing_y_m", "no_ball", "illegal_elbow",
    "full_toss", "front_knee_ffc_deg", "front_knee_release_deg", "trunk_flexion_release_deg",
    "trunk_lat_flexion_release_deg", "shoulder_alignment_ffc_deg", "hip_shoulder_separation_ffc_deg",
    "bowling_arm_elevation_ffc_deg", "elbow_extension_deg", "release_height_m", "release_angle_deg",
    "t_ffc_s", "t_release_s", "ffc_to_release_s", "peak_grf_front_bw", "peak_lumbar_comp_bw",
    "peak_lumbar_shear_bw", "stride_length_m", "pelvis_speed_ffc_ms",
]


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str, override: str | None = None) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if override:
        with open(override) as f:
            cfg = deep_update(cfg, yaml.safe_load(f))
    return cfg


def make_env_fn(env_cfg: EnvConfig, seed: int, rank: int, eval_mode: bool = False):
    def _init():
        c = copy.deepcopy(env_cfg)
        c.seed = seed + rank
        if eval_mode:
            c.init_noise_pos = 0.0
            c.init_noise_vel = 0.0
            c.obs_noise = 0.0
            c.action_noise = 0.0
            c.root_assist = c.root_assist_end
        env = FastBowlingEnv(c)
        if eval_mode:
            env.set_curriculum(1.0)
        return env
    return _init


def build_callbacks(cfg: Dict[str, Any], run_dir: str, train_env, eval_env, total_timesteps: int):
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

    tr = cfg["train"]

    class Curriculum(BaseCallback):
        def __init__(self):
            super().__init__()
            self.last = -1.0

        def _on_step(self) -> bool:
            frac = self.num_timesteps / total_timesteps
            a, b = tr["curriculum_start"], tr["curriculum_end"]
            p = float(np.clip((frac - a) / max(b - a, 1e-9), 0.0, 1.0))
            if abs(p - self.last) > 0.01:
                self.training_env.env_method("set_curriculum", p)
                self.last = p
                self.logger.record("curriculum/progress", p)
            return True

    class EpisodeMetrics(BaseCallback):
        """Logs the biomechanical metrics from `info` of finished episodes to tensorboard."""
        def __init__(self, window: int = 200):
            super().__init__()
            self.buf: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS + ["ep_return", "released", "fell"]}
            self.window = window
            self.terms: Dict[str, List[float]] = {}

        def _on_step(self) -> bool:
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                if not done:
                    continue
                self.buf["released"].append(float(info.get("released", False)))
                self.buf["fell"].append(float(info.get("fail_reason") == "fell"))
                if "episode" in info:
                    self.buf["ep_return"].append(float(info["episode"]["r"]))
                for k in METRIC_KEYS:
                    v = info.get(k)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        self.buf[k].append(float(v))
                for k, v in info.get("reward_terms", {}).items():
                    self.terms.setdefault(k, []).append(float(v))
            if self.n_calls % 2000 == 0:
                for k, v in self.buf.items():
                    if v:
                        self.logger.record(f"bowl/{k}", float(np.mean(v[-self.window:])))
                for k, v in self.terms.items():
                    if v:
                        self.logger.record(f"reward_terms/{k}", float(np.mean(v[-self.window:])))
            return True

    class EvalMetrics(EvalCallback):
        """EvalCallback that also writes the biomechanical metrics of the eval episodes."""
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.csv_path = os.path.join(run_dir, "eval", "eval_metrics.csv")
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            self._rows: List[dict] = []

        def _log_success_callback(self, locals_, globals_):
            info = locals_["info"]
            if locals_["done"]:
                row = {k: info.get(k) for k in METRIC_KEYS}
                row["released"] = info.get("released")
                row["timesteps"] = self.num_timesteps
                self._rows.append(row)
            super()._log_success_callback(locals_, globals_)

        def _on_step(self) -> bool:
            n0 = len(self._rows)
            ok = super()._on_step()
            new = self._rows[n0:]
            if new:
                with open(self.csv_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(new[0].keys()))
                    if f.tell() == 0:
                        w.writeheader()
                    w.writerows(new)
                for k in ("release_speed_kmh", "landing_error_m", "no_ball", "illegal_elbow", "peak_lumbar_comp_bw"):
                    vals = [r[k] for r in new if r.get(k) is not None]
                    if vals:
                        self.logger.record(f"eval_bowl/{k}", float(np.mean(vals)))
            return ok

    callbacks = [
        Curriculum(),
        EpisodeMetrics(),
        CheckpointCallback(save_freq=max(tr["checkpoint_freq"] // tr["n_envs"], 1),
                           save_path=os.path.join(run_dir, "checkpoints"), name_prefix="ppo",
                           save_vecnormalize=True),
        EvalMetrics(eval_env, best_model_save_path=os.path.join(run_dir, "best"),
                    log_path=os.path.join(run_dir, "eval"),
                    eval_freq=max(tr["eval_freq"] // tr["n_envs"], 1),
                    n_eval_episodes=tr["eval_episodes"], deterministic=True, render=False),
    ]
    return callbacks


def train(cfg: Dict[str, Any], run_dir: str, resume: str | None = None, total_timesteps: int | None = None):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    tr = cfg["train"]
    env_cfg = EnvConfig.from_dict(cfg["env"])
    total_timesteps = int(total_timesteps or tr["total_timesteps"])
    seed = int(tr.get("seed", 0))
    n_envs = int(tr["n_envs"])

    VecCls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    train_env = VecCls([make_env_fn(env_cfg, seed, i) for i in range(n_envs)])
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=tr["gamma"])
    eval_env = DummyVecEnv([make_env_fn(env_cfg, seed + 10_000, 0, eval_mode=True)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    lr0, lr1 = float(tr["learning_rate"]), float(tr.get("lr_final", tr["learning_rate"]))
    def lr_schedule(progress_remaining: float) -> float:
        return lr1 + (lr0 - lr1) * progress_remaining

    policy_kwargs = dict(net_arch=dict(pi=list(tr["net_arch"]), vf=list(tr["net_arch"])),
                         activation_fn=torch.nn.Tanh, log_std_init=float(tr["log_std_init"]),
                         ortho_init=True)
    device = tr.get("device", "auto")
    if resume:
        model = PPO.load(resume, env=train_env, device=device, learning_rate=lr_schedule)
        vn = os.path.join(os.path.dirname(resume), "vecnormalize.pkl")
        if os.path.exists(vn):
            train_env = VecNormalize.load(vn, train_env.venv)
            model.set_env(train_env)
    else:
        model = PPO("MlpPolicy", train_env, learning_rate=lr_schedule, n_steps=tr["n_steps"],
                    batch_size=tr["batch_size"], n_epochs=tr["n_epochs"], gamma=tr["gamma"],
                    gae_lambda=tr["gae_lambda"], clip_range=tr["clip_range"], ent_coef=tr["ent_coef"],
                    vf_coef=tr["vf_coef"], max_grad_norm=tr["max_grad_norm"], policy_kwargs=policy_kwargs,
                    tensorboard_log=os.path.join(run_dir, "tb"), seed=seed, device=device, verbose=1)

    # keep the eval env's normalisation in sync with training
    class SyncNorm:
        def __init__(self, train_env, eval_env):
            self.t, self.e = train_env, eval_env
        def __call__(self):
            self.e.obs_rms = copy.deepcopy(self.t.obs_rms)
    sync = SyncNorm(train_env, eval_env)
    callbacks = build_callbacks(cfg, run_dir, train_env, eval_env, total_timesteps)
    from stable_baselines3.common.callbacks import BaseCallback
    class _Sync(BaseCallback):
        def _on_step(self):
            if self.n_calls % 500 == 0:
                sync()
            return True
    callbacks.append(_Sync())

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False,
                reset_num_timesteps=resume is None)
    model.save(os.path.join(run_dir, "model.zip"))
    train_env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    with open(os.path.join(run_dir, "train_summary.json"), "w") as f:
        json.dump({"total_timesteps": total_timesteps, "wall_s": time.time() - t0,
                   "reference": str(train_env.get_attr("reference_source")[0])}, f, indent=2)
    print(f"[train] done in {(time.time() - t0) / 60:.1f} min -> {run_dir}/model.zip")
    train_env.close()
    eval_env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--override", default=None)
    ap.add_argument("--run", default="runs/ppo_base")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    if args.n_envs:
        cfg["train"]["n_envs"] = args.n_envs
    train(cfg, args.run, resume=args.resume, total_timesteps=args.timesteps)


if __name__ == "__main__":
    main()
