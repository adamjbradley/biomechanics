"""DEP-RL training hook for the muscle-driven env (Phase 3).

DEP-RL (Schumacher et al., ICLR 2023) adds Differential Extrinsic Plasticity exploration to
TD3/MPO and is the exploration method that works on MyoSuite's overactuated models.

    pip install deprl
    python scripts/train_deprl.py --myo-xml myo_subject01/model_cvt3.xml --reference data/reference/subject01_reference.csv

This registers `MyoBowling-v0` and writes a deprl JSON config; deprl's own trainer runs it.
"""
import argparse, json, os, sys
import gymnasium as gym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--myo-xml", required=True)
    ap.add_argument("--reference", default="synthetic")
    ap.add_argument("--out", default="runs/deprl_myo")
    ap.add_argument("--steps", type=int, default=5_000_000)
    a = ap.parse_args()
    from bowlrl.envs.bowling_env import EnvConfig
    from bowlrl.myo.myo_bowling_env import MyoBowlingEnv, MyoMapping
    cfg = EnvConfig(reference=a.reference)
    gym.register(id="MyoBowling-v0", entry_point=lambda: MyoBowlingEnv(a.myo_xml, MyoMapping(), cfg))
    os.makedirs(a.out, exist_ok=True)
    conf = {
        "tonic": {"header": "import deprl, gymnasium as gym\nfrom bowlrl.myo.myo_bowling_env import MyoBowlingEnv, MyoMapping\nfrom bowlrl.envs.bowling_env import EnvConfig",
                  "agent": "deprl.custom_agents.dep_factory(3, deprl.custom_mpo_torch.TunedMPO())(replay=deprl.replays.buffers.Buffer(return_steps=3, batch_size=256, steps_between_batches=1000, batch_iterations=30, steps_before_batches=2e5))",
                  "environment": f"deprl.environments.Gym('MyoBowling-v0')",
                  "test_environment": None, "trainer": f"deprl.custom_trainer.Trainer(steps=int({a.steps}), epoch_steps=int(2e5), save_steps=int(1e6))",
                  "before_training": "", "after_training": "", "parallel": 8, "sequential": 2, "seed": 0,
                  "name": "bowling_deprl", "environment_name": "MyoBowling-v0", "checkpoint": "last", "path": a.out},
        "DEP": {"test_episode_every": 3, "kappa": 1169.7, "tau": 40, "bias_rate": 0.002, "buffer_size": 200,
                "s4avg": 2, "time_dist": 5, "normalization": "independent", "sensor_delay": 1, "regularization": 32,
                "with_learning": True, "q_norm_selector": "l2", "intervention_length": 5, "intervention_proba": 0.0004,
                "test_episode_every": 3, "force_scale": 0.0}}
    path = os.path.join(a.out, "deprl_config.json")
    with open(path, "w") as f:
        json.dump(conf, f, indent=2)
    print(f"wrote {path}\nrun:  python -m deprl.main {path}")


if __name__ == "__main__":
    main()
