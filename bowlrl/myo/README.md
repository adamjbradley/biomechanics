# Phase 3 — muscle-driven bowler (MyoSuite / MyoConverter)

`MyoBowlingEnv` wraps any MuJoCo musculoskeletal model with the same reward, legality
rules, ball flight and lumbar/GRF metrics as the torque-driven env. Muscles replace the
torque generators, so strength, fatigue and co-contraction costs become physiological.

## Getting a full-body muscle model (all open source)

Option 1 — MyoSuite assets (Apache-2.0). `pip install myosuite`; the first environment
creation downloads the `myo_sim` model repository (git submodule, needs internet):

```python
from myosuite.utils import gym
gym.make("myoLegWalk-v0")   # triggers the asset download
```
Then `myosuite/simhive/myo_sim/` holds `leg/myolegs.xml` (80 leg muscles),
`arm/myoarm.xml` (63 arm muscles), `torso/` and `body/` (MyoSkeleton, 324 DoF). Combine
with MuJoCo's `<include>` or edit `MyoMapping` to the names in the model you choose. For a
first muscle run use `envs/myo/assets/leg/myolegs_with_torso.xml` and drive the arms with
torque motors via `MyoMapping.arm_torque_joints`.

Option 2 — subject-specific, from your own capture (recommended for Phase 4 validation):
OpenCap gives a scaled OpenSim model (`LaiUhlrich2022_scaled.osim`); convert it with
MyoConverter (https://github.com/MyoHub/myoconverter):

```
pip install myoconverter   # or clone; needs OpenSim python bindings
python -m myoconverter.O2MPipeline LaiUhlrich2022_scaled.osim ./myo_subject01 \
    --convert_steps 1 2 3 --muscle_list all
```
The output `myo_subject01/<name>_cvt3.xml` is a MuJoCo model with muscle actuators,
validated step-wise against OpenSim (joint moment arms, muscle lengths).

## Running

```python
from bowlrl.myo.myo_bowling_env import MyoBowlingEnv, MyoMapping
from bowlrl.envs.bowling_env import EnvConfig
mp = MyoMapping(hand_body="hand_r", upper_arm_body="humerus_r", forearm_body="ulna_r",
                foot_l_body="calcn_l", foot_r_body="calcn_r", toes_l_body="toes_l", toes_r_body="toes_r",
                torso_body="torso", pelvis_body="pelvis", root_joint="root")
env = MyoBowlingEnv("myo_subject01/model_cvt3.xml", mp, EnvConfig(reference="data/reference/subject01_reference.csv"))
```
Action = `[muscle excitations (policy output mapped to 0..1) | torque motors | release]`.
Observation = torque-env observation + muscle activations, lengths and velocities.

Exploration with 80-300 muscles is hard for Gaussian PPO. Use the policy from Phase 2 as a
teacher (track its joint trajectory with a tracking reward) and/or DEP-RL
(`pip install deprl`, https://github.com/martius-lab/depRL) or Lattice exploration, both
built for MyoSuite. `bowlrl/train.py` works unchanged for the Myo env if you register it
with Gymnasium and set `train.algo: ppo`; `scripts/train_deprl.py` shows the DEP-RL call.

## Throughput

MuJoCo CPU: ~100-300 steps/s per core for a 300-muscle model. For millions of steps use
MJX (`mujoco.mjx`) or MuJoCo Warp on a GPU: thousands of parallel envs, 5-100x faster.
The env logic here is NumPy; porting the step/reward to JAX follows MyoSuite's MJX examples.
