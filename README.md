# bowlrl — RL optimisation of a cricket fast-bowling delivery (speed × accuracy)

Open-source, end-to-end: **capture a bowler → scale a musculoskeletal model → optimise the
delivery in simulation (evolutionary + PPO) → extract coaching cues → validate with radar**.

Everything runs on MuJoCo + Gymnasium + Stable-Baselines3 (Phase 2) and MyoSuite /
MyoConverter (Phase 3), with OpenCap / Pose2Sim / OpenSim for capture (Phase 1).

```
bowlrl/
  models/bowler.xml          16-segment torque-driven bowler (23 DoF) + ball + pitch, MJCF
  envs/bowling_env.py        FastBowlingEnv (Gymnasium): reward, legality, lumbar/GRF, release, flight
  sim/aero.py                ball flight: drag + Magnus + pitch bounce (RK4)
  sim/metrics.py             GRF, lumbar compression/shear, alignment, trunk angles, cues
  sim/scaling.py             subject anthropometrics + strength (dynamometry) -> model
  imitation/reference.py     reference motion (synthetic keyframes | captured | CEM optimum)
  imitation/pd_controller.py reference-tracking PD (+ inverse-dynamics feed-forward) baseline
  optim/cem.py               Felton-style technique optimisation: CEM over activation profiles
  train.py                   PPO with imitation + root-assist curriculum, TB logging of biomech metrics
  evaluate.py                policy/baseline evaluation, cues.md, episodes.csv, summary.json, video
  capture/                   OpenSim .mot -> reference CSV; OpenCap / Pose2Sim instructions
  myo/                       Phase 3 muscle-driven env on MyoSuite / MyoConverter models
configs/base.yaml            every tunable (env, reward, PPO, curriculum)
configs/subject_example.yaml subject-specific overrides (height, mass, joint torques, reference)
scripts/validate_radar.py    sim vs measured speed/landing (bias, RMSE, Welch t, detectable gain)
scripts/train_deprl.py       DEP-RL hook for the muscle model
tests/                       17 tests: physics, rules, flight, reference, importer, CEM
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate          # Python 3.10-3.13
pip install -e .                                           # mujoco, gymnasium, stable-baselines3, torch, scipy, pyyaml, tensorboard, imageio
pytest -q                                                  # ~10 s
# optional: pip install myosuite deprl pose2sim           # Phase 3 / capture
```
Videos need an offscreen GL backend: `export MUJOCO_GL=egl` (NVIDIA) or `osmesa`.

## Docker

```bash
docker compose build                                   # CPU image (~2.5 GB); EGL inside for videos
docker compose run --rm test                           # 17 tests
WORKERS=12 docker compose run --rm cem                 # technique search -> ./data/reference/
WORKERS=12 RUN=ppo_base docker compose run --rm train  # PPO -> ./runs/ppo_base/
docker compose run --rm eval                           # cues.md + delivery.mp4
docker compose up -d tensorboard                       # http://localhost:6006
docker compose --profile gpu run --rm train-gpu        # CUDA image (needs nvidia-container-toolkit)
```
`runs/`, `data/` and `configs/` are bind-mounted, so results and your subject configs live on
the host. The physics is CPU-bound: set `WORKERS` to your core count minus one. A GPU only
speeds up the PPO network updates unless you move to MJX (see `bowlrl/myo/README.md`).

## Run the whole pipeline

```bash
# 0. baselines: does the model bowl at all?
python -m bowlrl.evaluate --baseline pd --episodes 5         # PD tracking of the reference (~50 km/h: a placeholder action)

# 1. technique optimisation (Felton-style forward-dynamics optimum) -> data/reference/cem_reference.csv
python -m bowlrl.optim.cem --iters 150 --pop 96 --elites 10 --workers 16   # ~1-2 h on 16 cores
python -m bowlrl.evaluate --baseline cem --episodes 1

# 2. PPO: refine for speed + accuracy + legality + lumbar load, imitating the CEM/captured reference
python -m bowlrl.train --config configs/base.yaml --run runs/ppo_base             # 20M steps ≈ 3-6 h on 16 cores
tensorboard --logdir runs/ppo_base/tb                                              # bowl/*, eval_bowl/*, reward_terms/*

# 3. evaluate + cues
python -m bowlrl.evaluate --run runs/ppo_base --episodes 100 --video runs/ppo_base/delivery.mp4
cat runs/ppo_base/eval/cues.md

# 4. subject-specific: capture, scale, retrain, validate
python -m bowlrl.capture.opensim_to_reference trial03_ik.mot --out data/reference/subject01_reference.csv --bfc 1.42 --ffc 1.56 --release 1.68
python -m bowlrl.train --config configs/base.yaml --override configs/subject_example.yaml --run runs/subject01 --resume runs/ppo_base/model.zip
python -m bowlrl.evaluate --run runs/subject01 --episodes 100
python scripts/validate_radar.py --sim runs/subject01/eval/episodes.csv --real data/validation/subject01_radar.csv
```

## What the agent optimises

Episode = one delivery stride from back-foot contact (BFC); the agent controls 23 joint
torque generators at 100 Hz (Hill-type torque-velocity limits, peak torques from
dynamometry) plus a release signal. The ball is welded to the hand until release; an
analytic flight model (drag, Magnus, bounce) gives the pitching point.

```
r = 10·(v_release/40 m/s)                 speed
  +  6·exp(-(landing error/0.6 m)²)       accuracy to a (randomised) target length/line
  + w_track·imitation(reference)          annealed 2.0 -> 0.2 over training (DeepMimic-style)
  - 0.02·effort - lumbar overload penalty (compression > 8 BW, shear > 2.5 BW; model-based proxy)
  - 5 if no-ball (front foot over the popping crease) or illegal elbow (>15° extension from
      upper-arm-horizontal to release) - 3 if full toss - 5 if no release / fall
```
A **root-assist curriculum** (virtual spring-damper on the pelvis, annealed 0.6 → 0) lets
PPO learn the limb coordination before it has to solve balance; the reference is first
made dynamically consistent by the CEM technique search (`optim/cem.py`), which is the
same forward-dynamics optimisation the Loughborough/ECB group used (Felton et al. 2015-2025,
*J Biomech* 158:111765) but in 3D with contacts.

Observation (93-d): root height/orientation/velocity, joint angles and velocities, ball-held
flag, time/phase, target offset, hand position/velocity, front-foot contact, and the
reference pose one step ahead.

## Outputs you can act on

`evaluate.py` writes `cues.md`: front-knee angle at FFC and release, bowling-arm elevation
at FFC (arm delay), hip–shoulder separation and alignment (front-on/semi-open/side-on),
trunk flexion/lateral flexion at release, elbow extension, release height/angle, FFC→release
time, peak front-foot GRF and lumbar loads — each as *policy vs reference (delta)*, with the
literature reading. Treat deltas as a direction of travel, test **one cue at a time** with
the radar, and use `validate_radar.py` to check the predicted gain exceeds the radar's
scatter (`min_detectable_gain_kmh`).

## Accuracy: what is and isn't validated

- Physics: MuJoCo 1 kHz, elliptic friction cones, implicit-fast integrator; anthropometrics
  de Leva (1996) at 1.85 m / 85 kg scaled per subject; peak torques literature-range (edit
  `joint_torques` from dynamometry — this is the single biggest accuracy lever).
- Ball flight: C_D 0.45, Magnus saturating at C_L 0.28, pitch COR 0.35 (Mehta; Sayers &
  Hill). No seam swing (hook: `aero.extra_force`).
- Lumbar load = MuJoCo interaction force across the 3-DoF lumbar joint — a comparative
  proxy, not L4/L5 disc pressure. Compare techniques; don't quote it as absolute.
- Capture: OpenCap sagittal-plane angles ±1-5° vs marker-based; transverse plane ±5-8°.
- The torque-driven optimum (Phase 2) ignores muscle co-contraction and bi-articular
  coupling; Phase 3 (`bowlrl/myo`) adds muscles for the final subject-specific answer.
- Sim-to-real: speeds reached in simulation are upper bounds for *that body and strength*;
  the coaching value is the ranked cue list, verified with radar.

## State of this build (what has been run here)

The sandbox this was built in had 2 CPU cores and no GPU/GL, so only smoke runs were
possible: the PD baseline completes a legal delivery; a 6k-step PPO run trains and
evaluates end-to-end; a short CEM search (≤50 iterations, pop 48) climbs from ~25 to
~50 km/h legal deliveries before being stopped. The full searches above (CEM 150×96, PPO
20M steps) are what produce elite-range technique; expect the CEM optimum to reach
100-130 km/h for the template body if the torque generators are realistic, and the PPO
policy to trade a few km/h for accuracy and lumbar load.

## References

Felton PJ et al. (2023) J Biomech 158:111765 — optimal FFC technique, 10 elite bowlers.
Felton PJ (2015) PhD thesis, Loughborough — factors limiting fast bowling (torque-driven simulation).
Worthington PJ, King MA, Ranson CA (2013) J Appl Biomech 29:78 — technique and ball speed.
Portus MR et al. (2004) Sports Biomech 3:263 — technique factors and lumbar injury.
Ranson C et al. (2008) J Sports Sci 26:267 — lumbar loading and injury in fast bowlers.
Caggiano V et al. (2022) MyoSuite, arXiv:2205.13600.  Uhlrich SD et al. (2023) OpenCap, PLoS Comput Biol.
Pagnon D et al. (2022) Pose2Sim, JOSS 7(77).  Schumacher P et al. (2023) DEP-RL, ICLR.
Peng XB et al. (2018) DeepMimic, SIGGRAPH.  Mehta RD (2005) Sports Eng 8:181 — cricket ball aerodynamics.
