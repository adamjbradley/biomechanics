# Capture pipeline (Phase 1): real bowler -> reference motion + validation data

Everything here is open source. Two routes, both ending in an OpenSim IK `.mot` file that
`opensim_to_reference.py` converts into `data/reference/<subject>_reference.csv`.

## Route A — OpenCap (easiest; 2-5 iPhones/iPads)

1. Create an account at https://app.opencap.ai, follow the in-app calibration (checkerboard)
   and neutral-pose steps. Place cameras at ~45°/135° to the delivery direction, 3-5 m
   from the crease, at hip height; record at **60 fps minimum, 120 fps or 240 fps if the
   devices support it** (the delivery stride is ~0.3 s).
2. Record 6-12 deliveries per session, radar speed and landing spot noted for each
   (`data/validation/<subject>_radar.csv`, columns: `trial, speed_kmh, landing_x_m, landing_y_m`).
3. Download the session (OpenCap gives `OpenSimData/Kinematics/<trial>_ik.mot` plus the
   scaled model `.osim`). The OpenCap processing utilities
   (https://github.com/stanfordnmbl/opencap-processing) can also run the
   dynamics/muscle-driven simulation of each trial, which is a second, independent
   estimate of joint torques to compare with the RL model.
4. Convert: 
   `python -m bowlrl.capture.opensim_to_reference OpenSimData/Kinematics/trial03_ik.mot --out data/reference/subject01_reference.csv --bfc 1.42 --ffc 1.56 --release 1.68`
   BFC/FFC/release times are read off the video (OpenCap's visualiser) or detected with
   `--auto-events` (check them!).

Validated accuracy (Uhlrich et al. 2023): sagittal-plane lower-limb angles within ~1-5°
of marker-based capture; hip rotation and pelvis obliquity are the noisiest, which is where
the shoulder-hip separation of a bowling action lives. Use 4+ cameras and treat
transverse-plane values as ±5-8°.

## Route B — Pose2Sim (any synchronised cameras, fully local)

```
pip install pose2sim
# follow https://github.com/perfanalytics/pose2sim — calibration, pose (RTMPose),
# triangulation, filtering, then OpenSim scaling + IK via Pose2Sim.kinematics()
python -m bowlrl.capture.opensim_to_reference pose2sim_project/opensim/trial03.mot --out data/reference/subject01_reference.csv --auto-events
```
Sports2D (same author) gives 2D sagittal angles from a single phone for quick checks of
front-knee angle and release height but not a 3D reference.

## Route C — marker-based lab (gold standard)

Export marker trajectories, scale the OpenSim model, run IK in the OpenSim GUI or
`opensim-cmd run-tool`, then convert the `.mot` as above. Force plates under the front foot
give the GRF and, through OpenSim ID, lumbar loads to validate `bowlrl.sim.metrics`.

## Subject-specific model

- Height/mass -> `configs/subject_example.yaml` (`subject_height_m`, `subject_mass_kg`).
- Strength -> isometric dynamometry per joint -> `joint_torques` in the same file. Without
  dynamometry, `strength_scale` sets a global multiplier.
- For the muscle-driven Phase 3 model, `bowlrl/myo/README.md` describes porting the scaled
  OpenCap `.osim` into MuJoCo with MyoConverter.

## What to log for validation (Phase 4)

Per delivery: radar speed (your radar project), landing x/y (overhead phone + pitch markings
every 1 m), FFC front-knee angle from side-on video. `scripts/validate_radar.py` compares
the policy's predicted speed/landing distribution with these.
