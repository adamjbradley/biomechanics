"""Compare simulated deliveries against measured radar / landing data (Phase 4).

    python scripts/validate_radar.py --sim runs/subject01/eval/episodes.csv \
        --real data/validation/subject01_radar.csv

real CSV columns: trial, speed_kmh, landing_x_m, landing_y_m  (landing optional)
Reports bias, RMSE and a Welch t-test on speed, plus the landing-error distributions, and
flags whether the simulation's predicted technique change is larger than the measurement
noise (otherwise it is not a coachable cue yet).
"""
import argparse, csv, json, sys
import numpy as np
from scipy import stats


def load(path, key):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "nan")]
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--speed-col-sim", default="release_speed_kmh")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sim = load(a.sim, a.speed_col_sim)
    real = load(a.real, "speed_kmh")
    out = {"n_sim": int(len(sim)), "n_real": int(len(real)),
           "sim_speed_mean": float(sim.mean()), "sim_speed_sd": float(sim.std(ddof=1)) if len(sim) > 1 else 0.0,
           "real_speed_mean": float(real.mean()), "real_speed_sd": float(real.std(ddof=1)) if len(real) > 1 else 0.0}
    out["bias_kmh"] = out["sim_speed_mean"] - out["real_speed_mean"]
    t, p = stats.ttest_ind(sim, real, equal_var=False)
    out["welch_t"], out["welch_p"] = float(t), float(p)
    # a technique cue is only coachable if the predicted gain exceeds the radar's own scatter
    out["min_detectable_gain_kmh"] = float(1.96 * out["real_speed_sd"] / np.sqrt(max(len(real), 1)))
    try:
        sx, sy = load(a.sim, "landing_x_m"), load(a.sim, "landing_y_m")
        rx, ry = load(a.real, "landing_x_m"), load(a.real, "landing_y_m")
        out["sim_landing_x_mean"], out["real_landing_x_mean"] = float(sx.mean()), float(rx.mean())
        out["sim_landing_sd_m"] = float(np.sqrt(sx.var(ddof=1) + sy.var(ddof=1)))
        out["real_landing_sd_m"] = float(np.sqrt(rx.var(ddof=1) + ry.var(ddof=1)))
    except Exception:
        pass
    print(json.dumps(out, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
