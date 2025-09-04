#!/usr/bin/env python3
import argparse, numpy as np, subprocess, sys
from pathlib import Path
import matplotlib.pyplot as plt

def predict(T, eps_total, sigma, thold, extra):
    cmd = [sys.executable, "scripts/predict_time_fraction.py",
           "--T_K", str(T), "--eps_total", str(eps_total),
           "--sigma_MPa", str(sigma), "--t_hold_s", str(thold)]
    cmd += extra
    out = subprocess.check_output(cmd, text=True)
    for line in out.splitlines():
        if line.startswith("Predicted life:"):
            n_str = line.split("≈", 1)[1].split("cycles")[0].strip()
            return float(n_str.replace(",", ""))
    raise RuntimeError("Life not found")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T_K", type=float, default=977)
    ap.add_argument("--eps_total", type=float, default=0.004)
    ap.add_argument("--sigma_MPa", type=float, default=550)
    ap.add_argument("--thold_min", type=float, default=0)
    ap.add_argument("--thold_max", type=float, default=60)
    ap.add_argument("--points", type=int, default=25)
    ap.add_argument("--mode", choices=["rupture","rate"], default="rupture")
    ap.add_argument("--epsilon_crit", type=float, default=0.02)
    ap.add_argument("--rupture", type=Path, default=Path("models/calibrations/rupture/in718_rupture_best.yaml"))
    ap.add_argument("--out", type=Path, default=Path("reports/figures/time_fraction/life_vs_dwell.png"))
    args = ap.parse_args()

    extra = ["--creep_damage", args.mode]
    if args.mode == "rate":
        extra += ["--epsilon_crit", str(args.epsilon_crit)]
    if args.rupture:
        extra += ["--rupture", str(args.rupture)]

    th = np.linspace(args.thold_min, args.thold_max, args.points)
    N = [predict(args.T_K, args.eps_total, args.sigma_MPa, t, extra) for t in th]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,4), dpi=140)
    plt.plot(th, N, "-o", ms=3)
    plt.xlabel("Dwell per cycle t_hold [s]")
    plt.ylabel("Life N [cycles]")
    plt.title(f"Life vs dwell (T={args.T_K:g} K, σ={args.sigma_MPa:g} MPa, Δε/2={args.eps_total:g})")
    plt.grid(True, ls=":", lw=0.5)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
