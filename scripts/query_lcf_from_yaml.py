#!/usr/bin/env python3

"""
Usage: 

python scripts/query_lcf_from_yaml.py --eps_total 0.005
python scripts/query_lcf_from_yaml.py --eps_total 0.0035  # expect larger life

"""
import argparse, math, yaml
from pathlib import Path
import numpy as np

def _load_yaml(p: Path):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def _select_params(y, T_K=None):
    # supports either global "params" or "params_per_temperature"
    if "params" in y:
        P = y["params"]
        return {
            "log10_sigmaf_over_E": P.get("log10_sigmaf_over_E"),
            "b": P.get("b"),
            "log10_epsf": P.get("log10_epsf"),
            "c": P.get("c"),
        }
    groups = y.get("params_per_temperature")
    if not groups or T_K is None:
        raise ValueError("Temperature grouping present; please pass --T_K.")
    # nearest T
    g = min(groups, key=lambda r: abs(float(r["T_K"]) - float(T_K)))
    return {
        "log10_sigmaf_over_E": g.get("log10_sigmaf_over_E"),
        "b": g.get("b"),
        "log10_epsf": g.get("log10_epsf"),
        "c": g.get("c"),
    }

def eps_total(twoN, p):
    e = 0.0
    if p.get("log10_sigmaf_over_E") is not None and p.get("b") is not None:
        e += (10.0 ** p["log10_sigmaf_over_E"]) * (twoN ** p["b"])
    if p.get("log10_epsf") is not None and p.get("c") is not None:
        e += (10.0 ** p["log10_epsf"]) * (twoN ** p["c"])
    return e

def invert_for_Nf(eps_target, p, N_lo=10.0, N_hi=1e9):
    # bracket in log-space for robustness
    lo, hi = math.log10(2.0 * N_lo), math.log10(2.0 * N_hi)
    for _ in range(80):  # bisection
        mid = 0.5 * (lo + hi)
        val = eps_total(10 ** mid, p)
        if val > eps_target:
            lo = mid
        else:
            hi = mid
    twoN = 10 ** (0.5 * (lo + hi))
    return twoN / 2.0

def main():
    ap = argparse.ArgumentParser(description="Query Coffin–Manson life from YAML.")
    ap.add_argument("--yaml", type=Path, default=Path("models/calibrations/fatigue/in718_lcf_coffin_manson.yaml"))
    ap.add_argument("--eps_total", type=float, required=True, help="Total strain amplitude Δε/2 (unitless, e.g. 0.005)")
    ap.add_argument("--T_K", type=float, help="Temperature in K (needed if YAML is per-temperature).")
    args = ap.parse_args()

    y = _load_yaml(args.yaml)
    p = _select_params(y, T_K=args.T_K)
    Nf = invert_for_Nf(args.eps_total, p)
    # Print also the elastic/plastic split at Nf for insight
    e_el = (10**p["log10_sigmaf_over_E"])*(2*Nf)**p["b"] if p.get("log10_sigmaf_over_E") is not None else 0.0
    e_pl = (10**p["log10_epsf"])*(2*Nf)**p["c"] if p.get("log10_epsf") is not None else 0.0
    print(f"Δε/2 = {args.eps_total:g} → Nf ≈ {Nf:.3g} cycles")
    print(f"  elastic part ≈ {e_el:.3g}, plastic part ≈ {e_pl:.3g}")

if __name__ == "__main__":
    main()
