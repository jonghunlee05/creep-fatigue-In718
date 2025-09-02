#!/usr/bin/env python3
"""
Query IN718 HCF life using Basquin YAML (per-temperature).
Usage:
  python scripts/query_hcf_life.py --sigma 500 --T 650C
  python scripts/query_hcf_life.py --sigma 450 --T 873K
  # optional: --yaml models/calibrations/fatigue/in718_hcf_basquin.yaml
"""
import argparse, yaml, math, numpy as np, os

def parse_T(Ts):
    t = Ts.strip().upper()
    if t.endswith("C"): return float(t[:-1])
    if t.endswith("K"): return float(t[:-1]) - 273.15
    raise SystemExit("Temperature must end with C or K (e.g., 650C or 923K)")

def load_params(path):
    d = yaml.safe_load(open(path, "r"))
    return d["parameters"]  # dict of T_C -> {a, k}

def nearest_T(params, T_C):
    Ts = np.array([float(k) for k in params.keys()])
    i = int(np.argmin(np.abs(Ts - T_C)))
    return Ts[i], params[str(Ts[i])]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, required=True, help="stress amplitude [MPa]")
    ap.add_argument("--T", required=True, help="temperature like '650C' or '923K'")
    ap.add_argument("--yaml", default="models/calibrations/fatigue/in718_hcf_basquin.yaml")
    args = ap.parse_args()

    T_C_req = parse_T(args.T)
    params = load_params(args.yaml)
    T_used, pk = nearest_T(params, T_C_req)
    a, k = pk["a"], pk["k"]

    log10N = a - k * math.log10(args.sigma)
    Nf = 10**log10N
    print(f"[Basquin] Using T={T_used:.1f} °C fit (closest to requested {T_C_req:.1f} °C).")
    print(f"Predicted life: Nf ≈ {Nf:.3g} cycles at σ={args.sigma:.1f} MPa.")

if __name__ == "__main__":
    main()
