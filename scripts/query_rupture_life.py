#!/usr/bin/env python3
"""
query_rupture_life.py — quick CLI using models/calibrations/in718_lmp.yaml

Usage:
  python3 scripts/query_rupture_life.py --sigma 700 --T 650C
  python3 scripts/query_rupture_life.py --sigma 500 --T 923K
"""
import argparse, yaml, math

def load_params(path="models/calibrations/in718_lmp.yaml"):
    d = yaml.safe_load(open(path))
    p = d["parameters"]
    return p["alpha"], p["beta"], p["C"], p.get("scale", 1e-3)

def predict_tr_h(sigma, T_K, alpha, beta, C, scale):
    log10_sigma = math.log10(sigma)
    log10_tr = (log10_sigma - alpha) / (beta * scale * T_K) - C
    return 10 ** log10_tr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, required=True, help="stress [MPa]")
    ap.add_argument("--T", required=True, help="temperature like '650C' or '923K'")
    args = ap.parse_args()

    t = args.T.strip().upper()
    if t.endswith("C"): T_K = float(t[:-1]) + 273.15
    elif t.endswith("K"): T_K = float(t[:-1])
    else: raise SystemExit("Temperature must end with C or K, e.g., 650C or 923K")

    alpha, beta, C, scale = load_params()
    tr_h = predict_tr_h(args.sigma, T_K, alpha, beta, C, scale)
    print(f"Predicted rupture life: {tr_h:.2f} h ({tr_h/24:.2f} days) at {args.sigma} MPa and {T_K:.1f} K")

if __name__ == "__main__":
    main()
