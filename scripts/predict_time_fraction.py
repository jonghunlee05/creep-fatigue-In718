#!/usr/bin/env python3
"""
Predict life via Robinson time-fraction for Inconel 718.

Damage model
------------
D(N) = N / Nf  +  N * d_c  = 1

Fatigue Nf:
  - LCF (Coffin–Manson + Basquin elastic): Δε/2 = (σ'f/E)*(2N)^b + ε'f*(2N)^c  (b,c < 0)
  - HCF (Basquin):          log10 Nf = a - k * log10(σ_a[MPa])

Creep per-cycle damage d_c:
  - 'rupture':  d_c = t_hold / t_r(σ, T)           (t_r from LMP or Manson–Haferd YAML)
  - 'rate':     d_c = (ε̇(σ,T) * t_hold) / ε_crit  (ε̇ from Norton YAML; ε_crit user-supplied)

Solution:
  N = 1 / ( 1/Nf + d_c ).  (Constant-amplitude, constant T and dwell.)

Usage (examples)
----------------
LCF + rupture-based creep:
  python scripts/predict_time_fraction.py --T_K 977 --sigma_MPa 550 --eps_total 0.004 \
    --t_hold_s 5 --creep_damage rupture

HCF (Basquin only):
  python scripts/predict_time_fraction.py --T_K 300 --sigma_a_MPa 500 --fatigue_model hcf

Rate-based creep (choose eps_crit):
  python scripts/predict_time_fraction.py --T_K 977 --sigma_MPa 550 --eps_total 0.004 \
    --t_hold_s 5 --creep_damage rate --epsilon_crit 0.02
"""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------- helpers --------------------
def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _load_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return yaml.safe_load(f)

def _nearest_params_by_T(y: Dict[str, Any], T_K: float) -> Dict[str, Any]:
    groups = y.get("params_per_temperature")
    if not groups:
        return y.get("params", {})
    # choose the record whose T_K is closest to requested T
    rec = min(groups, key=lambda r: abs(float(r["T_K"]) - float(T_K)))
    # normalize to same keys as "params"
    out = {k: rec[k] for k in rec if k not in ("T_K", "N")}
    return out

# -------------------- fatigue models --------------------
@dataclass
class CMParams:
    log10_sigmaf_over_E: Optional[float]
    b: Optional[float]
    log10_epsf: Optional[float]
    c: Optional[float]

def _load_cm_params(y_path: Path, T_K: Optional[float]) -> CMParams:
    y = _load_yaml(y_path)
    P = _nearest_params_by_T(y, T_K) if T_K is not None else y.get("params", {})
    return CMParams(
        log10_sigmaf_over_E=P.get("log10_sigmaf_over_E"),
        b=P.get("b"),
        log10_epsf=P.get("log10_epsf"),
        c=P.get("c"),
    )

def _cm_eps_total(twoN: np.ndarray, p: CMParams) -> np.ndarray:
    eps = np.zeros_like(twoN, dtype=float)
    if p.log10_sigmaf_over_E is not None and p.b is not None:
        eps += (10.0 ** p.log10_sigmaf_over_E) * (twoN ** p.b)
    if p.log10_epsf is not None and p.c is not None:
        eps += (10.0 ** p.log10_epsf) * (twoN ** p.c)
    return eps

def _invert_cm_for_Nf(eps_total: float, p: CMParams,
                      N_lo: float = 5.0, N_hi: float = 1e10) -> float:
    """Invert Δε/2 -> Nf using bisection on log10(2N)."""
    lo, hi = math.log10(2.0 * N_lo), math.log10(2.0 * N_hi)
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        val = _cm_eps_total(10.0 ** mid, p)
        if val > eps_total:
            lo = mid
        else:
            hi = mid
    twoN = 10.0 ** (0.5 * (lo + hi))
    return twoN / 2.0

@dataclass
class BasquinParams:
    a: float
    k: float

def _load_basquin_params(y_path: Path, T_K: Optional[float]) -> BasquinParams:
    y = _load_yaml(y_path)
    if "params_per_temperature" in y and T_K is not None:
        P = _nearest_params_by_T(y, T_K)
    else:
        P = y.get("params", {})
    return BasquinParams(a=float(P["a"]), k=float(P["k"]))

def _basquin_Nf(sig_a_MPa: float, p: BasquinParams) -> float:
    return 10.0 ** (p.a - p.k * math.log10(sig_a_MPa))

# -------------------- creep: Norton rate --------------------
@dataclass
class NortonParams:
    A: float
    n: float
    Q: float
    R: float

def _load_norton_params(y_path: Path) -> NortonParams:
    y = _load_yaml(y_path)
    P = y["params"]
    R = P.get("R_J_per_molK", 8.314462618)
    return NortonParams(A=float(P["A"]), n=float(P["n"]), Q=float(P["Q_J_per_mol"]), R=float(R))

def _norton_rate(stress_MPa: float, T_K: float, p: NortonParams) -> float:
    return p.A * (stress_MPa ** p.n) * math.exp(-p.Q / (p.R * T_K))

# -------------------- rupture: LMP / Manson–Haferd --------------------

def _get_param_block(y: Dict[str, Any]) -> Dict[str, Any]:
    """Support both {'params':...} and your {'parameters':...} schema."""
    if isinstance(y, dict):
        if "params" in y and isinstance(y["params"], dict):
            return y["params"]
        if "parameters" in y and isinstance(y["parameters"], dict):
            return y["parameters"]
    return {}

def _rupture_time_LMP_from_params(params: Dict[str, Any], sigma_MPa: float, T_K: float) -> float:
    """
    LMP forms used in your fitter (t_r in HOURS):
      linear   : log10(sigma) = alpha + beta * P
      quadratic: log10(sigma) = a + b*P + c*P^2
      with P = scale * T * (C + log10 t_r), where 'scale' is stored in YAML (usually 1e-3)
    Returns rupture time in **seconds** (converted from hours).
    """
    scale = float(params.get("scale", 1e-3))
    log10sigma = math.log10(sigma_MPa)

    # Linear
    if all(k in params for k in ("alpha", "beta", "C")):
        alpha, beta, C = float(params["alpha"]), float(params["beta"]), float(params["C"])
        P = (log10sigma - alpha) / beta
        log10tr_h = (P / (scale * T_K)) - C
        return (10.0 ** log10tr_h) * 3600.0  # hours -> seconds

    # Quadratic
    if all(k in params for k in ("a", "b", "c", "C")):
        a, b, c, C = float(params["a"]), float(params["b"]), float(params["c"]), float(params["C"])
        # Solve a + bP + cP^2 = log10sigma  → cP^2 + bP + (a - log10sigma)=0
        A = c; B = b; CC = a - log10sigma
        disc = B*B - 4*A*CC
        if disc < 0:
            raise ValueError("LMP quadratic: negative discriminant; check parameters.")
        # choose the physically reasonable root (larger P typically)
        P = max(( -B + math.sqrt(disc) ) / (2*A), ( -B - math.sqrt(disc) ) / (2*A))
        log10tr_h = (P / (scale * T_K)) - C
        return (10.0 ** log10tr_h) * 3600.0  # hours -> seconds

    raise ValueError("LMP params missing required keys.")

def _rupture_time_MH_from_params(params: Dict[str, Any], sigma_MPa: float, T_K: float) -> float:
    """
    Your Manson–Haferd (t_r in HOURS):
      log10 sigma = A + B * ((T - T*) * (log10 t_r + C*))
      → log10 t_r = ((log10 sigma - A)/B)/(T - T*) - C*
    Returns **seconds**.
    """
    # Accept your exact keys
    if not all(k in params for k in ("A", "B", "T_star", "C_star")):
        raise ValueError("Manson–Haferd parameters incomplete (need A, B, T_star, C_star).")
    A, B, Tstar, Cstar = float(params["A"]), float(params["B"]), float(params["T_star"]), float(params["C_star"])
    denom = B * (T_K - Tstar)
    if abs(denom) < 1e-12:
        raise ValueError("Manson–Haferd: T close to T_star; numerical issue.")
    log10tr_h = ( (math.log10(sigma_MPa) - A) / denom ) - Cstar
    return (10.0 ** log10tr_h) * 3600.0  # hours -> seconds

def _rupture_time_from_yaml(y_path: Path, sigma_MPa: float, T_K: float) -> float:
    """
    Reads your rupture YAMLs (in718_rupture_*.yaml) and returns t_r in **seconds**.
    Supports models: 'LMP_linear', 'LMP_quadratic', 'MansonHaferd'.
    """
    y = _load_yaml(y_path)
    model_raw = str(y.get("model", "")).strip()
    model = model_raw.replace("-", "_").replace(" ", "_").lower()
    params = _get_param_block(y)

    if model in ("lmp_linear", "lmp_quadratic"):
        return _rupture_time_LMP_from_params(params, sigma_MPa, T_K)

    if model in ("mansonhaferd", "manson_haferd", "mansonhaferd"):
        return _rupture_time_MH_from_params(params, sigma_MPa, T_K)

    # If model name is missing in 'best' file, try to infer from keys.
    if {"alpha","beta","C"} <= set(params.keys()) or {"a","b","c","C"} <= set(params.keys()):
        return _rupture_time_LMP_from_params(params, sigma_MPa, T_K)
    if {"A","B","T_star","C_star"} <= set(params.keys()):
        return _rupture_time_MH_from_params(params, sigma_MPa, T_K)

    raise ValueError(f"Unrecognized rupture YAML schema in {y_path}: model='{model_raw}', keys={list(params.keys())}")

# -------------------- figure --------------------
def _save_breakdown_figure(N: float, Nf: float, dc_per_cycle: float, out_png: Path) -> None:
    _ensure_dir(out_png)
    Df = N / Nf
    Dc = N * dc_per_cycle
    # simple stacked bar (they should sum to ~1)
    plt.figure(figsize=(4.2, 3.6), dpi=140)
    plt.bar([0], [Df], label="Fatigue Df = N/Nf")
    plt.bar([0], [Dc], bottom=[Df], label="Creep Dc = N·d_c")
    plt.xticks([0], [f"N ≈ {N:.3g}"])
    plt.ylabel("Damage fraction")
    plt.title("Time-fraction damage breakdown")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def _row_val(d, key, default=None): 
    v = d.get(key, default)
    return default if (v is None or (isinstance(v,float) and (v!=v))) else v

# -------------------- main --------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Predict life via time-fraction (fatigue + creep).")
    # Inputs
    ap.add_argument("--T_K", type=float, help="Temperature [K].")
    ap.add_argument("--sigma_MPa", type=float, required=False,
                    help="Hold stress for creep [MPa] (needed for creep damage).")
    ap.add_argument("--eps_total", type=float, help="Total strain amplitude Δε/2 (LCF).")
    ap.add_argument("--sigma_a_MPa", type=float, help="Stress amplitude σ_a [MPa] (HCF).")
    ap.add_argument("--fatigue_model", choices=["lcf","hcf"], default="lcf")
    ap.add_argument("--t_hold_s", type=float, default=0.0, help="Hold time per cycle [s].")
    ap.add_argument("--creep_damage", choices=["rupture","rate"], default="rupture",
                    help="'rupture' uses tr(σ,T); 'rate' uses ε̇ and ε_crit.")
    ap.add_argument("--epsilon_crit", type=float, default=0.02,
                    help="Critical creep strain for rate-based damage (default 0.02 = 2%).")
    # Calibrations
    ap.add_argument("--norton",  type=Path, default=Path("models/calibrations/creep/in718_norton.yaml"))
    ap.add_argument("--coffin",  type=Path, default=Path("models/calibrations/fatigue/in718_lcf_coffin_manson.yaml"))
    ap.add_argument("--basquin", type=Path, default=Path("models/calibrations/fatigue/in718_hcf_basquin.yaml"))
    ap.add_argument("--rupture", type=Path, default=Path("models/calibrations/rupture/in718_rupture_best.yaml"))
    # Outputs
    ap.add_argument("--fig", type=Path, default=Path("reports/figures/time_fraction/time_fraction_breakdown.png"))
    ap.add_argument("--debug", action="store_true")
    # Batch mode
    ap.add_argument("--csv_in", type=Path, help="Batch input CSV: columns like T_K, eps_total, sigma_MPa, t_hold_s, fatigue_model, sigma_a_MPa, creep_damage, epsilon_crit")
    ap.add_argument("--csv_out", type=Path, help="Where to write results CSV")
    args = ap.parse_args()

    # Batch mode handler
    if args.csv_in:
        df = pd.read_csv(args.csv_in)
        rows = []
        for _, r in df.iterrows():
            T_K = float(_row_val(r, "T_K", args.T_K))
            fatigue = str(_row_val(r, "fatigue_model", args.fatigue_model)).lower()
            eps_total = _row_val(r, "eps_total", None)
            sigma_a = _row_val(r, "sigma_a_MPa", None)
            sigma_hold = _row_val(r, "sigma_MPa", None)
            t_hold_s = float(_row_val(r, "t_hold_s", 0.0))
            creep_mode = str(_row_val(r, "creep_damage", args.creep_damage)).lower()
            eps_crit = float(_row_val(r, "epsilon_crit", args.epsilon_crit))

            # compute N (reuse your existing functions)
            if fatigue == "lcf":
                if eps_total is None: raise SystemExit("CSV row missing eps_total for LCF.")
                cm = _load_cm_params(args.coffin, T_K=T_K)
                Nf = _invert_cm_for_Nf(float(eps_total), cm)
            else:
                if sigma_a is None: raise SystemExit("CSV row missing sigma_a_MPa for HCF.")
                bp = _load_basquin_params(args.basquin, T_K=T_K)
                Nf = _basquin_Nf(float(sigma_a), bp)

            dc = 0.0
            if t_hold_s > 0 and sigma_hold is not None:
                if creep_mode == "rupture":
                    tr = _rupture_time_from_yaml(args.rupture, float(sigma_hold), T_K)
                    dc = t_hold_s / tr
                else:
                    nt = _load_norton_params(args.norton)
                    edot = _norton_rate(float(sigma_hold), T_K, nt)
                    dc = (edot * t_hold_s) / eps_crit

            N = 1.0 / ((1.0/Nf) + dc)
            rows.append({**r.to_dict(), "N_pred_cycles": N, "Nf_fatigue_cycles": Nf, "Df": N/Nf, "Dc": N*dc})

        out = pd.DataFrame(rows)
        if args.csv_out:
            args.csv_out.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(args.csv_out, index=False)
            print(f"Wrote {args.csv_out}")
        else:
            print(out.head())
        return

    if args.T_K is None:
        raise SystemExit("Temperature --T_K is required when not using batch mode (--csv_in).")
    
    T_K = float(args.T_K)

    # --- Fatigue life Nf ---
    if args.fatigue_model == "lcf":
        if args.eps_total is None:
            raise SystemExit("LCF selected: please provide --eps_total (Δε/2).")
        cm = _load_cm_params(args.coffin, T_K=T_K)
        Nf = _invert_cm_for_Nf(args.eps_total, cm)
        fatigue_desc = f"LCF Coffin–Manson (Δε/2={args.eps_total:g})"
    else:
        if args.sigma_a_MPa is None:
            raise SystemExit("HCF selected: please provide --sigma_a_MPa (stress amplitude MPa).")
        bp = _load_basquin_params(args.basquin, T_K=T_K)
        Nf = _basquin_Nf(args.sigma_a_MPa, bp)
        fatigue_desc = f"HCF Basquin (σ_a={args.sigma_a_MPa:g} MPa)"

    # --- Creep per-cycle damage d_c ---
    dc_per_cycle = 0.0
    creep_desc = "none"
    if args.t_hold_s > 0.0:
        if args.creep_damage == "rupture":
            if args.sigma_MPa is None:
                raise SystemExit("rupture damage requires --sigma_MPa.")
            tr = _rupture_time_from_yaml(args.rupture, sigma_MPa=args.sigma_MPa, T_K=T_K)
            if tr <= 0:
                raise SystemExit("Computed rupture time tr <= 0; check rupture YAML/inputs.")
            dc_per_cycle = args.t_hold_s / tr
            creep_desc = f"rupture-based (t_hold={args.t_hold_s:g}s, tr≈{tr:.3g}s)"
        else:  # rate-based
            if args.sigma_MPa is None:
                raise SystemExit("rate damage requires --sigma_MPa.")
            nt = _load_norton_params(args.norton)
            edot = _norton_rate(args.sigma_MPa, T_K, nt)
            dc_per_cycle = (edot * args.t_hold_s) / float(args.epsilon_crit)
            creep_desc = f"rate-based (ε̇≈{edot:.3g}/s, t_hold={args.t_hold_s:g}s, εcrit={args.epsilon_crit:g})"

    # --- Solve N from D(N)=1 ---
    # N = 1 / (1/Nf + dc_per_cycle)
    denom = (1.0 / Nf) + dc_per_cycle
    if denom <= 0:
        raise SystemExit("Non-positive damage rate; check inputs.")
    N = 1.0 / denom

    # --- Report ---
    Df = N / Nf
    Dc = N * dc_per_cycle
    print("=== Time-fraction life prediction ===")
    print(f"T = {T_K:.2f} K")
    if args.sigma_MPa is not None:
        print(f"σ_hold = {args.sigma_MPa:g} MPa")
    print(f"Fatigue: {fatigue_desc}")
    if args.t_hold_s > 0.0:
        print(f"Creep  : {creep_desc}")
    else:
        print("Creep  : no dwell (t_hold=0) → Dc=0")
    print(f"\nPredicted life: N ≈ {N:.6g} cycles")
    print(f"Damage breakdown at N: Df={Df:.4f}, Dc={Dc:.4f} (Df+Dc≈{Df+Dc:.4f})")

    # --- Figure ---
    if args.fig:
        _save_breakdown_figure(N, Nf, dc_per_cycle, args.fig)
        print(f"Figure saved: {args.fig}")

if __name__ == "__main__":
    main()
