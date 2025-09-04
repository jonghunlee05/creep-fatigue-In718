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
def _try_keys(d: Dict[str, Any], *candidates: str) -> Optional[float]:
    for k in candidates:
        if k in d: return float(d[k])
    # case-insensitive
    lower = {k.lower(): k for k in d.keys()}
    for k in candidates:
        if k.lower() in lower:
            return float(d[lower[k.lower()]])
    return None

def _rupture_time_LMP(params: Dict[str, Any], sigma_MPa: float, T_K: float) -> float:
    """
    LMP linear:      log10 σ = α + β P,  P = 1e-3 T (C + log10 tr)
    LMP quadratic:   log10 σ = a + b P + c P^2
    """
    log10sigma = math.log10(sigma_MPa)
    # linear form
    alpha = _try_keys(params, "alpha", "A", "a0")
    beta  = _try_keys(params, "beta", "B", "b1")
    C     = _try_keys(params, "C", "C_const", "C0")
    a = _try_keys(params, "a"); b = _try_keys(params, "b"); c = _try_keys(params, "c")

    if alpha is not None and beta is not None and C is not None:
        P = (log10sigma - alpha) / beta
        log10tr = (P / (1e-3 * T_K)) - C
        return 10.0 ** log10tr

    if a is not None and b is not None and c is not None and C is not None:
        # Solve c P^2 + b P + (a - log10sigma) = 0, choose positive/real root with P>0
        A = c; B = b; CC = a - log10sigma
        disc = B*B - 4*A*CC
        if disc < 0:
            raise ValueError("LMP quadratic: negative discriminant; check parameters.")
        r1 = (-B + math.sqrt(disc)) / (2*A)
        r2 = (-B - math.sqrt(disc)) / (2*A)
        P = max(r1, r2)  # pick larger root (usually the physical one)
        log10tr = (P / (1e-3 * T_K)) - C
        return 10.0 ** log10tr

    raise ValueError("Unrecognized LMP parameter set. Expected (alpha,beta,C) or (a,b,c,C).")

def _rupture_time_MH(params: Dict[str, Any], sigma_MPa: float, T_K: float) -> float:
    """
    Manson–Haferd:
      log10 σ = A + B * ((T - T*) * (log10 tr + C*))
      → log10 tr = ((log10 σ - A)/B)/(T - T*) - C*
    """
    A = _try_keys(params, "A", "a")
    B = _try_keys(params, "B", "b")
    Tstar = _try_keys(params, "T_star", "T*", "Tstar")
    Cstar = _try_keys(params, "C_star", "C*", "Cstar")
    if None in (A, B, Tstar, Cstar):
        raise ValueError("Manson–Haferd parameters incomplete.")
    numerator = (math.log10(sigma_MPa) - A) / B
    denom = (T_K - Tstar)
    if abs(denom) < 1e-9:
        raise ValueError("Manson–Haferd: T very close to T*; numerical issue.")
    log10tr = (numerator / denom) - Cstar
    return 10.0 ** log10tr

def _rupture_time_from_yaml(y_path: Path, sigma_MPa: float, T_K: float) -> float:
    y = _load_yaml(y_path)
    model = str(y.get("model", "")).lower()
    params = y.get("params", {})
    if "manson" in model or "haferd" in model:
        return _rupture_time_MH(params, sigma_MPa, T_K)
    if "lmp" in model:
        return _rupture_time_LMP(params, sigma_MPa, T_K)
    # fallback: try both
    try:
        return _rupture_time_LMP(params, sigma_MPa, T_K)
    except Exception:
        return _rupture_time_MH(params, sigma_MPa, T_K)

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

# -------------------- main --------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Predict life via time-fraction (fatigue + creep).")
    # Inputs
    ap.add_argument("--T_K", type=float, required=True, help="Temperature [K].")
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
    args = ap.parse_args()

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
