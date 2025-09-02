#!/usr/bin/env python3
"""
Calibrate steady-state creep (Norton–Arrhenius) law for Inconel 718 from CSV.

Model
-----
    ε̇ = A * σ^n * exp(-Q / (R T))

Units:
    σ in MPa, T in K, ε̇ in 1/s → A in [1/s / MPa^n], n [-], Q [J/mol], R [J/(mol·K)]

CSV
---
Flexible column names, comment lines starting with '#'. Accepts:
  Temperature:  temperature_K, T_K, temp_K, temperature, T, temp,
                temperature_C, T_C, temp_C (converted to K)
  Stress (MPa): stress_MPa, sigma_MPa, stress, sigma, stress_mpa, "Stress (MPa)"
  Rate (1/s):   strain_rate_per_s, strain_rate_1_per_s, epsdot_1_per_s,
                epsdot, edot, rate, "strain rate (1/s)"

Usage
-----
python scripts/fit_norton_from_csv.py \
  --csv data/processed/in718_creep_steady_state_SI.csv \
  --calib models/calibrations/creep/in718_norton.yaml \
  --debug
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math
import re
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

R_GAS = 8.314462618  # J/(mol·K)

# ---------- Utilities ----------
def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    # exact (case-insensitive) match first
    lower_map = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    # then try fuzzy: contains all alias tokens (space/paren tolerant)
    tokensets = [re.findall(r"[a-z0-9]+", a.lower()) for a in aliases]
    for col in df.columns:
        cl = col.lower()
        for toks in tokensets:
            if all(t in cl for t in toks):
                return col
    return None

_num_unit_pat = re.compile(r"""
    (?P<num>[-+]?(\d+(\.\d*)?|\.\d+))      # base number
    (\s*[×x*]\s*10\^?\s*(?P<exp>[-+]?\d+))? # optional ×10^exp
""", re.VERBOSE)

def _coerce_series_numeric(s: pd.Series, debug: bool=False, name: str="") -> pd.Series:
    """
    Convert possibly unit-labeled strings to floats.
    Handles: '551 MPa', '1.3×10^-6 1/s', '1000 K', commas, spaces.
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    def parse_one(v):
        if pd.isna(v):
            return np.nan
        st = str(v).strip()
        # Replace commas as thousand separators
        st = st.replace(",", "")
        # Normalize unicode minus
        st = st.replace("−", "-")
        # Try to extract number and optional ×10^exp
        m = _num_unit_pat.search(st)
        if not m:
            # try simple to_numeric
            try:
                return float(st)
            except Exception:
                return np.nan
        base = float(m.group("num"))
        exp = m.group("exp")
        if exp is not None:
            return base * (10.0 ** int(exp))
        return base

    out = s.apply(parse_one).astype(float)
    if debug:
        n_nan = int(out.isna().sum())
        print(f"[debug] parsed '{name}': {len(out)-n_nan} numeric, {n_nan} NaN")
    return out

def _load_csv_flex(csv_path: Path, debug: bool=False) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment="#", engine="python", skipinitialspace=True)
    if debug:
        print(f"[debug] loaded CSV with columns: {list(df.columns)}")
        print(f"[debug] raw rows: {len(df)}")

    # Temperature
    col_TK = _find_col(df, ["temperature_K","T_K","temp_K","temperature","T","temp","Temperature (K)","T (K)"])
    col_TC = _find_col(df, ["temperature_C","T_C","temp_C","Temperature (C)","T (C)"])
    if col_TK:
        T_K = _coerce_series_numeric(df[col_TK], debug, "T_K")
    elif col_TC:
        T_C = _coerce_series_numeric(df[col_TC], debug, "T_C")
        T_K = T_C + 273.15
    else:
        raise ValueError("Temperature column not found.")

    # Stress (MPa)
    col_S = _find_col(df, ["stress_MPa","sigma_MPa","stress","sigma","stress_mpa","Stress (MPa)","Sigma (MPa)"])
    if not col_S:
        raise ValueError("Stress column not found (expect MPa).")
    S_MPa = _coerce_series_numeric(df[col_S], debug, "stress_MPa")

    # Strain rate (1/s)
    col_E = _find_col(df, ["strain_rate_per_s","strain_rate_1_per_s","epsdot_1_per_s","epsdot","edot","rate","strain rate (1/s)","StrainRate (1/s)"])
    if not col_E:
        raise ValueError("Strain rate column not found (1/s).")
    E_s = _coerce_series_numeric(df[col_E], debug, "epsdot_1_per_s")

    out = pd.DataFrame({"T_K": T_K, "stress_MPa": S_MPa, "epsdot_1_per_s": E_s})

    # Drop NaNs
    before = len(out)
    out = out.dropna(subset=["T_K","stress_MPa","epsdot_1_per_s"])
    after_drop = len(out)

    # Physical positivity
    out = out[(out["T_K"] > 0) & (out["stress_MPa"] > 0) & (out["epsdot_1_per_s"] > 0)]
    after_phys = len(out)

    if debug:
        print(f"[debug] after dropna: {after_drop} rows (from {before})")
        print(f"[debug] after positivity filter: {after_phys} rows")
        print(f"[debug] sample cleaned rows:\n{out.head(5)}")

    if len(out) == 0:
        raise ValueError("After cleaning, no valid rows remain. Check units/formatting (e.g., 'MPa', '1/s', '×10^-6').")

    return out.reset_index(drop=True)

# ---------- Model ----------
@dataclass
class NortonParams:
    A: float
    n: float
    Q_J_per_mol: float

def fit_norton(T_K: np.ndarray, S_MPa: np.ndarray, Edot: np.ndarray) -> NortonParams:
    """
    ln(ε̇) = ln(A) + n*ln(σ) - Q/(R*T)
           = [1, lnσ, -1/(R*T)] · [lnA, n, Q]
    """
    x1 = np.log(S_MPa.astype(float))
    x2 = -1.0 / (R_GAS * T_K.astype(float))
    X = np.column_stack([np.ones_like(x1), x1, x2])
    y = np.log(Edot.astype(float))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    lnA, n, Q = beta
    return NortonParams(A=float(np.exp(lnA)), n=float(n), Q_J_per_mol=float(Q))

def predict_rate(p: NortonParams, stress_MPa: np.ndarray, T_K: np.ndarray) -> np.ndarray:
    return p.A * np.power(stress_MPa, p.n) * np.exp(-p.Q_J_per_mol / (R_GAS * T_K))

def _metrics_log10(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    lt, lp = np.log10(y_true), np.log10(y_pred)
    resid = lt - lp
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_tot = float(np.sum((lt - np.mean(lt))**2))
    r2 = float(1.0 - np.sum(resid**2)/ss_tot) if ss_tot > 0 else float("nan")
    return {"R2_log10": r2, "RMSE_log10_rate": rmse}

def _save_yaml(p: NortonParams, out_path: Path, csv_path: Path, metrics: dict) -> None:
    _ensure_dir(out_path)
    payload = {
        "model": "norton",
        "units": {"stress": "MPa", "temperature": "K", "rate": "1/s"},
        "params": {
            "A": p.A,
            "n": p.n,
            "Q_J_per_mol": p.Q_J_per_mol,
            "R_J_per_molK": R_GAS,
        },
        "fit": {
            "method": "OLS on ln(rate) with [1, ln(sigma), -1/(R*T)]",
            "date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset": str(csv_path),
            "metrics": metrics,
            "N": int(metrics.get("N", 0)),
        },
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, width=88)

def _save_summary_txt(p: NortonParams, out_txt: Path, metrics: dict, csv_path: Path) -> None:
    _ensure_dir(out_txt)
    lines = [
        "Norton–Arrhenius fit summary",
        "============================",
        f"Dataset: {csv_path}",
        f"Date   : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')} (UTC)",
        "",
        "Model: epsdot = A * sigma^n * exp(-Q/(R T))",
        f"A   = {p.A:.6e} [1/s / MPa^n]",
        f"n   = {p.n:.6f} [-]",
        f"Q   = {p.Q_J_per_mol:.6e} [J/mol]",
        f"R   = {R_GAS:.9f} [J/(mol·K)]",
        "",
        "Metrics (on log10(rate))",
        f"R^2             : {metrics.get('R2_log10', float('nan')):.6f}",
        f"RMSE log10(rate): {metrics.get('RMSE_log10_rate', float('nan')):.6f}",
        f"N               : {metrics.get('N', 0)}",
        "",
    ]
    out_txt.write_text("\n".join(lines))

def _plot_pred_vs_meas(df: pd.DataFrame, y_pred: np.ndarray, out_png: Path) -> None:
    _ensure_dir(out_png)
    plt.figure(figsize=(5, 5), dpi=140)
    plt.loglog(df["epsdot_1_per_s"].values, y_pred, "o", ms=4, alpha=0.75)
    lo = min(df["epsdot_1_per_s"].min(), y_pred.min())
    hi = max(df["epsdot_1_per_s"].max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "--", lw=1)
    plt.xlabel("Measured ε̇ [1/s]")
    plt.ylabel("Predicted ε̇ [1/s]")
    plt.title("Norton: Predicted vs Measured (log–log)")
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def _plot_loglog_by_temperature(df: pd.DataFrame, p: NortonParams, out_png: Path) -> None:
    _ensure_dir(out_png)
    Tq = np.quantile(df["T_K"].values, [0.1, 0.3, 0.5, 0.7, 0.9])
    sig = np.logspace(
        math.log10(max(1e-6, df["stress_MPa"].min() * 0.8)),
        math.log10(df["stress_MPa"].max() * 1.25),
        120,
    )
    plt.figure(figsize=(6, 4), dpi=140)
    for T in Tq:
        epsdot = predict_rate(p, sig, np.full_like(sig, T))
        plt.loglog(sig, epsdot, label=f"T≈{T:.0f} K")
    plt.xlabel("Stress σ [MPa]")
    plt.ylabel("Steady-state ε̇ [1/s]")
    plt.title("Norton law: ε̇–σ at representative T")
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main() -> None:
    ap = argparse.ArgumentParser(description="Fit Norton–Arrhenius law from CSV.")
    ap.add_argument("--csv", type=Path, default=Path("data/processed/in718_creep_steady_state_SI.csv"))
    ap.add_argument("--calib", type=Path, default=Path("models/calibrations/creep/in718_norton.yaml"))
    ap.add_argument("--figdir", type=Path, default=Path("reports/figures/creep"))
    ap.add_argument("--summary", type=Path, default=Path("reports/calibration/creep/norton_fit.txt"))
    ap.add_argument("--debug", action="store_true", help="Print column detection and cleaning stats.")
    args = ap.parse_args()

    df = _load_csv_flex(args.csv, debug=args.debug)

    params = fit_norton(df["T_K"].to_numpy(), df["stress_MPa"].to_numpy(), df["epsdot_1_per_s"].to_numpy())

    y_pred = predict_rate(params, df["stress_MPa"].to_numpy(), df["T_K"].to_numpy())
    metrics = _metrics_log10(df["epsdot_1_per_s"].to_numpy(), y_pred)
    metrics["N"] = int(len(df))

    _save_yaml(params, args.calib, args.csv, metrics)
    _save_summary_txt(params, args.summary, metrics, args.csv)

    _plot_pred_vs_meas(df, y_pred, args.figdir / "norton_pred_vs_meas.png")
    _plot_loglog_by_temperature(df, params, args.figdir / "norton_loglog_byT.png")

    print("✅ Norton fit complete.")
    print(f"   YAML   : {args.calib}")
    print(f"   Summary: {args.summary}")
    print(f"   Figures: {args.figdir / 'norton_pred_vs_meas.png'}")
    print(f"            {args.figdir / 'norton_loglog_byT.png'}")

if __name__ == "__main__":
    main()
