#!/usr/bin/env python3
"""
Calibrate Basquin HCF law for Inconel 718 from CSV.

Model
-----
    log10(Nf) = a - k * log10(Sa_MPa)

Where Sa_MPa is stress amplitude in MPa.

This script is robust to varied HCF CSVs:
- Finds stress amplitude via many aliases or computes from:
  * Smax & Smin:      Sa = (Smax - Smin)/2
  * Smax & R-ratio:   Sa = Smax * (1 - R)/2
  * range Sr:         Sa = Sr/2
  * plain 'stress_MPa' treated as amplitude if nothing else found
- Accepts MPa or ksi (auto-converted)
- Accepts life as cycles or log10Nf, or reversals_to_failure/2
- Optional per-temperature fits if T is present

Outputs
-------
- YAML   : models/calibrations/fatigue/in718_hcf_basquin.yaml
- Figures: reports/figures/fatigue/basquin_pred_vs_meas.png
           reports/figures/fatigue/basquin_SN_byT.png
- Summary: reports/calibration/fatigue/basquin_fit.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math
import re
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


# ---------- utils ----------
def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _find_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    # loose contains-all-tokens fallback
    token_sets = [re.findall(r"[a-z0-9]+", a.lower()) for a in aliases]
    for c in df.columns:
        cl = c.lower()
        for toks in token_sets:
            if all(t in cl for t in toks):
                return c
    return None

_num_unit_pat = re.compile(r"""
    (?P<num>[-+]?(\d+(\.\d*)?|\.\d+))
    (\s*[×x*]\s*10\^?\s*(?P<exp>[-+]?\d+))?
""", re.VERBOSE)

def _coerce_series_numeric(s: pd.Series, debug: bool=False, name: str="") -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        out = s.astype(float)
        if debug:
            print(f"[debug] '{name}' numeric: N={len(out)}, NaN={int(out.isna().sum())}")
        return out
    def parse_one(v):
        if pd.isna(v): return np.nan
        st = str(v).strip().replace(",", "").replace("−", "-")
        m = _num_unit_pat.search(st)
        if not m:
            try: return float(st)
            except Exception: return np.nan
        base = float(m.group("num"))
        exp = m.group("exp")
        return base * (10.0 ** int(exp)) if exp is not None else base
    out = s.apply(parse_one).astype(float)
    if debug:
        print(f"[debug] '{name}' parsed: N={len(out)}, NaN={int(out.isna().sum())}")
    return out


# ---------- load ----------
def _load_csv_flex(csv_path: Path, debug: bool=False) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment="#", engine="python", skipinitialspace=True)
    if debug:
        print(f"[debug] loaded {csv_path} with columns: {list(df.columns)} (rows={len(df)})")

    # Temperature (optional)
    col_TK = _find_col(df, ["temperature_K","T_K","temp_K","temperature","T","temp","Temperature (K)","T (K)"])
    col_TC = _find_col(df, ["temperature_C","T_C","temp_C","Temperature (C)","T (C)"])
    if col_TK:
        T_K = _coerce_series_numeric(df[col_TK], debug, "T_K")
    elif col_TC:
        T_C = _coerce_series_numeric(df[col_TC], debug, "T_C")
        T_K = T_C + 273.15
    else:
        T_K = pd.Series([np.nan]*len(df))

    # Stress amplitude (many routes)
    col_Sa_MPa = _find_col(df, ["stress_amplitude_MPa","sigma_a_MPa","stress_a_MPa","Sa_MPa","sigma_a","Sa"])
    col_Sa_ksi = _find_col(df, ["stress_amplitude_ksi","sigma_a_ksi","Sa_ksi"])
    col_Splain_MPa = _find_col(df, ["stress_MPa","sigma_MPa","stress","sigma","stress_mpa","Stress (MPa)"])
    col_Splain_ksi = _find_col(df, ["stress_ksi","sigma_ksi","Stress (ksi)"])
    col_Smax = _find_col(df, ["stress_max_MPa","sigma_max_MPa","Smax_MPa","sigma_max"])
    col_Smin = _find_col(df, ["stress_min_MPa","sigma_min_MPa","Smin_MPa","sigma_min"])
    col_R    = _find_col(df, ["R_ratio","stress_ratio_R","R","stress_ratio"])
    col_Srange = _find_col(df, ["stress_range_MPa","sigma_range_MPa","range_MPa","Sr_MPa","stress_range"])

    Sa = None
    unit = "MPa"

    if col_Sa_MPa:
        Sa = _coerce_series_numeric(df[col_Sa_MPa], debug, "sigma_a_MPa"); unit = "MPa"
    elif col_Sa_ksi:
        Sa = _coerce_series_numeric(df[col_Sa_ksi], debug, "sigma_a_ksi") * 6.894759086; unit = "MPa"
    elif col_Smax and col_Smin:
        Smax = _coerce_series_numeric(df[col_Smax], debug, "sigma_max_MPa")
        Smin = _coerce_series_numeric(df[col_Smin], debug, "sigma_min_MPa")
        Sa = (Smax - Smin) / 2.0; unit = "MPa"
        if debug: print("[debug] computed Sa from Smax/Smin")
    elif col_Smax and col_R:
        Smax = _coerce_series_numeric(df[col_Smax], debug, "sigma_max_MPa")
        Rv   = _coerce_series_numeric(df[col_R], debug, "R_ratio")
        Sa = Smax * (1.0 - Rv) / 2.0; unit = "MPa"
        if debug: print("[debug] computed Sa from Smax and R")
    elif col_Srange:
        Sr = _coerce_series_numeric(df[col_Srange], debug, "stress_range_MPa")
        Sa = Sr / 2.0; unit = "MPa"
        if debug: print("[debug] computed Sa from stress range")
    elif col_Splain_MPa:
        Sa = _coerce_series_numeric(df[col_Splain_MPa], debug, "stress_MPa (assumed amplitude)"); unit = "MPa"
        if debug: print("[debug] using plain stress as amplitude (assumed)")
    elif col_Splain_ksi:
        Sa = _coerce_series_numeric(df[col_Splain_ksi], debug, "stress_ksi (assumed amplitude)") * 6.894759086; unit = "MPa"
        if debug: print("[debug] using plain stress (ksi) as amplitude (assumed)")
    else:
        raise ValueError("Could not infer stress amplitude. Provide Sa_MPa (preferred) or Smax/Smin or Smax+R, etc.")

    # Life
    col_log10N = _find_col(df, ["log10Nf","log10_Nf","logNf"])
    col_rev    = _find_col(df, ["reversals_to_failure","reversals","2Nf"])
    col_cycles = _find_col(df, ["cycles_to_failure","Nf","cycles","life_cycles","N","cycles to failure"])

    if col_log10N:
        log10N = _coerce_series_numeric(df[col_log10N], debug, "log10Nf")
        Nf = 10 ** log10N
    elif col_rev:
        rev = _coerce_series_numeric(df[col_rev], debug, "reversals_to_failure")
        Nf = rev / 2.0
    elif col_cycles:
        Nf = _coerce_series_numeric(df[col_cycles], debug, "Nf_cycles")
    else:
        raise ValueError("Life column not found. Expected log10Nf or cycles_to_failure / Nf or reversals_to_failure.")

    out = pd.DataFrame({"T_K": T_K, "sigma_a_MPa": Sa, "Nf_cycles": Nf})
    out = out.dropna(subset=["sigma_a_MPa","Nf_cycles"])
    out = out[(out["sigma_a_MPa"] > 0) & (out["Nf_cycles"] > 1)]
    if T_K.notna().any():
        out = out[(~out["T_K"].isna()) & (out["T_K"] > 0)]

    # sanity conversions if units look off
    Sa_med = float(out["sigma_a_MPa"].median()) if len(out) else np.nan
    if Sa_med < 0.5:  # likely input was in GPa
        out["sigma_a_MPa"] *= 1000.0
    if debug:
        print(f"[debug] cleaned rows: {len(out)}  (Sa unit assumed {unit}->MPa)")
        if len(out):
            print(f"[debug] Sa MPa range: {out['sigma_a_MPa'].min():.3g} .. {out['sigma_a_MPa'].max():.3g}")
            print(f"[debug] Nf range    : {out['Nf_cycles'].min():.3g} .. {out['Nf_cycles'].max():.3g}")
            if out['T_K'].notna().any():
                print(f"[debug] T_K range  : {out['T_K'].min():.3g} .. {out['T_K'].max():.3g}")

    if len(out) == 0:
        raise ValueError("After cleaning, no valid rows remain. Check CSV names/units.")
    return out.reset_index(drop=True)


# ---------- model ----------
@dataclass
class BasquinParams:
    a: float  # intercept in log10 space
    k: float  # positive slope magnitude (since y = a - k*x)

def _fit_line_log10(S_MPa: np.ndarray, Nf: np.ndarray) -> BasquinParams:
    x = np.log10(S_MPa.astype(float))
    y = np.log10(Nf.astype(float))
    m, b = np.polyfit(x, y, 1)    # y = m*x + b
    return BasquinParams(a=float(b), k=float(-m))

def _predict_log10(p: BasquinParams, S_MPa: np.ndarray) -> np.ndarray:
    return p.a - p.k * np.log10(S_MPa.astype(float))

def _metrics_log10(y_true_log10: np.ndarray, y_pred_log10: np.ndarray) -> Dict[str, float]:
    resid = y_true_log10 - y_pred_log10
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_tot = float(np.sum((y_true_log10 - np.mean(y_true_log10))**2))
    r2 = float(1.0 - np.sum(resid**2)/ss_tot) if ss_tot > 0 else float("nan")
    return {"R2_log10N": r2, "RMSE_log10N": rmse}


# ---------- outputs ----------
def _save_yaml_global(p: BasquinParams, out_path: Path, csv_path: Path, metrics: Dict[str, float], N: int) -> None:
    _ensure_dir(out_path)
    payload = {
        "model": "basquin",
        "form": "log10(Nf) = a - k * log10(sigma_a_MPa)",
        "units": {"stress": "MPa", "life": "cycles"},
        "params": {"a": p.a, "k": p.k},
        "fit": {
            "method": "OLS in log10 space",
            "date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset": str(csv_path),
            "metrics": metrics,
            "N": int(N),
        },
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, width=88)

def _save_yaml_byT(records: List[dict], out_path: Path, csv_path: Path,
                   global_metrics: Dict[str, float], N: int) -> None:
    _ensure_dir(out_path)
    payload = {
        "model": "basquin",
        "form": "log10(Nf) = a - k * log10(sigma_a_MPa) (per T)",
        "units": {"stress": "MPa", "life": "cycles", "temperature": "K"},
        "params_per_temperature": records,
        "fit": {
            "method": "OLS per unique T_K",
            "date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset": str(csv_path),
            "global_metrics": global_metrics,
            "N": int(N),
        },
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, width=88)

def _save_summary_txt_global(p: BasquinParams, out_txt: Path, metrics: Dict[str, float], csv_path: Path, N: int) -> None:
    _ensure_dir(out_txt)
    lines = [
        "Basquin HCF fit summary (global)",
        "================================",
        f"Dataset: {csv_path}",
        f"Date   : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')} (UTC)",
        "",
        "Model: log10(Nf) = a - k * log10(Sa)",
        f"a = {p.a:.6f}",
        f"k = {p.k:.6f}",
        "",
        "Metrics (log10 space):",
        f"R^2           : {metrics.get('R2_log10N', float('nan')):.6f}",
        f"RMSE log10(N) : {metrics.get('RMSE_log10N', float('nan')):.6f}",
        f"N             : {N}",
        "",
    ]
    out_txt.write_text("\n".join(lines))

def _save_summary_txt_byT(records: List[dict], out_txt: Path, global_metrics: Dict[str, float],
                          csv_path: Path, N: int) -> None:
    _ensure_dir(out_txt)
    lines = [
        "Basquin HCF fit summary (per temperature)",
        "=========================================",
        f"Dataset: {csv_path}",
        f"Date   : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')} (UTC)",
        "",
        "Model: log10(Nf) = a - k * log10(Sa) (separate fits per T)",
        "",
        "Per-temperature parameters:",
    ]
    for r in records:
        lines.append(f"  T={r['T_K']:.2f} K : a={r['a']:.6f}, k={r['k']:.6f} (N={r['N']})")
    lines += [
        "",
        "Global metrics (parity built from per-T fits):",
        f"R^2           : {global_metrics.get('R2_log10N', float('nan')):.6f}",
        f"RMSE log10(N) : {global_metrics.get('RMSE_log10N', float('nan')):.6f}",
        f"N             : {N}",
        "",
    ]
    out_txt.write_text("\n".join(lines))

def _plot_pred_vs_meas_loglog(N_meas: np.ndarray, N_pred: np.ndarray, out_png: Path) -> None:
    _ensure_dir(out_png)
    plt.figure(figsize=(5, 5), dpi=140)
    plt.loglog(N_meas, N_pred, "o", ms=4, alpha=0.8)
    lo = min(float(N_meas.min()), float(N_pred.min()))
    hi = max(float(N_meas.max()), float(N_pred.max()))
    plt.plot([lo, hi], [lo, hi], "--", lw=1)
    plt.xlabel("Measured life Nf [cycles]")
    plt.ylabel("Predicted life Nf [cycles]")
    plt.title("Basquin: Predicted vs Measured (log–log)")
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def _plot_SN_byT(df: pd.DataFrame, params_byT: List[dict], out_png: Path) -> None:
    _ensure_dir(out_png)
    plt.figure(figsize=(6.5, 4.5), dpi=140)
    for r in params_byT:
        T = r["T_K"]
        sel = np.isclose(df["T_K"].values, T)
        S = df.loc[sel, "sigma_a_MPa"].values
        N = df.loc[sel, "Nf_cycles"].values
        plt.loglog(S, N, "o", alpha=0.7, label=f"T≈{T:.0f} K (data)")
        Sspan = np.logspace(math.log10(S.min()*0.9), math.log10(S.max()*1.1), 120)
        Npred = 10 ** (r["a"] - r["k"] * np.log10(Sspan))
        plt.loglog(Sspan, Npred, "-", alpha=0.9, label=f"T≈{T:.0f} K fit")
    plt.xlabel("Stress amplitude σ_a [MPa]")
    plt.ylabel("Life Nf [cycles]")
    plt.title("Basquin S–N (by temperature)")
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# ---------- main ----------
def main() -> None:
    ap = argparse.ArgumentParser(description="Fit Basquin HCF law from CSV (robust loader; matches other fitters).")
    ap.add_argument("--csv", type=Path, default=Path("data/processed/in718_fatigue_HCF_isothermal_SI.csv"))
    ap.add_argument("--calib", type=Path, default=Path("models/calibrations/fatigue/in718_hcf_basquin.yaml"))
    ap.add_argument("--figdir", type=Path, default=Path("reports/figures/fatigue"))
    ap.add_argument("--summary", type=Path, default=Path("reports/calibration/fatigue/basquin_fit.txt"))
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-groupbyT", action="store_true", help="Force single global fit (ignore T groups).")
    args = ap.parse_args()

    df = _load_csv_flex(args.csv, debug=args.debug)

    has_T = df["T_K"].notna().any() and np.isfinite(df["T_K"]).any()
    group_by_T = has_T and (not args.no_groupbyT)

    if not group_by_T:
        p = _fit_line_log10(df["sigma_a_MPa"].to_numpy(), df["Nf_cycles"].to_numpy())
        ytrue = np.log10(df["Nf_cycles"].to_numpy())
        ypred = _predict_log10(p, df["sigma_a_MPa"].to_numpy())
        metrics = _metrics_log10(ytrue, ypred)

        _save_yaml_global(p, args.calib, args.csv, metrics, N=len(df))
        _save_summary_txt_global(p, args.summary, metrics, args.csv, N=len(df))
        _plot_pred_vs_meas_loglog(df["Nf_cycles"].to_numpy(), 10 ** ypred, args.figdir / "basquin_pred_vs_meas.png")

        # produce an S–N plot even without T separation (use median T tag if present)
        Tmed = float(np.nanmedian(df["T_K"].values)) if has_T else 0.0
        rec = [{"T_K": Tmed, "a": p.a, "k": p.k, "N": len(df)}]
        _plot_SN_byT(df.assign(T_K=Tmed), rec, args.figdir / "basquin_SN_byT.png")

    else:
        params_byT: List[dict] = []
        ytrue_all, ypred_all = [], []
        for T in sorted(np.unique(df["T_K"].round(6))):
            sel = np.isclose(df["T_K"].values, T)
            S = df.loc[sel, "sigma_a_MPa"].to_numpy()
            N = df.loc[sel, "Nf_cycles"].to_numpy()
            if len(S) < 2:
                continue
            p = _fit_line_log10(S, N)
            params_byT.append({"T_K": float(T), "a": p.a, "k": p.k, "N": int(len(S))})
            ytrue_all.append(np.log10(N))
            ypred_all.append(_predict_log10(p, S))

        if not params_byT:
            raise RuntimeError("No temperature group had enough points. Try --no-groupbyT.")

        ytrue_all = np.concatenate(ytrue_all)
        ypred_all = np.concatenate(ypred_all)
        metrics = _metrics_log10(ytrue_all, ypred_all)

        _save_yaml_byT(params_byT, args.calib, args.csv, metrics, N=len(df))
        _save_summary_txt_byT(params_byT, args.summary, metrics, args.csv, N=len(df))

        _plot_pred_vs_meas_loglog(10 ** ytrue_all, 10 ** ypred_all, args.figdir / "basquin_pred_vs_meas.png")
        _plot_SN_byT(df, params_byT, args.figdir / "basquin_SN_byT.png")

    print("✅ Basquin fit complete.")
    print(f"   YAML   : {args.calib}")
    print(f"   Summary: {args.summary}")
    print(f"   Figures: {args.figdir / 'basquin_pred_vs_meas.png'}")
    print(f"            {args.figdir / 'basquin_SN_byT.png'}")


if __name__ == "__main__":
    main()
