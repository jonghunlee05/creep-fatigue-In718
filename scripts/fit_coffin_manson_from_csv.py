#!/usr/bin/env python3
"""
Calibrate Coffin–Manson LCF / Total Strain–Life for Inconel 718 from CSV.

Forms (log10 space)
-------------------
Elastic (Basquin part):
    Δε_e/2 = (σ'_f / E) * (2N)^b     →  log10(Δε_e/2) = log10(σ'_f/E) + b * log10(2N)

Plastic (Coffin–Manson part):
    Δε_p/2 =  ε'_f       * (2N)^c     →  log10(Δε_p/2) = log10(ε'_f)    + c * log10(2N)

Total strain–life (Manson–Coffin–Basquin):
    Δε/2   = (σ'_f / E) * (2N)^b  +  ε'_f * (2N)^c

This script:
- Loads common LCF columns with comments tolerated.
- If both *total* and *plastic* amplitudes are present, it computes Δε_e/2 = Δε/2 − Δε_p/2
  and performs two separate linear fits in log space (no nonlinear solver needed).
- If only *plastic* is present → fits plastic-only.
- If only *total* is present → fits a single power law to total (fallback; warns).

Temperature grouping:
- If a temperature column exists, fits parameters per unique T (rounded) unless --no-groupbyT.

Inputs (column aliases accepted)
--------------------------------
Life:
  cycles_to_failure, Nf, cycles, reversals_to_failure (uses 2N = reversals), 2Nf
Strains (unitless; e.g., 0.005 for 0.5%):
  total_strain_amplitude, total_strain_amp, strain_total, de_over_2_total
  plastic_strain_amplitude, plastic_strain_amp, strain_plastic, de_over_2_plastic
  elastic_strain_amplitude (optional; if absent but total & plastic exist, computed as total-plastic)
Temperature:
  temperature_K, T_K, temp_K  OR  temperature_C, T_C  OR  mean_temp_F (Fahrenheit; converted)

Optional modulus (if you want σ'_f reported):
  E_GPa, youngs_modulus_GPa   (if provided, script also computes σ'_f = (σ'_f/E)*E )

Usage
-----
  python scripts/fit_coffin_manson_from_csv.py
  python scripts/fit_coffin_manson_from_csv.py --csv data/processed/in718_fatigue_LCF_isothermal_SI.csv --debug
  python scripts/fit_coffin_manson_from_csv.py --no-groupbyT

Outputs
-------
- YAML   : models/calibrations/fatigue/in718_lcf_coffin_manson.yaml
- Figures: reports/figures/fatigue/coffin_parity.png
           reports/figures/fatigue/coffin_strainlife_byT.png
- Summary: reports/calibration/fatigue/coffin_fit.txt
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


# -------------------- utils --------------------
def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _find_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    # soft fallback: all tokens contained (order/spacing tolerant)
    token_sets = [re.findall(r"[a-z0-9]+", a.lower()) for a in aliases]
    for c in df.columns:
        cl = c.lower()
        for toks in token_sets:
            if all(t in cl for t in toks):
                return c
    return None

_num_pat = re.compile(r"""
    (?P<num>[-+]?(\d+(\.\d*)?|\.\d+))
    (\s*[×x*]\s*10\^?\s*(?P<exp>[-+]?\d+))?
""", re.VERBOSE)

def _coerce_series_numeric(s: pd.Series, debug: bool=False, name: str="") -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        out = s.astype(float)
        if debug: print(f"[debug] '{name}': numeric dtype; NaN={int(out.isna().sum())}/{len(out)}")
        return out

    def parse_one(v):
        if pd.isna(v): return np.nan
        st = str(v).strip().replace(",", "").replace("−", "-")
        if st.endswith("%"):
            try: return float(st[:-1]) / 100.0
            except Exception: return np.nan
        m = _num_pat.search(st)
        if not m:
            try: return float(st)
            except Exception: return np.nan
        base = float(m.group("num"))
        exp = m.group("exp")
        return base * (10.0 ** int(exp)) if exp is not None else base

    out = s.apply(parse_one).astype(float)
    if debug: print(f"[debug] '{name}' parsed; NaN={int(out.isna().sum())}/{len(out)}")
    return out


# -------------------- loading --------------------
def _load_csv_flex(csv_path: Path, debug: bool=False) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment="#", engine="python", skipinitialspace=True)
    if debug:
        print(f"[debug] loaded columns: {list(df.columns)}  (rows={len(df)})")

    # Life
    col_rev = _find_col(df, ["reversals_to_failure","reversals","2Nf"])
    col_N   = _find_col(df, ["cycles_to_failure","Nf","cycles","life_cycles","N","cycles to failure"])

    if col_rev:
        twoN = _coerce_series_numeric(df[col_rev], debug, "reversals_to_failure")
        Nf   = twoN / 2.0
    elif col_N:
        Nf   = _coerce_series_numeric(df[col_N], debug, "Nf_cycles")
        twoN = 2.0 * Nf
    else:
        raise ValueError("Life column not found (need cycles_to_failure or reversals_to_failure).")

    # Temperature
    col_TK = _find_col(df, ["temperature_K","T_K","temp_K","temperature","T","temp","Temperature (K)","T (K)"])
    col_TC = _find_col(df, ["temperature_C","T_C","temp_C","Temperature (C)","T (C)"])
    col_TF = _find_col(df, ["mean_temp_F","temperature_F","T_F"])
    if col_TK:
        T_K = _coerce_series_numeric(df[col_TK], debug, "T_K")
    elif col_TC:
        T_C = _coerce_series_numeric(df[col_TC], debug, "T_C")
        T_K = T_C + 273.15
    elif col_TF:
        T_F = _coerce_series_numeric(df[col_TF], debug, "T_F")
        T_K = (T_F - 32.0) * (5.0/9.0) + 273.15
    else:
        T_K = pd.Series([np.nan]*len(df))  # allowed

    # Strain amplitudes (unitless)
    col_total   = _find_col(df, ["total_strain_amplitude","total_strain_amp","strain_total","de_over_2_total","total strain amplitude"])
    col_plastic = _find_col(df, ["plastic_strain_amplitude","plastic_strain_amp","strain_plastic","de_over_2_plastic","plastic strain amplitude"])
    col_elastic = _find_col(df, ["elastic_strain_amplitude","elastic_strain_amp","strain_elastic","de_over_2_elastic"])

    eps_total   = _coerce_series_numeric(df[col_total],   debug, "total_strain_amplitude")   if col_total   else None
    eps_plastic = _coerce_series_numeric(df[col_plastic], debug, "plastic_strain_amplitude") if col_plastic else None
    eps_elastic = _coerce_series_numeric(df[col_elastic], debug, "elastic_strain_amplitude") if col_elastic else None

    # Optional modulus
    col_E = _find_col(df, ["E_GPa","youngs_modulus_GPa","modulus_GPa"])
    E_GPa = _coerce_series_numeric(df[col_E], debug, "E_GPa") if col_E else None

    out = pd.DataFrame({"T_K": T_K, "Nf_cycles": Nf, "twoN": twoN})
    if eps_total is not None:   out["eps_total"]   = eps_total
    if eps_plastic is not None: out["eps_plastic"] = eps_plastic
    if eps_elastic is not None: out["eps_elastic"] = eps_elastic
    if E_GPa is not None:       out["E_GPa"]       = E_GPa

    # Clean basic
    out = out.dropna(subset=["Nf_cycles","twoN"])
    out = out[(out["Nf_cycles"] > 1)]
    if out["T_K"].notna().any():
        out = out[(~out["T_K"].isna()) & (out["T_K"] > 0)]

    # Derive elastic if we have total and plastic
    if "eps_total" in out.columns and "eps_plastic" in out.columns and "eps_elastic" not in out.columns:
        out["eps_elastic"] = out["eps_total"] - out["eps_plastic"]

    # Filter physical strains
    for col in ["eps_total","eps_plastic","eps_elastic"]:
        if col in out.columns:
            out = out[(~out[col].isna()) & (out[col] > 0)]

    if debug:
        print(f"[debug] cleaned rows: {len(out)}")
        for col in ["eps_total","eps_plastic","eps_elastic"]:
            if col in out.columns:
                print(f"[debug] {col} range: {out[col].min():.3g} .. {out[col].max():.3g}")
        if "E_GPa" in out.columns:
            print(f"[debug] E_GPa range : {out['E_GPa'].min():.3g} .. {out['E_GPa'].max():.3g}")

    if len(out) == 0:
        raise ValueError("After cleaning, no valid rows remain. Check CSV contents/units.")
    return out.reset_index(drop=True)


# -------------------- model / fitting --------------------
@dataclass
class CMElastic:
    log10_sigmaf_over_E: float   # intercept for elastic term
    b: float                     # usually negative
    # derived:
    @property
    def sigmaf_over_E(self) -> float:
        return 10.0 ** self.log10_sigmaf_over_E

@dataclass
class CMPlastic:
    log10_epsf: float            # intercept for plastic term
    c: float                     # usually negative
    @property
    def epsf(self) -> float:
        return 10.0 ** self.log10_epsf

@dataclass
class CMTotal:
    elastic: Optional[CMElastic]
    plastic: Optional[CMPlastic]

def _fit_powerlaw(x_log10: np.ndarray, y_log10: np.ndarray) -> tuple[float,float]:
    """
    Fit y = A + m x  (both y and x are already log10).
    Returns (A, m). Use for elastic or plastic sub-fit.
    """
    m, A = np.polyfit(x_log10, y_log10, 1)  # y = m*x + A
    return float(A), float(m)

def fit_coffin_manson(df: pd.DataFrame) -> CMTotal:
    """
    Decide fit mode based on available columns in df:
    - If eps_elastic and eps_plastic exist → fit both terms separately.
    - Else if eps_plastic only → plastic-only.
    - Else if eps_total only → single power-law fallback on total (warn).
    """
    x = np.log10(df["twoN"].astype(float))  # x = log10(2N)

    elastic = None
    plastic = None

    if "eps_elastic" in df.columns and "eps_plastic" in df.columns:
        # Elastic term
        ye = np.log10(df["eps_elastic"].astype(float))
        Ae, me = _fit_powerlaw(x, ye)           # ye = Ae + me * x
        elastic = CMElastic(log10_sigmaf_over_E=Ae, b=me)

        # Plastic term
        yp = np.log10(df["eps_plastic"].astype(float))
        Ap, mp = _fit_powerlaw(x, yp)           # yp = Ap + mp * x
        plastic = CMPlastic(log10_epsf=Ap, c=mp)

    elif "eps_plastic" in df.columns:
        yp = np.log10(df["eps_plastic"].astype(float))
        Ap, mp = _fit_powerlaw(x, yp)
        plastic = CMPlastic(log10_epsf=Ap, c=mp)

    elif "eps_total" in df.columns:
        # WARNING: single power law on total (no separation)
        yt = np.log10(df["eps_total"].astype(float))
        At, mt = _fit_powerlaw(x, yt)
        # Store as "plastic-only style" with a note; caller will mark in YAML.
        plastic = CMPlastic(log10_epsf=At, c=mt)

    else:
        raise RuntimeError("No strain amplitude columns found to fit.")

    return CMTotal(elastic=elastic, plastic=plastic)


# -------------------- predictions & metrics --------------------
def predict_total(p: CMTotal, twoN: np.ndarray) -> np.ndarray:
    x = np.log10(twoN.astype(float))
    eps = np.zeros_like(x, dtype=float)

    if p.elastic is not None:
        eps_e = (10.0 ** (p.elastic.log10_sigmaf_over_E)) * (10.0 ** (p.elastic.b * x))
        eps += eps_e
    if p.plastic is not None:
        eps_p = (10.0 ** (p.plastic.log10_epsf)) * (10.0 ** (p.plastic.c * x))
        eps += eps_p
    return eps

def _metrics_log10(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    lt, lp = np.log10(y_true), np.log10(y_pred)
    resid = lt - lp
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_tot = float(np.sum((lt - np.mean(lt))**2))
    r2 = float(1.0 - np.sum(resid**2)/ss_tot) if ss_tot > 0 else float("nan")
    return {"R2_log10": r2, "RMSE_log10_strain": rmse}


# -------------------- outputs --------------------
def _save_yaml(p: CMTotal, out_path: Path, csv_path: Path, metrics: Dict[str, float],
               N: int, E_GPa_global: Optional[float], mode: str) -> None:
    """
    mode: "both_terms", "plastic_only", "total_single_power"
    """
    _ensure_dir(out_path)

    payload: Dict[str, object] = {
        "model": "coffin_manson",
        "form": "Delta_eps/2 = (sigma_f'/E)*(2N)^b + eps_f'*(2N)^c",
        "units": {"strain": "unitless", "life": "cycles", "E": "GPa"},
        "fit": {
            "method": "OLS in log10 space (separate elastic/plastic when available)",
            "date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset": str(csv_path),
            "metrics": metrics,
            "N": int(N),
            "mode": mode,
        },
        "params": {},
    }

    if p.elastic is not None:
        payload["params"]["log10_sigmaf_over_E"] = p.elastic.log10_sigmaf_over_E
        payload["params"]["b"] = p.elastic.b
    if p.plastic is not None:
        payload["params"]["log10_epsf"] = p.plastic.log10_epsf
        payload["params"]["c"] = p.plastic.c

    if E_GPa_global is not None:
        payload["params"]["E_GPa"] = float(E_GPa_global)
        if p.elastic is not None:
            sigmaf_over_E = 10.0 ** p.elastic.log10_sigmaf_over_E
            payload["params"]["sigmaf_prime_MPa"] = float(sigmaf_over_E * E_GPa_global * 1000.0)  # GPa→MPa

    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, width=88)

def _save_summary_txt(p: CMTotal, out_txt: Path, metrics: Dict[str, float], csv_path: Path,
                      N: int, E_GPa_global: Optional[float], mode: str) -> None:
    _ensure_dir(out_txt)
    lines = [
        "Coffin–Manson LCF fit summary",
        "=============================",
        f"Dataset: {csv_path}",
        f"Date   : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')} (UTC)",
        "",
        "Model: Δε/2 = (σ'_f/E)*(2N)^b + ε'_f*(2N)^c",
        f"Mode : {mode}",
    ]
    if p.elastic is not None:
        lines += [
            "",
            "Elastic term:",
            f"  log10(σ'_f/E) = {p.elastic.log10_sigmaf_over_E:.6f}",
            f"  b             = {p.elastic.b:.6f}",
        ]
    if p.plastic is not None:
        lines += [
            "",
            "Plastic term:",
            f"  log10(ε'_f)   = {p.plastic.log10_epsf:.6f}",
            f"  c             = {p.plastic.c:.6f}",
        ]
    if E_GPa_global is not None and p.elastic is not None:
        sigmaf_over_E = 10.0 ** p.elastic.log10_sigmaf_over_E
        lines += [
            "",
            f"Assumed/mean E  = {E_GPa_global:.3f} GPa",
            f"σ'_f (derived)  = {sigmaf_over_E * E_GPa_global * 1000.0:.3f} MPa",
        ]
    lines += [
        "",
        "Metrics (on log10(strain amplitude))",
        f"R^2                 : {metrics.get('R2_log10', float('nan')):.6f}",
        f"RMSE log10(strain)  : {metrics.get('RMSE_log10_strain', float('nan')):.6f}",
        f"N                   : {N}",
        "",
    ]
    out_txt.write_text("\n".join(lines))

def _plot_parity(y_meas: np.ndarray, y_pred: np.ndarray, out_png: Path, title: str) -> None:
    _ensure_dir(out_png)
    plt.figure(figsize=(5,5), dpi=140)
    plt.loglog(y_meas, y_pred, "o", ms=4, alpha=0.8)
    lo = min(float(y_meas.min()), float(y_pred.min()))
    hi = max(float(y_meas.max()), float(y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "--", lw=1)
    plt.xlabel("Measured Δε/2 [–]")
    plt.ylabel("Predicted Δε/2 [–]")
    plt.title(title)
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def _plot_strainlife_byT(df: pd.DataFrame, params_byT: List[dict], out_png: Path) -> None:
    """
    Plot Δε/2 vs N (total) using fitted terms for each temperature group.
    If only plastic-only mode is available, the plot shows plastic prediction.
    """
    _ensure_dir(out_png)
    plt.figure(figsize=(6.5, 4.5), dpi=140)
    for r in params_byT:
        T = r["T_K"]
        sel = np.isclose(df["T_K"].values, T)
        N = df.loc[sel, "Nf_cycles"].values
        twoN = df.loc[sel, "twoN"].values

        # measured total if present, else plastic
        if "eps_total" in df.columns:
            y_meas = df.loc[sel, "eps_total"].values
        elif "eps_plastic" in df.columns:
            y_meas = df.loc[sel, "eps_plastic"].values
        else:
            y_meas = None

        if y_meas is not None:
            plt.loglog(N, y_meas, "o", alpha=0.7, label=f"T≈{T:.0f} K (data)")

        # predicted curve over span
        Nspan = np.logspace(math.log10(max(N.min()*0.9, 5.0)), math.log10(N.max()*1.1), 160)
        twoNspan = 2.0 * Nspan
        # reconstruct parameters
        eps = np.zeros_like(Nspan, dtype=float)
        if r.get("log10_sigmaf_over_E") is not None and r.get("b") is not None:
            eps += (10.0 ** r["log10_sigmaf_over_E"]) * (twoNspan ** r["b"])
        if r.get("log10_epsf") is not None and r.get("c") is not None:
            eps += (twoNspan ** r["c"]) * (10.0 ** r["log10_epsf"])
        plt.loglog(Nspan, eps, "-", alpha=0.9, label=f"T≈{T:.0f} K fit")

    plt.xlabel("Life Nf [cycles]")
    plt.ylabel("Strain amplitude Δε/2 [–]")
    plt.title("Coffin–Manson Strain–Life (by temperature)")
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# -------------------- main --------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Fit Coffin–Manson / Total Strain–Life from CSV (similar format to other fitters).")
    ap.add_argument("--csv", type=Path, default=Path("data/processed/in718_fatigue_LCF_isothermal_537C_SI.csv"),
                    help="Input CSV path.")
    ap.add_argument("--calib", type=Path, default=Path("models/calibrations/fatigue/in718_lcf_coffin_manson.yaml"),
                    help="Output YAML path.")
    ap.add_argument("--figdir", type=Path, default=Path("reports/figures/fatigue"),
                    help="Directory for figures.")
    ap.add_argument("--summary", type=Path, default=Path("reports/calibration/fatigue/coffin_fit.txt"),
                    help="Plain-text summary path.")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-groupbyT", action="store_true", help="Force single global fit (ignore T groups).")
    args = ap.parse_args()

    df = _load_csv_flex(args.csv, debug=args.debug)

    has_T = df["T_K"].notna().any() and np.isfinite(df["T_K"]).any()
    group_by_T = has_T and (not args.no_groupbyT)

    E_GPa_global = None
    if "E_GPa" in df.columns:
        # use median E if provided row-wise
        E_GPa_global = float(np.nanmedian(df["E_GPa"].values))

    params_records: List[dict] = []
    ytrue, ypred = [], []

    if not group_by_T:
        p = fit_coffin_manson(df)
        # choose target for parity: total if available else plastic
        if "eps_total" in df.columns:
            meas = df["eps_total"].to_numpy()
        elif "eps_plastic" in df.columns:
            meas = df["eps_plastic"].to_numpy()
        else:
            meas = predict_total(p, df["twoN"].to_numpy())  # degenerate case; will give perfect line

        pred = predict_total(p, df["twoN"].to_numpy())
        metrics = _metrics_log10(meas, pred)

        # records for plotting
        Tmed = float(np.nanmedian(df["T_K"].values)) if has_T else 0.0
        rec = {"T_K": Tmed}
        if p.elastic is not None:
            rec["log10_sigmaf_over_E"] = p.elastic.log10_sigmaf_over_E
            rec["b"] = p.elastic.b
        if p.plastic is not None:
            rec["log10_epsf"] = p.plastic.log10_epsf
            rec["c"] = p.plastic.c
        rec["N"] = int(len(df))
        params_records = [rec]

        mode = ("both_terms" if (p.elastic is not None and p.plastic is not None)
                else "plastic_only" if (p.elastic is None and p.plastic is not None and "eps_plastic" in df.columns)
                else "total_single_power")

        _save_yaml(p, args.calib, args.csv, metrics, N=len(df), E_GPa_global=E_GPa_global, mode=mode)
        _save_summary_txt(p, args.summary, metrics, args.csv, N=len(df), E_GPa_global=E_GPa_global, mode=mode)

        _plot_parity(meas, pred, args.figdir / "coffin_parity.png", "Coffin–Manson: Predicted vs Measured (log–log)")
        _plot_strainlife_byT(df.assign(T_K=Tmed), params_records, args.figdir / "coffin_strainlife_byT.png")

    else:
        # per-temperature fits
        for T in sorted(np.unique(df["T_K"].round(6))):
            sel = np.isclose(df["T_K"].values, T)
            dT = df.loc[sel].copy()
            if len(dT) < 2:
                continue
            p = fit_coffin_manson(dT)

            rec = {"T_K": float(T), "N": int(len(dT))}
            if p.elastic is not None:
                rec["log10_sigmaf_over_E"] = p.elastic.log10_sigmaf_over_E
                rec["b"] = p.elastic.b
            if p.plastic is not None:
                rec["log10_epsf"] = p.plastic.log10_epsf
                rec["c"] = p.plastic.c
            params_records.append(rec)

            # accumulate parity arrays
            if "eps_total" in dT.columns:
                ytrue.append(dT["eps_total"].to_numpy())
            elif "eps_plastic" in dT.columns:
                ytrue.append(dT["eps_plastic"].to_numpy())
            pred = predict_total(p, dT["twoN"].to_numpy())
            ypred.append(pred)

        if not params_records:
            raise RuntimeError("No temperature group had enough points to fit. Try --no-groupbyT.")

        ytrue_all = np.concatenate(ytrue) if ytrue else np.concatenate(ypred)  # degenerate fallback
        ypred_all = np.concatenate(ypred)
        metrics = _metrics_log10(ytrue_all, ypred_all)

        # save a compact YAML with params_per_temperature
        _ensure_dir(args.calib)
        payload = {
            "model": "coffin_manson",
            "form": "Delta_eps/2 = (sigma_f'/E)*(2N)^b + eps_f'*(2N)^c  (per temperature)",
            "units": {"strain": "unitless", "life": "cycles", "E": "GPa"},
            "params_per_temperature": params_records,
            "fit": {
                "method": "OLS in log10 space (separate elastic/plastic when available) per T",
                "date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "dataset": str(args.csv),
                "global_metrics": metrics,
                "N": int(len(df)),
            },
        }
        with open(args.calib, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False, width=88)

        # summary
        _ensure_dir(args.summary)
        lines = [
            "Coffin–Manson LCF fit summary (per temperature)",
            "===============================================",
            f"Dataset: {args.csv}",
            f"Date   : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')} (UTC)",
            "",
            "Per-temperature parameters:",
        ]
        for r in params_records:
            s_el = (f"log10(σ'_f/E)={r['log10_sigmaf_over_E']:.6f}, b={r['b']:.6f}"
                    if "log10_sigmaf_over_E" in r else "elastic: n/a")
            s_pl = (f"log10(ε'_f)={r['log10_epsf']:.6f}, c={r['c']:.6f}"
                    if "log10_epsf" in r else "plastic: n/a")
            lines.append(f"  T={r['T_K']:.2f} K (N={r['N']}): {s_el}; {s_pl}")
        lines += [
            "",
            "Global metrics (on log10(strain amplitude))",
            f"R^2                 : {metrics.get('R2_log10', float('nan')):.6f}",
            f"RMSE log10(strain)  : {metrics.get('RMSE_log10_strain', float('nan')):.6f}",
            f"N                   : {len(df)}",
            "",
        ]
        Path(args.summary).write_text("\n".join(lines))

        # figures
        _plot_parity(ytrue_all, ypred_all, args.figdir / "coffin_parity.png",
                     "Coffin–Manson: Predicted vs Measured (log–log)")
        _plot_strainlife_byT(df, params_records, args.figdir / "coffin_strainlife_byT.png")

    print("✅ Coffin–Manson fit complete.")
    print(f"   YAML   : {args.calib}")
    print(f"   Summary: {args.summary}")
    print(f"   Figures: {args.figdir / 'coffin_parity.png'}")
    print(f"            {args.figdir / 'coffin_strainlife_byT.png'}")


if __name__ == "__main__":
    main()
