#!/usr/bin/env python3
"""
fit_norton_from_csv.py — Calibrate Norton (power-law) creep for Inconel 718

Purpose
-------
Read processed creep data (stress, temperature, creep rate) and fit Norton’s law:
    epsdot = A * stress^n * exp(-Q / (R*T))

Outputs
-------
1) models/calibrations/in718_norton.yaml  (A, n, Q with 1-sigma and metadata)
2) reports/figures/norton_fit_in718_loglog.png
3) reports/figures/norton_pred_vs_meas.png
4) reports/calibration/norton_fit_in718.txt  (human-readable summary)

Notes
-----
• If your dataset has a SINGLE temperature, A and Q are not separately identifiable.
  This script will, by default, FIX Q (300 kJ/mol) and fit A,n. You can override with --fix-q.
• For multi-temperature datasets, it fits A, n, and Q together.
• Keep the mechanistic function in models/creep/norton.py; this script just orchestrates I/O + fitting.

Usage
-----
python scripts/fit_norton_from_csv.py \
  --csv data/processed/creep_in718.csv \
  --stress-col stress_MPa --temp-col T_K --rate-col epsdot_1_per_s
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import your model
import sys
sys.path.append(os.path.abspath("."))  # allow repo-root imports
from models.creep.norton import R, norton_law, fit_norton  # noqa: E402


def guess_column(df, candidates):
    """Pick the first existing column from a list of candidate names."""
    for c in candidates:
        if c in df.columns: return c
    raise KeyError(f"None of the candidate columns found: {candidates}")


def ensure_dirs():
    os.makedirs("models/calibrations", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/calibration", exist_ok=True)


def fit_with_strategy(stress, T, epsdot, fix_q=None):
    """
    Fit Norton parameters with sensible defaults.
    If fix_q is not None, keep Q fixed and fit A, n only via curve_fit by wrapping Q.
    """
    stress = np.asarray(stress, float)
    T = np.asarray(T, float)
    epsdot = np.asarray(epsdot, float)

    if fix_q is not None:
        # Fit A, n with Q fixed
        def model_fixed_Q(X, A, n):
            s, temp = X
            return norton_law(s, temp, A, n, fix_q)
        # Initial guess
        p0 = (1e-7, 5.0)
        popt, pcov = None, None
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(model_fixed_Q, (stress, T), epsdot, p0=p0, maxfev=20000)
        A_fit, n_fit = popt
        Q_fit = fix_q
        return np.array([A_fit, n_fit, Q_fit]), pcov, True
    else:
        # Fit A, n, Q together
        p0 = (1e-7, 5.0, 3.0e5)  # A, n, Q[J/mol]
        popt, pcov = fit_norton(stress, T, epsdot, p0=p0)
        return popt, pcov, False


def summarize_fit(stress, T, epsdot, params):
    A, n, Q = params
    pred = norton_law(stress, T, A, n, Q)
    # R^2 in log space is more meaningful for rates spanning orders of magnitude
    y = np.log10(epsdot)
    yhat = np.log10(pred)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_log = 1 - ss_res / ss_tot
    return pred, r2_log


def save_yaml(out_path, params, sigmas, meta):
    A, n, Q = params
    sA, sn, sQ = sigmas
    data = {
        "model": "Norton (power-law) creep",
        "material": "Inconel 718",
        "units": {
            "stress": "MPa",
            "temperature": "K",
            "rate": "1/s",
            "Q": "J/mol"
        },
        "parameters": {
            "A": {"value": float(A), "stdev": float(sA) if sA is not None else None},
            "n": {"value": float(n), "stdev": float(sn) if sn is not None else None},
            "Q": {"value": float(Q), "stdev": float(sQ) if sQ is not None else None},
        },
        "meta": meta
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/creep_in718.csv")
    parser.add_argument("--stress-col", default=None)
    parser.add_argument("--temp-col", default=None)
    parser.add_argument("--rate-col", default=None)
    parser.add_argument("--fix-q", type=float, default=None,
                        help="Fix Q [J/mol]. If not set and data has single T, default 3.0e5.")
    args = parser.parse_args()

    ensure_dirs()
    df = pd.read_csv(args.csv)

    stress_col = args.stress_col or guess_column(df, ["stress_MPa", "sigma_MPa", "stress"])
    temp_col   = args.temp_col   or guess_column(df, ["T_K", "temperature_K", "temp_K"])
    rate_col   = args.rate_col   or guess_column(df, ["epsdot_1_per_s", "creep_rate_1_per_s", "epsdot"])

    stress = df[stress_col].to_numpy()
    T = df[temp_col].to_numpy()
    epsdot = df[rate_col].to_numpy()

    unique_T = np.unique(np.round(T, 2))
    single_T = len(unique_T) == 1

    fix_q = args.fix_q
    if fix_q is None and single_T:
        fix_q = 3.0e5  # default 300 kJ/mol for single-T datasets
        print(f"[info] single-T dataset detected (T={unique_T[0]} K). Using default fixed Q={fix_q:.2e} J/mol")

    popt, pcov, used_fixed_Q = fit_with_strategy(stress, T, epsdot, fix_q=fix_q)

    # 1-sigma from covariance (if available)
    sigmas = (None, None, None)
    if pcov is not None:
        try:
            sigmas = tuple(np.sqrt(np.diag(pcov)))
            if used_fixed_Q:  # covariance is for (A, n) only; pad Q sigma with None
                sigmas = (sigmas[0], sigmas[1], None)
        except Exception:
            pass

    A, n, Q = popt
    pred, r2_log = summarize_fit(stress, T, epsdot, popt)

    # Save figures
    plt.figure()
    plt.loglog(stress, epsdot, "o", label="data")
    # for plotting, use a line over a stress range
    sgrid = np.linspace(stress.min()*0.9, stress.max()*1.1, 200)
    Tplot = np.median(T)
    plt.loglog(sgrid, norton_law(sgrid, Tplot, A, n, Q), "-", label=f"fit @ ~{Tplot:.0f} K")
    plt.xlabel("Stress [MPa] (log)")
    plt.ylabel("Creep rate [1/s] (log)")
    plt.title("Norton fit — log–log")
    plt.legend()
    fig1_path = "reports/figures/norton_fit_in718_loglog.png"
    plt.savefig(fig1_path, dpi=180, bbox_inches="tight")

    plt.figure()
    plt.loglog(epsdot, pred, "o")
    lims = [min(epsdot.min(), pred.min())*0.8, max(epsdot.max(), pred.max())*1.2]
    plt.plot(lims, lims, "--")
    plt.xlabel("Measured rate [1/s] (log)")
    plt.ylabel("Predicted rate [1/s] (log)")
    plt.title("Predicted vs Measured")
    fig2_path = "reports/figures/norton_pred_vs_meas.png"
    plt.savefig(fig2_path, dpi=180, bbox_inches="tight")

    # Save YAML calibration
    meta = {
        "fit_datetime": datetime.utcnow().isoformat() + "Z",
        "csv_source": os.path.abspath(args.csv),
        "columns": {"stress": stress_col, "temperature": temp_col, "rate": rate_col},
        "single_temperature": bool(single_T),
        "used_fixed_Q": bool(used_fixed_Q),
        "R": R,
        "r2_log": float(r2_log)
    }
    yaml_path = "models/calibrations/in718_norton.yaml"
    save_yaml(yaml_path, popt, sigmas, meta)

    # Save a small text report
    report = [
        "Norton power-law creep — Inconel 718",
        f"A = {A:.3e}  [1/s·MPa^-n]",
        f"n = {n:.3f}  [-]",
        f"Q = {Q:.3e}  [J/mol] {'(fixed)' if used_fixed_Q else ''}",
        f"R2 (log-space) = {r2_log:.4f}",
        f"Figure (log–log): {os.path.abspath(fig1_path)}",
        f"Figure (pred vs meas): {os.path.abspath(fig2_path)}",
        f"YAML saved to: {os.path.abspath(yaml_path)}",
    ]
    with open("reports/calibration/norton_fit_in718.txt", "w") as f:
        f.write("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
