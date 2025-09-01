#!/usr/bin/env python3
"""
fit_larson_miller_from_csv.py — Calibrate Larson–Miller rupture model for Inconel 718

Purpose
-------
Fit alpha, beta, C in a simple linear relation between log10(stress) and a scaled LMP:
    LMP = T * (C + log10(t_r[h]))
    log10(sigma[MPa]) = alpha + beta * (1e-3 * LMP)

Notes
-----
• Your CSV uses Celsius for temperature: 'temperature_C'.
  This script converts to Kelvin internally: T_K = temperature_C + 273.15.
• Time to rupture is in hours: 'time_to_rupture_h' (keep it that way; LMP expects hours).

Outputs
-------
1) models/calibrations/in718_lmp.yaml
2) reports/figures/lmp_sigma_vs_LMP.png
3) reports/figures/lmp_pred_vs_meas_tr.png
4) reports/calibration/lmp_fit_in718.txt
"""

import os, sys, yaml
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime

# Allow "from models...." imports when run from repo root
sys.path.append(os.path.abspath("."))

# --- Model utilities ---------------------------------------------------------
# (kept tiny and local so this script is self-contained to run immediately)
SCALE = 1e-3  # LMP scaling to keep numbers ~ O(10)

def lmp(T_K: np.ndarray, t_r_h: np.ndarray, C: float) -> np.ndarray:
    """Larson–Miller parameter: LMP = T * (C + log10(t_r[h]))"""
    return T_K * (C + np.log10(t_r_h))

def _model_for_curve_fit(X, alpha, beta, C):
    """Return log10(sigma) given (T, t_r) and parameters."""
    T, t_r_h = X
    LMPs = SCALE * lmp(T, t_r_h, C)
    return alpha + beta * LMPs

def fit_lmp(stress_MPa, T_K, t_r_h, p0=(2.0, 1e-2, 20.0)):
    """Fit (alpha, beta, C) by least squares on log10(sigma)."""
    from scipy.optimize import curve_fit
    y = np.log10(np.asarray(stress_MPa, float))
    T = np.asarray(T_K, float); tr = np.asarray(t_r_h, float)
    popt, pcov = curve_fit(_model_for_curve_fit, (T, tr), y, p0=p0, maxfev=50000)
    return popt, pcov

def predict_time_to_rupture_h(stress_MPa, T_K, alpha, beta, C):
    """Invert the fitted relation to compute t_r for given (sigma, T)."""
    log10_sigma = np.log10(np.asarray(stress_MPa, float))
    T = np.asarray(T_K, float)
    log10_tr = (log10_sigma - alpha) / (beta * SCALE * T) - C
    return 10.0 ** log10_tr
# ----------------------------------------------------------------------------

def main():
    # --- Paths & I/O dirs
    csv_path = "data/processed/in718_creep_rupture_isothermal_SI.csv"
    os.makedirs("models/calibrations", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/calibration", exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"[error] CSV not found at: {csv_path}")
        return

    # --- Load CSV (your columns)
    print(f"[info] Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    required = ["temperature_C", "stress_MPa", "time_to_rupture_h"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[error] Missing columns: {missing}")
        print(f"[hint] Found columns: {list(df.columns)}")
        return

    # Convert to modeling columns
    df = df[required].dropna().copy()
    df["T_K"] = df["temperature_C"] + 273.15
    stress = df["stress_MPa"].to_numpy(dtype=float)
    T = df["T_K"].to_numpy(dtype=float)
    tr = df["time_to_rupture_h"].to_numpy(dtype=float)

    print(f"[info] Loaded rows: {len(df)}; T range (K): {T.min():.1f}–{T.max():.1f}; "
          f"σ range (MPa): {stress.min():.1f}–{stress.max():.1f}")

    # --- Fit
    (alpha, beta, C), pcov = fit_lmp(stress, T, tr, p0=(2.0, 1e-2, 20.0))
    print(f"[fit] alpha={alpha:.4f}, beta={beta:.4e}, C={C:.3f}")

    # --- Figures
    LMP_scaled = SCALE * lmp(T, tr, C)
    plt.figure()
    plt.plot(LMP_scaled, np.log10(stress), "o", label="data")
    xg = np.linspace(LMP_scaled.min()*0.95, LMP_scaled.max()*1.05, 300)
    plt.plot(xg, alpha + beta*xg, "-", label="fit")
    plt.xlabel("Scaled LMP = 1e-3 * T * (C + log10 t_r[h])")
    plt.ylabel("log10(stress [MPa])")
    plt.title("IN718 Larson–Miller fit")
    plt.legend()
    fig1 = "reports/figures/lmp_sigma_vs_LMP.png"
    plt.savefig(fig1, dpi=180, bbox_inches="tight")

    tr_pred = predict_time_to_rupture_h(stress, T, alpha, beta, C)
    plt.figure()
    plt.loglog(tr, tr_pred, "o")
    lo = min(tr.min(), tr_pred.min()) * 0.7
    hi = max(tr.max(), tr_pred.max()) * 1.3
    plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel("Measured t_r [h]")
    plt.ylabel("Predicted t_r [h]")
    plt.title("Larson–Miller: predicted vs measured")
    fig2 = "reports/figures/lmp_pred_vs_meas_tr.png"
    plt.savefig(fig2, dpi=180, bbox_inches="tight")
    
    
    # --- Diagnostics: errors in log-space and factor-of-k accuracy
    import numpy as np
    from math import log10

    def predict_time_to_rupture_h(sigma_MPa, T_K, alpha, beta, C, scale=1e-3):
        log10_sigma = np.log10(np.asarray(sigma_MPa, float))
        T = np.asarray(T_K, float)
        log10_tr = (log10_sigma - alpha) / (beta * scale * T) - C
        return 10.0 ** log10_tr

    tr_pred = predict_time_to_rupture_h(stress, T, alpha, beta, C)

    # Errors in decades (log10 hours)
    err_dec = np.log10(tr_pred) - np.log10(tr)
    r2_log = 1 - np.sum((err_dec)**2) / np.sum((np.log10(tr) - np.log10(tr).mean())**2)
    rmse_dec = np.sqrt(np.mean(err_dec**2))
    mae_dec = np.mean(np.abs(err_dec))

    def frac_within(k):
        return float(np.mean(np.maximum(tr_pred/tr, tr/tr_pred) <= k))

    f2 = frac_within(2.0)
    f3 = frac_within(3.0)

    print(f"[diag] R^2 (log-life) = {r2_log:.3f}")
    print(f"[diag] RMSE = {rmse_dec:.3f} decades (~×{10**rmse_dec:.2f})")
    print(f"[diag] MAE  = {mae_dec:.3f} decades (~×{10**mae_dec:.2f})")
    print(f"[diag] within ×2: {100*f2:.1f}%  | within ×3: {100*f3:.1f}%")

    # Residuals vs scaled LMP
    LMP_scaled = 1e-3 * (T * (C + np.log10(tr)))
    plt.figure()
    plt.scatter(LMP_scaled, err_dec, s=28)
    plt.axhline(0, ls="--", lw=1)
    plt.xlabel("Scaled LMP")
    plt.ylabel("Residual (log10 h): pred - meas")
    plt.title("Residuals vs LMP")
    plt.savefig("reports/figures/lmp_residuals_vs_LMP.png", dpi=180, bbox_inches="tight")

    # Parity colored by temperature (°C)
    T_C = T - 273.15
    plt.figure()
    sc = plt.scatter(tr, tr_pred, c=T_C, cmap="viridis")
    lo = min(tr.min(), tr_pred.min()) * 0.7
    hi = max(tr.max(), tr_pred.max()) * 1.3
    plt.plot([lo, hi], [lo, hi], "--", lw=1)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Measured t_r [h]"); plt.ylabel("Predicted t_r [h]")
    plt.title("Parity (colored by temperature °C)")
    cb = plt.colorbar(sc); cb.set_label("Temperature [°C]")
    plt.savefig("reports/figures/lmp_parity_colored_by_T.png", dpi=180, bbox_inches="tight")


    # --- Save YAML + text report
    calib = {"alpha": float(alpha), "beta": float(beta), "C": float(C), "scale": float(SCALE)}
    meta = {
        "model": "Larson–Miller rupture",
        "material": "Inconel 718",
        "units": {"stress": "MPa", "temperature": "K (converted from C)", "t_r": "hours"},
        "fit_datetime": datetime.utcnow().isoformat() + "Z",
        "csv_source": os.path.abspath(csv_path),
        "columns": {"temperature_C": "temperature_C", "stress": "stress_MPa", "t_r": "time_to_rupture_h"},
    }
    with open("models/calibrations/in718_lmp.yaml", "w") as f:
        yaml.safe_dump({"parameters": calib, "meta": meta}, f, sort_keys=False)

    with open("reports/calibration/lmp_fit_in718.txt", "w") as f:
        f.write("\n".join([
            "Larson–Miller rupture — Inconel 718",
            f"alpha = {alpha:.6f}",
            f"beta  = {beta:.6e}",
            f"C     = {C:.3f}",
            f"Figure 1: {os.path.abspath(fig1)}",
            f"Figure 2: {os.path.abspath(fig2)}",
            f"YAML saved to: {os.path.abspath('models/calibrations/in718_lmp.yaml')}",
        ]))

    print("[done] Saved:")
    print(f"  - {fig1}")
    print(f"  - {fig2}")
    print(f"  - models/calibrations/in718_lmp.yaml")
    print(f"  - reports/calibration/lmp_fit_in718.txt")

if __name__ == "__main__":
    main()
