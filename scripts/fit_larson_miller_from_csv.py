#!/usr/bin/env python3
"""
Fit and compare rupture-life models for Inconel 718.

Models
------
1) LMP-Linear:
   log10(sigma) = alpha + beta * P,  P = 1e-3 * T * (C + log10 t_r)

2) LMP-Quadratic:
   log10(sigma) = a + b * P + c * P^2,  P = 1e-3 * T * (C + log10 t_r)

3) Manson–Haferd:
   log10(sigma) = A + B * ((T - T*) * (log10 t_r + C*))

Input CSV
---------
data/processed/in718_creep_rupture_isothermal_SI.csv
Columns (Celsius, MPa, hours); comment lines starting with '#' are allowed:
- temperature_C, stress_MPa, time_to_rupture_h

Outputs
-------
models/calibrations/rupture/in718_rupture_LMP_linear.yaml
models/calibrations/rupture/in718_rupture_LMP_quadratic.yaml
models/calibrations/rupture/in718_rupture_MansonHaferd.yaml
models/calibrations/rupture/in718_rupture_best.yaml

reports/figures/rupture/rupture_<model>_collapse.png
reports/figures/rupture/rupture_<model>_parity.png
reports/figures/rupture/rupture_<model>_parity_bands.png
reports/figures/rupture/rupture_<model>_residuals.png

reports/calibration/rupture/rupture_compare_metrics.txt
"""

import os, sys, yaml, math, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.abspath("."))

SCALE  = 1e-3  # LMP scaling
CAL_DIR = os.path.join("models", "calibrations", "rupture")
FIG_DIR = os.path.join("reports", "figures", "rupture")
REP_DIR = os.path.join("reports", "calibration", "rupture")


# ---------------- I/O ----------------
def read_data(csv_path):
    df = pd.read_csv(csv_path, comment="#", engine="python",
                     skip_blank_lines=True, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    needed = ["temperature_C", "stress_MPa", "time_to_rupture_h"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing columns: {miss}. Found: {list(df.columns)}")
    df = df[needed].dropna().copy()
    df["T_K"] = df["temperature_C"] + 273.15
    return df


# ---------------- Metrics ----------------
def parity_metrics(tr_true, tr_pred):
    tr_true = np.asarray(tr_true, float)
    tr_pred = np.asarray(tr_pred, float)
    m = np.isfinite(tr_true) & np.isfinite(tr_pred)
    y, yh = np.log10(tr_true[m]), np.log10(tr_pred[m])
    err = yh - y
    r2  = 1 - np.sum(err**2) / np.sum((y - y.mean())**2)
    rmse_dec = float(np.sqrt(np.mean(err**2)))
    mae_dec  = float(np.mean(np.abs(err)))
    within2 = float(np.mean(np.maximum(tr_pred[m]/tr_true[m], tr_true[m]/tr_pred[m]) <= 2.0))
    within3 = float(np.mean(np.maximum(tr_pred[m]/tr_true[m], tr_true[m]/tr_pred[m]) <= 3.0))
    return dict(R2_log=float(r2), RMSE_dec=rmse_dec, MAE_dec=mae_dec,
                within2=within2, within3=within3)

def aicc(n, k, rss):
    return n * math.log(rss / n) + 2*k + (2*k*(k+1))/(max(n-k-1,1))


# ---------------- Fits ----------------
def fit_lmp_linear(stress, T, tr, p0=(2.0, -0.10, 23.0)):
    from scipy.optimize import curve_fit
    def model(X, alpha, beta, C):
        TT, trh = X
        P = SCALE * (TT * (C + np.log10(trh)))
        return alpha + beta * P  # predicts log10 sigma
    y = np.log10(stress)
    popt, _ = curve_fit(model, (T, tr), y, p0=p0, maxfev=60000)
    alpha, beta, C = popt

    def pred_tr(sigma, TT):
        log10_tr = (np.log10(sigma) - alpha)/(beta * SCALE * TT) - C
        return 10.0**log10_tr

    tr_pred = pred_tr(stress, T)
    m = np.isfinite(tr_pred)
    rss = float(np.sum((np.log10(tr_pred[m]) - np.log10(tr[m]))**2))
    return {"name":"LMP_linear","params":{"alpha":float(alpha),"beta":float(beta),"C":float(C),"scale":SCALE},
            "pred_tr":pred_tr, "rss":rss, "k":3}


def fit_lmp_quadratic(stress, T, tr, p0=(2.0, -0.10, 0.002, 23.0)):
    # robust inversion (root-finder) to avoid NaNs
    from scipy.optimize import curve_fit, brentq
    def model(X, a, b, c, C):
        TT, trh = X
        P = SCALE * (TT * (C + np.log10(trh)))
        return a + b * P + c * P**2
    y = np.log10(stress)
    popt, _ = curve_fit(model, (T, tr), y, p0=p0, maxfev=80000)
    a, b, c, C = popt

    def pred_tr_single(sigma, TT):
        ytarget = np.log10(sigma)
        def g(x):  # x = log10 tr
            P = SCALE * (TT * (C + x))
            return a + b*P + c*P*P - ytarget
        lo, hi = -4.0, 9.0
        for _ in range(6):
            if g(lo)*g(hi) <= 0: break
            lo -= 1.0; hi += 1.0
        try:
            x = brentq(g, lo, hi, maxiter=200)
        except Exception:
            # fallback: vertex or linearized
            P = -b/(2*c) if abs(c) > 1e-14 else (ytarget - a)/b
            x = (P/(SCALE*TT)) - C
        return 10.0**x

    def pred_tr(sigma, TT):
        sigma = np.atleast_1d(sigma); TT = np.atleast_1d(TT)
        return np.array([pred_tr_single(si, Ti) for si, Ti in zip(sigma, TT)], dtype=float)

    tr_pred = pred_tr(stress, T)
    m = np.isfinite(tr_pred)
    rss = float(np.sum((np.log10(tr_pred[m]) - np.log10(tr[m]))**2))
    return {"name":"LMP_quadratic","params":{"a":float(a),"b":float(b),"c":float(c),"C":float(C),"scale":SCALE},
            "pred_tr":pred_tr, "rss":rss, "k":4}


def fit_manson_haferd(stress, T, tr, p0=(3.0, -1e-2, 450.0, 20.0)):
    from scipy.optimize import curve_fit
    def model(X, A, B, Tstar, Cstar):
        TT, trh = X
        P = (TT - Tstar) * (np.log10(trh) + Cstar)
        return A + B * P
    y = np.log10(stress)
    popt, _ = curve_fit(model, (T, tr), y, p0=p0, maxfev=80000)
    A, B, Tstar, Cstar = popt

    def pred_tr(sigma, TT):
        y = np.log10(sigma)
        denom = B * (TT - Tstar)
        log10_tr = (y - A)/denom - Cstar
        return 10.0**log10_tr

    tr_pred = pred_tr(stress, T)
    m = np.isfinite(tr_pred)
    rss = float(np.sum((np.log10(tr_pred[m]) - np.log10(tr[m]))**2))
    return {"name":"MansonHaferd","params":{"A":float(A),"B":float(B),"T_star":float(Tstar),"C_star":float(Cstar)},
            "pred_tr":pred_tr, "rss":rss, "k":4}


# ---------------- CV (leave-one-temperature-out) ----------------
def temp_groups(T):
    return np.unique(np.round(T, 1))

def cv_by_temperature(fit_func, stress, T, tr):
    uniq = temp_groups(T)
    errs = []
    for tv in uniq:
        mask = np.round(T,1) != tv
        if mask.sum() < 5:  # avoid degenerate fits
            continue
        model = fit_func(stress[mask], T[mask], tr[mask])
        tr_pred = model["pred_tr"](stress[~mask], T[~mask])
        m = np.isfinite(tr_pred)
        if m.any():
            err = np.log10(tr_pred[m]) - np.log10(tr[~mask][m])
            errs.append(err)
    if not errs:
        return dict(RMSE_dec=np.nan, MAE_dec=np.nan)
    E = np.concatenate(errs)
    return dict(RMSE_dec=float(np.sqrt(np.mean(E**2))), MAE_dec=float(np.mean(np.abs(E))))


# ---------------- Save helpers & plots ----------------
def save_yaml(path, model_name, params, meta):
    with open(path, "w") as f:
        yaml.safe_dump({"model": model_name,
                        "parameters": params,
                        "meta": meta}, f, sort_keys=False)

def make_plots(tag, stress, T, tr, tr_pred, collapse_x, collapse_y):
    os.makedirs(FIG_DIR, exist_ok=True)

    # Parity
    plt.figure()
    plt.loglog(tr, tr_pred, "o")
    lo = min(np.nanmin(tr), np.nanmin(tr_pred))*0.6
    hi = max(np.nanmax(tr), np.nanmax(tr_pred))*1.6
    plt.plot([lo,hi],[lo,hi],"--")
    plt.xlabel("Measured t_r [h]"); plt.ylabel("Predicted t_r [h]")
    plt.title(f"{tag}: predicted vs measured")
    plt.savefig(os.path.join(FIG_DIR, f"rupture_{tag}_parity.png"), dpi=180, bbox_inches="tight")

    # Parity w/ bands
    T_C = T - 273.15
    plt.figure()
    sc = plt.scatter(tr, tr_pred, c=T_C, cmap="viridis")
    for mlt, ls, lab in [(1,"--","1:1"),(2,":","×2"),(1/2,":","÷2"),(3,"-.", "×3"),(1/3,"-.", "÷3")]:
        plt.plot([lo,hi],[lo*mlt,hi*mlt], ls, label=lab)
    plt.xscale("log"); plt.yscale("log"); plt.legend()
    cb = plt.colorbar(sc); cb.set_label("Temperature [°C]")
    plt.xlabel("Measured t_r [h]"); plt.ylabel("Predicted t_r [h]")
    plt.title("Parity (colored by temperature °C)")
    plt.savefig(os.path.join(FIG_DIR, f"rupture_{tag}_parity_bands.png"), dpi=180, bbox_inches="tight")

    # Collapse
    plt.figure()
    plt.plot(collapse_x, collapse_y, "o", label="data")
    if tag != "LMP_quadratic":
        m, b = np.polyfit(collapse_x, collapse_y, 1)
        xx = np.linspace(min(collapse_x)*0.95, max(collapse_x)*1.05, 300)
        plt.plot(xx, m*xx + b, "-", label="fit"); plt.legend()
    plt.xlabel("Collapse variable"); plt.ylabel("log10(stress [MPa])")
    plt.title(f"{tag} collapse")
    plt.savefig(os.path.join(FIG_DIR, f"rupture_{tag}_collapse.png"), dpi=180, bbox_inches="tight")

    # Residuals
    res = np.log10(tr_pred) - np.log10(tr)
    plt.figure()
    plt.scatter(collapse_x, res, s=28)
    plt.axhline(0, ls="--", lw=1)
    plt.xlabel("Collapse variable"); plt.ylabel("Residual (log10 h): pred - meas")
    plt.title(f"{tag}: residuals")
    plt.savefig(os.path.join(FIG_DIR, f"rupture_{tag}_residuals.png"), dpi=180, bbox_inches="tight")


# ---------------- Main ----------------
def main():
    os.makedirs(CAL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REP_DIR, exist_ok=True)

    csv_path = "data/processed/in718_creep_rupture_isothermal_SI.csv"
    print(f"[info] Loading: {csv_path}")
    df = read_data(csv_path)
    stress = df["stress_MPa"].to_numpy(float)
    T      = df["T_K"].to_numpy(float)
    tr     = df["time_to_rupture_h"].to_numpy(float)
    print(f"[info] Rows={len(df)}; T[K]={T.min():.1f}-{T.max():.1f}; sigma[MPa]={stress.min():.1f}-{stress.max():.1f}")

    # Fit all models
    models = [
        fit_lmp_linear(stress, T, tr),
        fit_lmp_quadratic(stress, T, tr),
        fit_manson_haferd(stress, T, tr),
    ]

    # Score + plots + YAML for each
    rows = []
    for m in models:
        tr_pred = m["pred_tr"](stress, T)
        met = parity_metrics(tr, tr_pred)
        n, k = len(tr), m["k"]
        rss  = m["rss"]
        met["AICc"] = float(aicc(n, k, rss))

        # collapse variable for plots
        if m["name"].startswith("LMP"):
            C = m["params"]["C"]; P = SCALE * (T * (C + np.log10(tr)))
            collapse_x = P; collapse_y = np.log10(stress)
        else:
            Tstar = m["params"]["T_star"]; Cstar = m["params"]["C_star"]
            collapse_x = (T - Tstar) * (np.log10(tr) + Cstar)
            collapse_y = np.log10(stress)

        make_plots(m["name"], stress, T, tr, tr_pred, collapse_x, collapse_y)

        # CV by temperature
        cv = (cv_by_temperature(fit_lmp_linear, stress, T, tr) if m["name"]=="LMP_linear"
              else cv_by_temperature(fit_lmp_quadratic, stress, T, tr) if m["name"]=="LMP_quadratic"
              else cv_by_temperature(fit_manson_haferd, stress, T, tr))
        met["CV_RMSE_dec"] = float(cv["RMSE_dec"]); met["CV_MAE_dec"] = float(cv["MAE_dec"])

        # Save per-model YAML
        meta = {
            "fit_datetime": datetime.utcnow().isoformat()+"Z",
            "csv_source": os.path.abspath(csv_path),
            "units": {"stress":"MPa","temperature":"K (from C)","t_r":"hours"},
            "metrics": met
        }
        save_yaml(os.path.join(CAL_DIR, f"in718_rupture_{m['name']}.yaml"),
                  m["name"], m["params"], meta)

        rows.append({"model": m["name"], **met})

    # Pick best: lowest AICc; tie-breaker lowest CV_RMSE_dec
    rows_sorted = sorted(rows, key=lambda r: (r["AICc"], r["CV_RMSE_dec"]))
    best = rows_sorted[0]["model"]

    # Copy best yaml to canonical name
    src = os.path.join(CAL_DIR, f"in718_rupture_{best}.yaml")
    dst = os.path.join(CAL_DIR, "in718_rupture_best.yaml")
    shutil.copyfile(src, dst)

    # Write comparison table
    lines = ["Model comparison (lower is better for errors/AICc)\n"]
    hdr = f"{'model':15s}  {'R2_log':>7s}  {'RMSE_dec':>9s}  {'within×2':>8s}  {'within×3':>8s}  {'AICc':>9s}  {'CV_RMSE':>9s}"
    lines.append(hdr)
    for r in rows_sorted:
        lines.append(f"{r['model']:15s}  {r['R2_log']:7.3f}  {r['RMSE_dec']:9.3f}  {100*r['within2']:8.1f}  {100*r['within3']:8.1f}  {r['AICc']:9.2f}  {r['CV_RMSE_dec']:9.3f}")
    out_txt = "\n".join(lines)
    with open(os.path.join(REP_DIR, "rupture_compare_metrics.txt"), "w") as f:
        f.write(out_txt)

    print(out_txt)
    print(f"[best] {best} → saved as {dst}")

if __name__ == "__main__":
    main()
