#!/usr/bin/env python3
"""
Fit Basquin S–N curves for IN718 HCF (per temperature).

Model (per temperature T):
    log10(Nf) = a(T) - k(T) * log10(Sigma_a[MPa])

CSV expected at:
    data/processed/in718_fatigue_HCF_isothermal_SI.csv

Typical columns (comments allowed):
    temperature_C, stress_MPa, cycles_to_failure
Robust guessing is used; customize CANDIDATES below if needed.
"""

import os, sys, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.abspath("."))

# ---- Directories ----
CAL_DIR = os.path.join("models", "calibrations", "fatigue")
FIG_DIR = os.path.join("reports", "figures", "fatigue")
REP_DIR = os.path.join("reports", "calibration", "fatigue")

# ---- Column name candidates ----
CAND = {
    "T_C": ["temperature_C", "temp_C", "T_C", "temperature_c"],
    "S_MPa": ["stress_MPa", "sigma_a_MPa", "stress_amplitude_MPa", "S_MPa", "sigma_MPa", "stress"],
    "Nf": ["cycles_to_failure", "Nf", "life_cycles", "cycles", "N_cycles"]
}

def guess_col(df, keys):
    for k in keys:
        if k in df.columns: return k
    raise KeyError(f"Missing expected columns. Looked for: {keys}. Found: {list(df.columns)}")

def load_data(csv_path):
    df = pd.read_csv(csv_path, comment="#", engine="python",
                     skip_blank_lines=True, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    tcol = guess_col(df, CAND["T_C"])
    scol = guess_col(df, CAND["S_MPa"])
    ncol = guess_col(df, CAND["Nf"])
    df = df[[tcol, scol, ncol]].dropna().rename(
        columns={tcol:"temperature_C", scol:"stress_MPa", ncol:"Nf"}
    )
    # keep only positive, sensible rows
    df = df[(df["stress_MPa"]>0) & (df["Nf"]>0)]
    return df

def fit_basquin_loglog(stress_MPa, Nf):
    """
    Fit log10(Nf) = a - k * log10(S).
    Returns: a, k and predictions for provided data.
    """
    x = np.log10(np.asarray(stress_MPa, float))
    y = np.log10(np.asarray(Nf, float))
    # linear y = A + B*x  with B expected negative → we return k = -B
    A, B = np.polyfit(x, y, 1)
    a = float(A)
    k = float(-B)
    yhat = A + B * x
    return a, k, 10**yhat

def metrics_log_cycles(Nf_true, Nf_pred):
    y = np.log10(Nf_true); yh = np.log10(Nf_pred)
    err = yh - y
    r2 = 1 - np.sum(err**2) / np.sum((y - y.mean())**2)
    rmse_dec = float(np.sqrt(np.mean(err**2)))
    within2 = float(np.mean(np.maximum(Nf_pred/Nf_true, Nf_true/Nf_pred) <= 2.0))
    within3 = float(np.mean(np.maximum(Nf_pred/Nf_true, Nf_true/Nf_pred) <= 3.0))
    return dict(R2_log=float(r2), RMSE_dec=rmse_dec, within2=within2, within3=within3)

def save_yaml(params_by_T, csv_path, metrics_by_T):
    os.makedirs(CAL_DIR, exist_ok=True)
    payload = {
        "model": "Basquin S–N (per temperature)",
        "material": "Inconel 718",
        "equation": "log10(Nf) = a(T) - k(T)*log10(stress_MPa)",
        "parameters": params_by_T,   # dict T_C -> {a, k}
        "meta": {
            "fit_datetime": datetime.utcnow().isoformat()+"Z",
            "csv_source": os.path.abspath(csv_path),
            "units": {"stress":"MPa", "temperature":"°C", "Nf":"cycles"},
            "metrics_by_temperature": metrics_by_T
        }
    }
    out = os.path.join(CAL_DIR, "in718_hcf_basquin.yaml")
    with open(out, "w") as f: yaml.safe_dump(payload, f, sort_keys=False)
    print(f"[save] {out}")

def plot_per_temperature(df, params_by_T):
    os.makedirs(FIG_DIR, exist_ok=True)
    # combined plot (log-log S–N) per T
    plt.figure()
    for T, g in df.groupby(np.round(df["temperature_C"],1)):
        a = params_by_T[str(T)]["a"]; k = params_by_T[str(T)]["k"]
        S = g["stress_MPa"].to_numpy()
        Nf_pred = 10**(a - k*np.log10(S))
        plt.loglog(g["Nf"], g["stress_MPa"], "o", label=f"{T:.0f} °C data")
        # draw line across range
        Sgrid = np.linspace(S.min()*0.9, S.max()*1.1, 200)
        Ngrid = 10**(a - k*np.log10(Sgrid))
        plt.loglog(Ngrid, Sgrid, "-")
    plt.xlabel("Cycles to failure, Nf (log)")
    plt.ylabel("Stress amplitude [MPa] (log)")
    plt.title("IN718 HCF — Basquin per temperature")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, "hcf_basquin_SN_per_T.png"), dpi=180, bbox_inches="tight")

def parity_plot(df, params_by_T):
    os.makedirs(FIG_DIR, exist_ok=True)
    Nf = df["Nf"].to_numpy()
    Nf_pred = []
    Tlab = []
    for _, r in df.iterrows():
        T = round(r["temperature_C"], 1)
        a = params_by_T[str(T)]["a"]; k = params_by_T[str(T)]["k"]
        Nf_pred.append(10**(a - k*np.log10(r["stress_MPa"])))
        Tlab.append(T)
    Nf_pred = np.array(Nf_pred)
    Tlab = np.array(Tlab)

    plt.figure()
    sc = plt.scatter(Nf, Nf_pred, c=Tlab, cmap="viridis")
    lo = min(Nf.min(), Nf_pred.min())*0.6
    hi = max(Nf.max(), Nf_pred.max())*1.6
    for mlt, ls, lab in [(1,"--","1:1"),(2,":","×2"),(1/2,":","÷2"),(3,"-.", "×3"),(1/3,"-.", "÷3")]:
        plt.plot([lo,hi],[lo*mlt,hi*mlt], ls, label=lab)
    plt.xscale("log"); plt.yscale("log"); plt.legend()
    cb = plt.colorbar(sc); cb.set_label("Temperature [°C]")
    plt.xlabel("Measured Nf (cycles)"); plt.ylabel("Predicted Nf (cycles)")
    plt.title("HCF Basquin — Parity by Temperature")
    plt.savefig(os.path.join(FIG_DIR, "hcf_basquin_parity.png"), dpi=180, bbox_inches="tight")

def main():
    os.makedirs(CAL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REP_DIR, exist_ok=True)

    csv_path = "data/processed/in718_fatigue_HCF_isothermal_SI.csv"
    df = load_data(csv_path)
    print(f"[info] rows={len(df)}, temps={sorted(df['temperature_C'].round(1).unique())}")

    params_by_T = {}
    metrics_by_T = {}
    rows = []

    for T, g in df.groupby(np.round(df["temperature_C"], 1)):
        a, k, Nf_pred = fit_basquin_loglog(g["stress_MPa"], g["Nf"])
        params_by_T[str(float(T))] = {"a": float(a), "k": float(k)}
        met = metrics_log_cycles(g["Nf"].to_numpy(), Nf_pred)
        metrics_by_T[str(float(T))] = met
        rows.append({"T_C": float(T), **met})

    # Save YAML + plots
    save_yaml(params_by_T, csv_path, metrics_by_T)
    plot_per_temperature(df, params_by_T)
    parity_plot(df, params_by_T)

    # Write a small metrics table
    rep_lines = ["HCF Basquin — per-temperature metrics\n",
                 f"{'T_C':>8s}  {'R2_log':>7s}  {'RMSE_dec':>9s}  {'within×2':>8s}  {'within×3':>8s}"]
    for r in sorted(rows, key=lambda x: x["T_C"]):
        rep_lines.append(f"{r['T_C']:8.1f}  {r['R2_log']:7.3f}  {r['RMSE_dec']:9.3f}  {100*r['within2']:8.1f}  {100*r['within3']:8.1f}")
    with open(os.path.join(REP_DIR, "hcf_basquin_metrics.txt"), "w") as f:
        f.write("\n".join(rep_lines))

    print("\n".join(rep_lines))
    print(f"[done] Saved figures to {FIG_DIR} and YAML to {CAL_DIR}")

if __name__ == "__main__":
    main()
