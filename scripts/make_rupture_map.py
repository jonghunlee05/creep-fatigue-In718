#!/usr/bin/env python3
"""
make_rupture_map.py — draw iso-life (hours) curves vs temperature for IN718 using LMP fit
Outputs: reports/figures/in718_rupture_map_isolife.png
"""
import numpy as np, yaml, matplotlib.pyplot as plt

def load_params(path="models/calibrations/in718_lmp.yaml"):
    d = yaml.safe_load(open(path)); p = d["parameters"]
    return p["alpha"], p["beta"], p["C"], p.get("scale", 1e-3)

def sigma_from_T_tr(T_K, tr_h, alpha, beta, C, scale):
    LMP_scaled = scale * T_K * (C + np.log10(tr_h))
    log10_sigma = alpha + beta * LMP_scaled
    return 10.0 ** log10_sigma

def main():
    alpha, beta, C, scale = load_params()
    T_C = np.linspace(500, 750, 200)      # adjust range as needed
    T_K = T_C + 273.15
    lives_h = [10, 100, 1e3, 1e4, 5e4]    # iso-life lines (hours)

    for tr in lives_h:
        sigma = sigma_from_T_tr(T_K, tr, alpha, beta, C, scale)
        plt.plot(T_C, sigma, label=f"{tr:g} h")

    plt.xlabel("Temperature [°C]")
    plt.ylabel("Stress [MPa]")
    plt.title("IN718 Creep–Rupture Map (Larson–Miller)")
    plt.legend(title="Iso-life")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig("reports/figures/in718_rupture_map_isolife.png", dpi=180, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
