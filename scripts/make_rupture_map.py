#!/usr/bin/env python3
"""
Draw iso-life curves using models/calibrations/in718_rupture_best.yaml.
Supports: LMP-Linear, LMP-Quadratic, MansonHaferd.
"""
import yaml, numpy as np, matplotlib.pyplot as plt

def load_best(path="models/calibrations/in718_rupture_best.yaml"):
    d = yaml.safe_load(open(path, "r"))
    return d["model"], d["parameters"]

def sigma_from_T_tr(model, params, T_K, tr_h):
    name = model
    if name == "LMP_linear":
        alpha, beta, C, scale = params["alpha"], params["beta"], params["C"], params.get("scale", 1e-3)
        P = scale * (T_K * (C + np.log10(tr_h)))
        log10_sigma = alpha + beta * P
        return 10.0**log10_sigma
    if name == "LMP_quadratic":
        a, b, c, C, scale = params["a"], params["b"], params["c"], params["C"], params.get("scale",1e-3)
        P = scale * (T_K * (C + np.log10(tr_h)))
        log10_sigma = a + b*P + c*(P**2)
        return 10.0**log10_sigma
    if name == "MansonHaferd":
        A, B, Tstar, Cstar = params["A"], params["B"], params["T_star"], params["C_star"]
        P = (T_K - Tstar) * (np.log10(tr_h) + Cstar)
        log10_sigma = A + B * P
        return 10.0**log10_sigma
    raise ValueError(f"Unsupported model: {name}")

def main():
    model, params = load_best()
    T_C = np.linspace(500, 750, 220)
    T_K = T_C + 273.15
    lives = [10, 100, 1e3, 1e4, 5e4]
    for tr in lives:
        sigma = sigma_from_T_tr(model, params, T_K, tr)
        plt.plot(T_C, sigma, label=f"{tr:g} h")
    plt.xlabel("Temperature [°C]")
    plt.ylabel("Stress [MPa]")
    plt.title(f"IN718 Creep–Rupture Map ({model})")
    plt.legend(title="Iso-life")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig("reports/figures/in718_rupture_map_best.png", dpi=180, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
