#!/usr/bin/env python3
import streamlit as st
from pathlib import Path
import yaml, math
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from predict_time_fraction import (
    _load_cm_params, _invert_cm_for_Nf,
    _load_basquin_params, _basquin_Nf,
    _load_norton_params, _norton_rate,
    _rupture_time_from_yaml
)

# --- Load YAMLs ---
def load_yaml(p: Path):
    with open(p, "r") as f:
        return yaml.safe_load(f)

st.set_page_config(page_title="In718 Life Prediction Demo", layout="wide")
st.title("Inconel 718 Creep–Fatigue Life Predictor")

st.sidebar.header("Inputs")

# --- Fatigue inputs ---
T_K = st.sidebar.number_input("Temperature T [K]", min_value=300.0, max_value=1400.0, value=977.0, step=1.0)
fatigue_model = st.sidebar.radio("Fatigue model", ["LCF (Coffin–Manson)", "HCF (Basquin)"])

if "LCF" in fatigue_model:
    eps_total = st.sidebar.number_input("Total strain amplitude Δε/2", min_value=0.001, max_value=0.02, value=0.004, step=0.001, format="%.4f")
    sigma_a = None
else:
    sigma_a = st.sidebar.number_input("Stress amplitude σ_a [MPa]", min_value=100.0, max_value=1200.0, value=600.0, step=10.0)
    eps_total = None

# --- Creep inputs ---
sigma_hold = st.sidebar.number_input("Hold stress [MPa]", min_value=100.0, max_value=1200.0, value=550.0, step=10.0)
t_hold_s = st.sidebar.number_input("Hold time per cycle [s]", min_value=0.0, max_value=3600.0, value=5.0, step=1.0)

creep_mode = st.sidebar.radio("Creep damage mode", ["rupture", "rate"])
epsilon_crit = st.sidebar.number_input("ε_crit (for rate mode)", min_value=0.005, max_value=0.1, value=0.02, step=0.005)

# --- File paths ---
coffin_path  = Path("models/calibrations/fatigue/in718_lcf_coffin_manson.yaml")
basquin_path = Path("models/calibrations/fatigue/in718_hcf_basquin.yaml")
norton_path  = Path("models/calibrations/creep/in718_norton.yaml")
rupture_path = Path("models/calibrations/rupture/in718_rupture_best.yaml")

# --- Fatigue life ---
if eps_total is not None:
    cm = _load_cm_params(coffin_path, T_K=T_K)
    Nf = _invert_cm_for_Nf(eps_total, cm)
    fatigue_desc = f"LCF CM (Δε/2={eps_total})"
else:
    bp = _load_basquin_params(basquin_path, T_K=T_K)
    Nf = _basquin_Nf(sigma_a, bp)
    fatigue_desc = f"HCF Basquin (σ_a={sigma_a} MPa)"

# --- Creep damage ---
dc_per_cycle = 0.0
creep_desc = "none"
if t_hold_s > 0:
    if creep_mode == "rupture":
        tr = _rupture_time_from_yaml(rupture_path, sigma_MPa=sigma_hold, T_K=T_K)
        dc_per_cycle = t_hold_s / tr
        creep_desc = f"rupture-based (tr≈{tr/3600:.2e} h)"
    else:
        nt = _load_norton_params(norton_path)
        edot = _norton_rate(sigma_hold, T_K, nt)
        dc_per_cycle = (edot * t_hold_s) / epsilon_crit
        creep_desc = f"rate-based (ε̇≈{edot:.2e}/s)"

# --- Solve N ---
denom = (1.0 / Nf) + dc_per_cycle
N = 1.0 / denom

Df = N / Nf
Dc = N * dc_per_cycle

# --- Display ---
st.subheader("Prediction")
st.markdown(f"""
- Temperature: **{T_K:.1f} K**
- Fatigue: **{fatigue_desc}**  
- Creep: **{creep_desc}**  
- Predicted life: **{N:,.0f} cycles**  
- Damage breakdown: Df={Df:.3f}, Dc={Dc:.3f}
""")

# --- Bar plot ---
fig, ax = plt.subplots(figsize=(4,3))
ax.bar(["Fatigue","Creep"], [Df,Dc], color=["tab:blue","tab:orange"])
ax.set_ylim(0,1.1)
ax.set_ylabel("Damage fraction")
ax.set_title(f"Total Damage = {Df+Dc:.2f}")
st.pyplot(fig)

# --- Sweep option ---
st.subheader("Life vs dwell time sweep")
max_dwell = st.slider("Max dwell [s]", min_value=0, max_value=600, value=60, step=10)
tholds = np.linspace(0,max_dwell,25)
lifes = []
for th in tholds:
    denom = (1.0/Nf) + (th/_rupture_time_from_yaml(rupture_path,sigma_hold,T_K) if creep_mode=="rupture" else (_norton_rate(sigma_hold,T_K,nt)*th/epsilon_crit))
    lifes.append(1.0/denom)
fig2, ax2 = plt.subplots()
ax2.plot(tholds, lifes, "-o")
ax2.set_xlabel("Dwell time per cycle [s]")
ax2.set_ylabel("Life N [cycles]")
ax2.set_title("Life vs dwell time")
st.pyplot(fig2)
