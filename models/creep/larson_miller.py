"""
larson_miller.py — Rupture life model for Inconel 718 (Larson–Miller Parameter, LMP)

Purpose
-------
Fit and use the classic LMP correlation for creep-rupture data:
    LMP = T * (C + log10(t_r))

We assume a global linear relation between stress and LMP:
    log10(sigma) = alpha + beta * (LMP_scaled)
where LMP_scaled = 1e-3 * LMP  (numerical scaling to keep values ~O(10))

From this, given (sigma, T) we can predict rupture life t_r, and vice versa.

Units
-----
T in Kelvin, sigma in MPa, t_r in hours (typical for LMP charts).
"""

import numpy as np
from scipy.optimize import curve_fit

SCALE = 1e-3  # scale LMP = T*(C+log10 t_r) to avoid huge numbers

def lmp(T_K: np.ndarray, t_r_h: np.ndarray, C: float) -> np.ndarray:
    return T_K * (C + np.log10(t_r_h))

def _model_for_curve_fit(X, alpha, beta, C):
    T, t_r_h = X
    LMPs = SCALE * lmp(T, t_r_h, C)
    return alpha + beta * LMPs  # predicts log10(sigma)

def fit_lmp(stress_MPa, T_K, t_r_h, p0=(2.0, 1e-2, 20.0)):
    """
    Fit alpha, beta, C to minimize the error in log10(sigma).

    Returns
    -------
    popt = (alpha, beta, C), pcov
    """
    y = np.log10(np.asarray(stress_MPa, float))
    T = np.asarray(T_K, float); tr = np.asarray(t_r_h, float)
    popt, pcov = curve_fit(_model_for_curve_fit, (T, tr), y, p0=p0, maxfev=50000)
    return popt, pcov

def predict_log10_sigma(T_K, t_r_h, alpha, beta, C):
    return _model_for_curve_fit((np.asarray(T_K,float), np.asarray(t_r_h,float)), alpha, beta, C)

def predict_sigma(T_K, t_r_h, alpha, beta, C):
    return 10.0 ** predict_log10_sigma(T_K, t_r_h, alpha, beta, C)

def predict_time_to_rupture_h(stress_MPa, T_K, alpha, beta, C):
    """
    Invert the regression to get rupture time for given (sigma, T):
        log10(sigma) = alpha + beta * SCALE * T*(C + log10 t_r)
    =>  log10 t_r = (log10(sigma) - alpha) / (beta * SCALE * T) - C
    """
    log10_sigma = np.log10(np.asarray(stress_MPa, float))
    T = np.asarray(T_K, float)
    log10_tr = (log10_sigma - alpha) / (beta * SCALE * T) - C
    return 10.0 ** log10_tr
