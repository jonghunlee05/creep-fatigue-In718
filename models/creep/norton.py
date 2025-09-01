"""
norton.py — Norton (power-law) creep model for Inconel 718
==========================================================

Purpose:
-----------
Implements Norton’s Law, also known as the power-law creep model.
This is the simplest mechanistic description of steady-state creep.

Equation:
    ε_dot = A * σ^n * exp(-Q / (R * T))

Where:
    ε_dot : creep strain rate [1/s]
    σ     : applied stress [MPa]
    T     : absolute temperature [K]
    A     : material constant (pre-exponential factor)
    n     : stress exponent (dimensionless)
    Q     : activation energy [J/mol]
    R     : universal gas constant = 8.314 J/(mol·K)

Usage:
------
- Import into analysis scripts to predict creep rates.
- Calibrate constants (A, n, Q) against processed Inconel 718 data.
- Serves as the mechanistic backbone for creep-fatigue interaction models.
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Universal gas constant [J/(mol·K)]
R = 8.314


def norton_law(stress: float, T: float, A: float, n: float, Q: float) -> float:
    """
    Norton’s Law: compute steady-state creep strain rate.

    Parameters
    ----------
    stress : float
        Applied stress [MPa]
    T : float
        Absolute temperature [K]
    A : float
        Pre-exponential constant
    n : float
        Stress exponent
    Q : float
        Activation energy [J/mol]

    Returns
    -------
    eps_dot : float
        Creep strain rate [1/s]
    """
    return A * (stress**n) * np.exp(-Q / (R * T))


def fit_norton(stress_data, T_data, creep_rate_data, p0=None):
    """
    Fit Norton’s Law parameters to experimental data.

    Parameters
    ----------
    stress_data : array-like
        Applied stresses [MPa]
    T_data : array-like
        Temperatures [K]
    creep_rate_data : array-like
        Measured creep strain rates [1/s]
    p0 : tuple, optional
        Initial guess for (A, n, Q).
        Example: (1e-5, 5, 3.2e5)

    Returns
    -------
    popt : ndarray
        Optimal values for (A, n, Q)
    pcov : 2D ndarray
        Covariance of fit
    """
    def model(X, A, n, Q):
        stress, T = X
        return norton_law(stress, T, A, n, Q)

    popt, pcov = curve_fit(model, (stress_data, T_data), creep_rate_data, p0=p0, maxfev=10000)
    return popt, pcov


if __name__ == "__main__":
    # ----------------------------------------------------------
    # 🔬 Demo run with synthetic data
    # Replace this with processed Inconel 718 dataset later.
    # ----------------------------------------------------------
    np.random.seed(0)

    # True parameters (for synthetic test)
    A_true, n_true, Q_true = 1e-5, 5.0, 3.0e5

    # Generate synthetic data
    stresses = np.linspace(100, 400, 20)       # MPa
    T = 973.0                                  # K (~700°C)
    creep_rates = norton_law(stresses, T, A_true, n_true, Q_true)
    creep_rates_noisy = creep_rates * (1 + 0.1*np.random.randn(len(stresses)))

    # Fit model
    popt, _ = fit_norton(stresses, np.full_like(stresses, T), creep_rates_noisy,
                         p0=(1e-6, 4.0, 2.5e5))

    A_fit, n_fit, Q_fit = popt
    print("Fitted parameters:")
    print(f"A = {A_fit:.3e}, n = {n_fit:.2f}, Q = {Q_fit:.2e} J/mol")

    # Plot results
    plt.scatter(stresses, creep_rates_noisy, label="Synthetic data", color="r")
    plt.plot(stresses, norton_law(stresses, T, *popt), label="Fitted Norton law", color="b")
    plt.yscale("log")
    plt.xlabel("Stress [MPa]")
    plt.ylabel("Creep rate [1/s] (log scale)")
    plt.title("Norton’s Law Fit (Synthetic Test)")
    plt.legend()
    plt.show()
