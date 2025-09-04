import subprocess, sys

def _life(args):
    out = subprocess.check_output([sys.executable, "scripts/predict_time_fraction.py", *args], text=True)
    for line in out.splitlines():
        if line.startswith("Predicted life:"):
            n_str = line.split("≈", 1)[1].split("cycles")[0].strip()
            return float(n_str.replace(",", ""))
    raise RuntimeError("Life not found in output:\n" + out)

def test_zero_dwell_equals_pure_fatigue():
    n0  = _life(["--T_K","977","--eps_total","0.004","--t_hold_s","0"])
    n5  = _life(["--T_K","977","--eps_total","0.004","--sigma_MPa","550","--t_hold_s","5"])
    assert n0 > n5 and (n0 / n5) > 1.0

def test_monotonic_in_dwell_and_stress():
    base = ["--T_K","977","--eps_total","0.004","--t_hold_s","5"]
    n550 = _life(base + ["--sigma_MPa","550"])
    n600 = _life(base + ["--sigma_MPa","600"])
    assert n600 < n550

    n5  = _life(["--T_K","977","--eps_total","0.004","--sigma_MPa","550","--t_hold_s","5"])
    n10 = _life(["--T_K","977","--eps_total","0.004","--sigma_MPa","550","--t_hold_s","10"])
    assert n10 < n5

def test_rate_vs_rupture_modes_run():
    n_rup = _life(["--T_K","977","--eps_total","0.004","--sigma_MPa","550","--t_hold_s","5",
                   "--creep_damage","rupture"])
    n_rate = _life(["--T_K","977","--eps_total","0.004","--sigma_MPa","550","--t_hold_s","5",
                    "--creep_damage","rate","--epsilon_crit","0.02"])
    assert n_rate > 0 and n_rup > 0
