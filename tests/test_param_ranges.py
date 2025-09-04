import yaml

def _load(p):
    with open(p,"r") as f: return yaml.safe_load(f)

def test_norton_ranges():
    y = _load("models/calibrations/creep/in718_norton.yaml")
    p = y["params"]
    assert 3.0 <= p["n"] <= 15.0  # Extended range for Inconel 718
    assert 3e5 <= p["Q_J_per_mol"] <= 7e5  # 300–700 kJ/mol

def test_basquin_signs():
    y = _load("models/calibrations/fatigue/in718_hcf_basquin.yaml")
    p = y["params"]
    assert p["k"] > 0  # line slopes down in log-log

def test_coffin_manson_signs():
    y = _load("models/calibrations/fatigue/in718_lcf_coffin_manson.yaml")
    p = y["params"]
    assert p["b"] < 0
    assert p["c"] < 0
    # rough magnitudes (not too restrictive)
    assert -4.0 < p["log10_epsf"] < 0.0
    assert -4.0 < p["log10_sigmaf_over_E"] < 0.0
