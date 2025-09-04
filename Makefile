.PHONY: calibrate-creep calibrate-hcf calibrate-lcf rupture-fit predict-timefrac sweeps test

calibrate-creep: ; python3 scripts/fit_norton_from_csv.py
calibrate-hcf:   ; python3 scripts/fit_basquin_from_csv.py
calibrate-lcf:   ; python3 scripts/fit_coffin_manson_from_csv.py
rupture-fit:     ; python3 scripts/fit_larson_miller_from_csv.py

predict-timefrac:
	python3 scripts/predict_time_fraction.py --T_K 977 --sigma_MPa 550 --eps_total 0.004 --t_hold_s 5

sweeps:
	python3 scripts/sweep_time_fraction.py --T_K 977 --sigma_MPa 550 --eps_total 0.004 --thold_max 30

test:
	python3 -m pytest -q
