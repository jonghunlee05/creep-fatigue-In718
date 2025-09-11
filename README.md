# Inconel 718 Creep-Fatigue Life Prediction

A comprehensive Python framework for predicting creep-fatigue life in Inconel 718 superalloy using time-fraction damage accumulation methods. This project provides both interactive web applications and command-line tools for material scientists and engineers working with high-temperature alloys.

## Features

- **Interactive Web App**: Streamlit-based interface for real-time life predictions
- **Multiple Fatigue Models**: 
  - Low-Cycle Fatigue (LCF) using Coffin-Manson equation
  - High-Cycle Fatigue (HCF) using Basquin equation
- **Creep Damage Models**:
  - Rupture-based damage using Larson-Miller Parameter
  - Rate-based damage using Norton creep law
- **Time-Fraction Damage Accumulation**: Combines fatigue and creep damage
- **Data Processing Pipeline**: Automated conversion from raw NASA data to SI units
- **Model Calibration**: Scripts for fitting model parameters from experimental data
- **Visualization**: Life vs dwell time sweeps and damage breakdown charts

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Theory](#models-and-theory)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Scripts Reference](#scripts-reference)
- [Contributing](#contributing)

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Interactive App**:
   ```bash
   make run-app
   # or directly:
   streamlit run app/LifeApp.py
   ```

3. **Open Browser**: Navigate to `http://localhost:8501`

4. **Try a Prediction**:
   - Set temperature: 977 K (704°C)
   - Choose LCF model with strain amplitude: 0.004
   - Set hold stress: 550 MPa, hold time: 5 seconds
   - View predicted life and damage breakdown

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Dependencies

The project requires the following Python packages (see `requirements.txt`):

```
streamlit>=1.28.0
matplotlib>=3.9.0
numpy>=1.26.0
pandas>=2.3.0
scipy>=1.13.0
pyyaml>=6.0
```

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd creep-fatigue-In718
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   make test
   ```

## Usage

### Interactive Web Application

The main interface is a Streamlit web app that provides:

- **Input Parameters**:
  - Temperature: 300-1400 K
  - Fatigue model selection (LCF/HCF)
  - Strain amplitude (LCF): 0.001-0.02
  - Stress amplitude (HCF): 100-1200 MPa
  - Hold stress: 100-1200 MPa
  - Hold time: 0-3600 seconds
  - Creep damage mode (rupture/rate)

- **Outputs**:
  - Predicted life in cycles
  - Damage breakdown (fatigue vs creep)
  - Interactive visualizations
  - Life vs dwell time sweeps

### Command Line Tools

#### Model Calibration

```bash
# Calibrate Norton creep model
make calibrate-creep

# Calibrate HCF Basquin model
make calibrate-hcf

# Calibrate LCF Coffin-Manson model
make calibrate-lcf

# Fit rupture models
make rupture-fit
```

#### Life Prediction

```bash
# Single prediction
make predict-timefrac

# Parameter sweeps
make sweeps
```

#### Individual Scripts

```bash
# Query specific models
python scripts/query_lcf_from_yaml.py
python scripts/query_hcf_life.py
python scripts/query_rupture_life.py

# Generate rupture maps
python scripts/make_rupture_map.py
```

## Models and Theory

### Time-Fraction Damage Accumulation

The core methodology uses the time-fraction rule:

```
D(N) = N/Nf + N·dc = 1
```

Where:
- `N` = total cycles to failure
- `Nf` = pure fatigue life
- `dc` = creep damage per cycle

### Fatigue Models

#### Low-Cycle Fatigue (Coffin-Manson)
```
Δε/2 = (σf'/E)·(2N)^b + εf'·(2N)^c
```

#### High-Cycle Fatigue (Basquin)
```
σa = σf'·(2N)^b
```

### Creep Models

#### Norton Creep (Rate-based)
```
ε̇ = A·σ^n·exp(-Q/RT)
```

#### Larson-Miller Parameter (Rupture-based)
```
LMP = T·(log(t) + C)
```

### Model Parameters

All calibrated parameters are stored in YAML files under `models/calibrations/`:

- **Fatigue**: `in718_lcf_coffin_manson.yaml`, `in718_hcf_basquin.yaml`
- **Creep**: `in718_norton.yaml`
- **Rupture**: `in718_rupture_best.yaml`

## Data Sources

### Primary Data Source
- **NASA Technical Report 19960008692**: "Probabilistic Material Strength Degradation Model for Inconel 718"
- **Special Metals INCONEL® 718 Technical Bulletin** (cross-reference)

### Datasets

The project includes processed datasets in SI units:

- **Fatigue Data**:
  - `in718_fatigue_HCF_isothermal_SI.csv`: High-cycle fatigue (stress control)
  - `in718_fatigue_LCF_isothermal_537C_SI.csv`: Low-cycle fatigue (strain control)
  - `in718_fatigue_TMF_316C_649C_SI.csv`: Thermal-mechanical fatigue

- **Creep Data**:
  - `in718_creep_rupture_isothermal_SI.csv`: Rupture times
  - `in718_creep_steady_state_SI.csv`: Steady-state creep rates

### Data Processing

Raw data is automatically converted from imperial to SI units:
- Temperature: °F → °C
- Stress: psi → MPa  
- Strain: % → unitless

## Project Structure

```
creep-fatigue-In718/
├── app/                          # Web application
│   ├── LifeApp.py               # Main Streamlit app
│   └── README.md                # App-specific documentation
├── data/                        # Data storage
│   ├── raw/                     # Original NASA data
│   ├── processed/               # Converted SI data
│   └── registry.yaml            # Data catalog
├── models/                      # Model definitions
│   ├── calibrations/            # Fitted parameters
│   │   ├── creep/
│   │   ├── fatigue/
│   │   └── rupture/
│   └── releases/                # Versioned models
├── reports/                     # Analysis outputs
│   ├── figures/                 # Generated plots
│   └── calibration/             # Fitting reports
├── scripts/                     # Command-line tools
│   ├── fit_*.py                 # Model calibration
│   ├── query_*.py               # Life prediction
│   └── make_*.py                # Visualization
├── tests/                       # Unit tests
├── Makefile                     # Build automation
├── params.yaml                  # Project configuration
└── requirements.txt             # Python dependencies
```

## Scripts Reference

### Calibration Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `fit_norton_from_csv.py` | Fit Norton creep parameters | Creep rate data | `in718_norton.yaml` |
| `fit_basquin_from_csv.py` | Fit Basquin HCF parameters | S-N data | `in718_hcf_basquin.yaml` |
| `fit_coffin_manson_from_csv.py` | Fit Coffin-Manson LCF parameters | ε-N data | `in718_lcf_coffin_manson.yaml` |
| `fit_larson_miller_from_csv.py` | Fit rupture parameters | Rupture data | `in718_rupture_*.yaml` |

### Prediction Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `predict_time_fraction.py` | Single life prediction | `--T_K 977 --sigma_MPa 550 --eps_total 0.004` |
| `sweep_time_fraction.py` | Parameter sweeps | `--T_K 977 --thold_max 30` |
| `query_*_life.py` | Query specific models | Various parameters |

### Visualization Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `make_rupture_map.py` | Generate rupture maps | `in718_rupture_map_best.png` |

## Testing

Run the test suite:

```bash
make test
# or
python -m pytest -q
```

Tests cover:
- Parameter range validation
- Time-fraction calculations
- Model loading and prediction

## Example Results

### Typical Predictions

For Inconel 718 at 977 K (704°C):
- **LCF**: Δε/2 = 0.004, σ_hold = 550 MPa, t_hold = 5s
  - Predicted life: ~1,000-10,000 cycles
  - Damage breakdown: ~60% fatigue, 40% creep

- **HCF**: σ_a = 600 MPa, σ_hold = 550 MPa, t_hold = 5s
  - Predicted life: ~10,000-100,000 cycles
  - Damage breakdown: ~80% fatigue, 20% creep

### Model Performance

- **Norton Creep**: R² = 0.92 (23 data points)
- **Coffin-Manson**: R² = 0.99 (4 data points)
- **Basquin**: R² = 0.95 (estimated)

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `make test`
5. Commit changes: `git commit -m "Add feature"`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

## License

This project uses data from NASA Technical Reports (likely public domain) and Special Metals documentation. Please verify licensing terms for commercial use.

## References

1. NASA Technical Report 19960008692: "Probabilistic Material Strength Degradation Model for Inconel 718"
2. Special Metals INCONEL® 718 Technical Bulletin
3. Larson, F.R. and Miller, J. (1952). "A Time-Temperature Relationship for Rupture and Creep Stresses"
4. Coffin, L.F. (1954). "A Study of the Effects of Cyclic Thermal Stresses on a Ductile Metal"
5. Basquin, O.H. (1910). "The Exponential Law of Endurance Tests"

## Support

For questions or issues:
1. Check the [Issues](https://github.com/your-repo/issues) page
2. Review the model calibration reports in `reports/calibration/`
3. Examine the example data in `data/processed/`

---

**Note**: This project is designed for research and educational purposes. Always validate predictions against experimental data for critical applications.
