# Inconel 718 Creep-Fatigue Life Predictor

Interactive web application for predicting creep-fatigue life in Inconel 718 using time-fraction damage accumulation.

## Features

- **Interactive Inputs**: Temperature, strain/stress amplitudes, hold times
- **Dual Fatigue Models**: LCF (Coffin-Manson) and HCF (Basquin)
- **Creep Damage Modes**: Rupture-based and rate-based
- **Real-time Visualization**: Damage breakdown and life vs dwell time sweeps
- **Parameter Validation**: Input ranges based on material properties

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   make run-app
   # or directly:
   streamlit run app/LifeApp.py
   ```

3. **Open Browser**: Navigate to `http://localhost:8501`

## Usage

### Input Parameters

**Fatigue Model Selection**:
- **LCF (Coffin-Manson)**: For low-cycle fatigue with strain control
- **HCF (Basquin)**: For high-cycle fatigue with stress control

**Temperature Range**: 300-1400 K (typical for Inconel 718)

**Strain/Stress Ranges**:
- Strain amplitude: 0.001-0.02 (0.1%-2%)
- Stress amplitude: 100-1200 MPa

**Creep Parameters**:
- Hold stress: 100-1200 MPa
- Hold time: 0-3600 seconds
- Critical strain (rate mode): 0.005-0.1 (0.5%-10%)

### Outputs

- **Predicted Life**: Total cycles to failure
- **Damage Breakdown**: Fatigue (Df) and creep (Dc) contributions
- **Visualization**: Bar chart of damage fractions
- **Sweep Analysis**: Life vs dwell time relationship

## Models Used

- **Norton Creep**: Rate-based creep damage
- **Larson-Miller Parameter**: Rupture-based creep damage
- **Coffin-Manson**: LCF fatigue life prediction
- **Basquin**: HCF fatigue life prediction

## Technical Details

The app uses the time-fraction damage accumulation rule:
```
D(N) = N/Nf + N·dc = 1
```

Where:
- N = total cycles to failure
- Nf = pure fatigue life
- dc = creep damage per cycle

## File Structure

```
app/
├── LifeApp.py          # Main Streamlit application
├── README.md           # This file
└── requirements.txt    # Python dependencies
```
