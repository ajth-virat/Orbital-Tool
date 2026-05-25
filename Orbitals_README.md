# Orbital GNC Design Tool

---

A browser-based spacecraft orbital analysis tool built with Streamlit. You plug in your orbital parameters — either Keplerian elements or Cartesian state vectors — and it runs the full chain: orbit propagation, delta-v budgets, ground coverage, atmospheric lifetime, radiation environment, and ADCS timeline, all displayed in interactive Plotly tabs. No simulation software needed. Everything runs locally in Python and opens in your browser.

---

## Quickstart

**1. Clone and install**
```bash
git clone https://github.com/your-username/Orbitals.git
cd Orbitals
pip install -r requirements.txt
```

**2. Run**
```bash
streamlit run app.py
```

Your browser opens automatically at `http://localhost:8501`.

---

## How to use it

1. Fill in orbital parameters in the left sidebar (altitude, inclination, eccentricity, etc.)
2. Choose input mode: Keplerian elements or Cartesian state vectors
3. Press **Run Analysis**
4. Results appear across nine tabs — each module has its own panel

---

## What it computes

| Tab | What it shows |
|---|---|
| Orbit | 3D interactive orbit and ground track |
| Altitude | Altitude variation over time |
| Keplerian | Orbital element evolution |
| Delta-V | Hohmann, bi-elliptic, phasing, and combined manoeuvre budgets |
| Coverage | Ground access windows, eclipse periods |
| Lifetime | Atmospheric drag lifetime under different solar activity scenarios |
| Environment | Radiation and space environment conditions |
| ADCS | Attitude control timeline and slew calculations |
| Data | Raw numerical outputs |

---



## Project structure

```
├── app.py               # Entry point — Streamlit UI and orchestrator
├── requirements.txt
│
├── constants.py         # Physical and mission constants
├── validator.py         # Input validation (Keplerian and Cartesian)
├── converter.py         # Keplerian ↔ Cartesian, period, velocity
├── propagator.py        # Orbit propagation
├── deltav.py            # Transfer manoeuvre calculations
├── perturbations.py     # Atmospheric density and perturbation models
├── coverage.py          # Ground track, access windows, eclipse
├── lifetime.py          # Drag lifetime under solar scenarios
├── environment.py       # Radiation and space environment
└── adcs_timeline.py     # Attitude control and slew calculations
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `streamlit` | Web UI |
| `plotly` | Interactive 3D plots and charts |
| `numpy` | Numerical computation |
| `scipy` | Scientific calculations |
| `pandas` | Data tables |
| `astropy` | Astronomical constants and coordinate transforms |

---

## Author

**Ajith Virat Sridhara Narasimhan**  
PG Certificate in Space Engineering — University of Surrey, UK  
BTech Aerospace Engineering — Bharath University (BIHER), India
