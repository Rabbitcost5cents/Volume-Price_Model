# Volume-Price Sales Forecasting Model

Mobile phone sales forecasting system combining the **Bass Diffusion Model** (lifecycle baseline) with **XGBoost** (daily sales fluctuation prediction). The system ingests hardware spec data and daily sales history, trains a regression model, and generates day-level or month-level sales forecasts based on configurable price, specs, and launch date.

Two interfaces are provided: a **Tkinter desktop GUI** for offline use and rapid iteration, and a **Streamlit web app** for richer visualization and team-accessible simulation history.

---

## Features

- **Sales Simulator** — forecast daily or monthly sales for new or existing products by adjusting price, hardware specs, and Bass model parameters
- **Dual Interface** — Tkinter desktop GUI (offline) and Streamlit web app (browser-based), both fully functional
- **Two-Tier Auth** — app access password (all users) and admin password (data management), stored as PBKDF2-HMAC-SHA256 hashes
- **Data Import** — upload Excel or CSV sales/spec files via UI; automatic schema validation and data quality filtering
- **Model Training** — retrain XGBoost models from the admin panel with live log output; previous models auto-backed up
- **Preset Management** — save, load, and delete named simulation configurations
- **Simulation History** — persistent record of all simulation runs, queryable from the Streamlit History page
- **i18n** — English / 中文 switchable in both interfaces

---

## Tech Stack

| Layer | Technology |
|---|---|
| Prediction | XGBoost, Bass Diffusion Model, scipy curve_fit |
| Desktop UI | Tkinter |
| Web UI | Streamlit, Plotly |
| Storage | SQLite (WAL mode) |
| Data | pandas, openpyxl |

---

## Project Structure

```
Volume-Price_Model/
├── config.json              ← Global parameters (elasticity, Bass constraints, holidays)
├── requirements.txt
├── data/                    ← SQLite database [gitignored — not committed]
├── models/                  ← Trained models & config [gitignored — not committed]
└── src/
    ├── gui_app.py           ← Desktop GUI entry point
    ├── streamlit_app.py     ← Streamlit web app entry point
    ├── app.py               ← Core simulation engine (SalesSimulator)
    ├── auth.py              ← Shared password hashing (PBKDF2-HMAC-SHA256)
    ├── bass_engine.py       ← Bass diffusion model fitting and prediction
    ├── data_processor_v2.py ← Data cleaning, feature engineering, ingestion
    ├── db.py                ← SQLite utility module
    ├── train_daily.py       ← XGBoost training script
    ├── rate_limiter.py      ← Simulation run-rate limiter
    ├── config_loader.py     ← config.json loader with deep-merge defaults
    ├── i18n.py              ← Internationalization strings
    ├── migrate_excel.py     ← One-time Excel → SQLite migration script
    ├── user_manual.md       ← Full bilingual user manual (中文 / English)
    └── tests/               ← pytest test suite (51 tests)
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Import data

Place your specs and sales files, then import via the admin panel (recommended) or command line:

```bash
python src/data_processor_v2.py path/to/specs.xlsx path/to/sales.xlsx
```

### 3. Train the model

```bash
python src/train_daily.py
```

Or use the **Retrain Model** button in the admin panel of either interface.

### 4. Launch

**Desktop GUI:**
```bash
python src/gui_app.py
```

**Streamlit web app:**
```bash
streamlit run src/streamlit_app.py
```

> On first launch, you will be prompted to set an app access password and an admin password.

---

## Data Format

### Specs file (Excel)
- Sheet name: `512GB`, `512`, `Specs`, or `Sheet1`
- Layout: horizontal (each column = one product, rows = spec fields)
- Required column: `PRODUCT MODEL`

### Sales file (Excel)
- Each product block starts with a header row where column 0 = `Model`
- Columns 2+ are date columns (`YYYY-MM-DD`), values are daily sales

### Sales file (CSV)
- Column 0 name must contain `model`
- Remaining columns are dates (`YYYY-MM-DD`), values are daily sales

---

## User Manual

See [`src/user_manual.md`](src/user_manual.md) for the full bilingual (中文 / English) operation guide covering all features, configuration reference, and troubleshooting.
