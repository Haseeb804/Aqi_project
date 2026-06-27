# 🌫️ AQI Prediction System – Faisalabad, Pakistan

End-to-end **serverless ML pipeline** for real-time **3-day Air Quality Index (AQI) forecasting**.  
Built with Python · Scikit-learn · TensorFlow · Hopsworks · Airflow · GitHub Actions · Streamlit · FastAPI · SHAP.

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                    │
│   AQICN API (real-time)          OpenWeather API (historical + live)     │
└────────────────┬─────────────────────────────────┬───────────────────────┘
                 │                                 │
                 ▼                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       FEATURE PIPELINE  (hourly)                         │
│  fetch_current_with_fallback → engineer_features → FeatureStore.upsert   │
│  • Time features: hour, day_of_week, month, season                       │
│  • Rolling averages: 3h / 6h / 12h  for all pollutants + AQI            │
│  • Derived: AQI change rate (momentum)                                   │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    FEATURE STORE  (Hopsworks / Local Parquet)            │
│                         data/aqi_features.parquet                        │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       TRAINING PIPELINE  (daily)                         │
│   ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐    │
│   │ Random Forest│  │    Ridge     │  │   LSTM (TensorFlow / Keras) │    │
│   │ 200 trees    │  │  Regression  │  │   128→64 units, MC Dropout  │    │
│   │ MultiOutput  │  │  MultiOutput │  │   24-step lookback sequence │    │
│   └──────────────┘  └──────────────┘  └────────────────────────────┘    │
│                  ↓ Best model selected by RMSE                           │
│              SHAP / Feature Importance  →  Model Registry                │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  WEB APPLICATION  (Streamlit + FastAPI)                  │
│  • Real-time 3-day AQI forecast  • SHAP explainability charts            │
│  • AQI gauge + hazardous level alerts  • Historical trend plots          │
│  • REST API: POST /predict, GET /current/{city}, GET /health             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
aqi_project/
├── .github/
│   └── workflows/
│       └── aqi_pipeline.yml        # GitHub Actions CI/CD (hourly + daily)
├── dags/
│   ├── aqi_feature_dag.py          # Airflow – hourly feature pipeline DAG
│   └── aqi_training_dag.py         # Airflow – daily training pipeline DAG
├── src/
│   ├── config.py                   # Central configuration (env-var driven)
│   ├── feature_pipeline.py         # Live hourly AQI feature ingestion
│   ├── backfill_pipeline.py        # 2-year historical backfill (CLI)
│   ├── training_pipeline.py        # RF + Ridge + LSTM training & evaluation
│   └── eda.py                      # EDA – trend, correlation, heatmap plots
├── utils/
│   ├── api_client.py               # AQICN + OpenWeather clients with retry
│   ├── feature_engineering.py      # Time, rolling, and change-rate features
│   └── store_client.py             # Hopsworks / local Parquet feature store
├── app/
│   ├── dashboard.py                # Streamlit dashboard (forecast + EDA)
│   └── api.py                      # FastAPI /predict endpoint
├── reports/
│   ├── README.md
│   └── figures/                    # Auto-generated EDA figures (via src/eda.py)
├── models/                         # Saved model artifacts (.pkl / .keras)
├── data/                           # Local Parquet feature store cache
├── logs/                           # Pipeline log files
├── docker-compose.yml              # Local Airflow (webserver + scheduler + postgres)
├── requirements.txt                # Python dependencies
├── requirements-airflow.txt        # Airflow-specific extras
├── .env.example                    # Template – copy to .env and fill in keys
└── colab_setup.ipynb               # Google Colab end-to-end notebook
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Haseeb804/Aqi_project.git
cd Aqi_project
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your keys:
#   AQICN_KEY        – https://aqicn.org/api/
#   OPENWEATHER_KEY  – https://openweathermap.org/api/air-pollution
#   HOPSWORKS_KEY    – https://app.hopsworks.ai (optional; falls back to local Parquet)
```

### 3. Backfill 2 years of historical data

```bash
python -m src.backfill_pipeline --start 2022-01-01 --end 2024-01-01
```

### 4. Run Exploratory Data Analysis

```bash
python -m src.eda
# Figures saved to reports/figures/
```

### 5. Train models

```bash
python -m src.training_pipeline
# Best model (RF / Ridge / LSTM) saved to models/
```

### 6. Test live feature pipeline

```bash
python -m src.feature_pipeline
```

### 7. Launch Streamlit dashboard

```bash
streamlit run app/dashboard.py
```

### 8. Launch REST API

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🐳 Local Airflow (Docker Compose)

```bash
# Start Airflow webserver + scheduler + PostgreSQL
docker-compose up -d

# Access web UI at http://localhost:8080  (admin / admin)
# Two DAGs are available:
#   aqi_feature_pipeline  – runs every hour
#   aqi_training_pipeline – runs daily at midnight UTC

# Stop all services
docker-compose down
```

---

## ☁️ Automated CI/CD (GitHub Actions)

The workflow `.github/workflows/aqi_pipeline.yml` runs automatically:

| Trigger | Job | Description |
|---------|-----|-------------|
| Every hour (`0 * * * *`) | `feature-pipeline` | Fetches AQI, engineers features, upserts to store |
| Every day at midnight UTC (`0 0 * * *`) | `training-pipeline` | Re-trains models, logs to MLflow |
| Manual `workflow_dispatch` | `backfill` | Historical data backfill for a date range |
| Push to `main` | `training-pipeline` | Re-train on code changes |

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `AQICN_KEY` | AQICN API token |
| `OPENWEATHER_KEY` | OpenWeatherMap API key |
| `HOPSWORKS_KEY` | Hopsworks API key (optional) |
| `HOPSWORKS_PROJECT` | Hopsworks project name (optional) |
| `SLACK_WEBHOOK_URL` | Slack incoming webhook for alerts (optional) |

---

## 🧠 Models

| Model | Architecture | Input Type | Output |
|-------|-------------|------------|--------|
| **Random Forest** | 200 trees, MultiOutputRegressor, 5-fold CV | Current feature row | Day 1/2/3 AQI |
| **Ridge Regression** | L2 regularised, StandardScaler, MultiOutput | Current feature row | Day 1/2/3 AQI |
| **LSTM** | 128→64 LSTM units, MC Dropout 0.2, EarlyStopping | 24h sequence | Day 1/2/3 AQI |

Evaluation metrics: **RMSE**, **MAE**, **R²** on a 15% held-out temporal test set (no shuffling to prevent data leakage).

---

## 🔬 Features Engineered

| Category | Features |
|----------|---------|
| **Raw pollutants** | PM2.5, PM10, NO₂, O₃, CO, SO₂ |
| **Time** | hour, day_of_week, month, season |
| **Rolling 3h** | `{pollutant}_3h_avg` for all 6 pollutants + AQI |
| **Rolling 6h** | `{pollutant}_6h_avg` for all 6 pollutants + AQI |
| **Rolling 12h** | `{pollutant}_12h_avg` for all 6 pollutants + AQI |
| **Derived** | `aqi_change_rate` (percentage change from previous hour) |

---

## 📊 EDA Outputs

Run `python -m src.eda` to generate all figures in `reports/figures/`:

| Figure | Content |
|--------|---------|
| `01_aqi_distribution.png` | AQI readings by US EPA category (bar + pie) |
| `02_seasonal_trends.png` | Monthly boxplot & seasonal violin |
| `03_time_heatmap.png` | Mean AQI by hour × day-of-week |
| `04_pollutant_correlation.png` | Pearson correlation matrix |
| `05_rolling_averages.png` | Raw vs 3h/6h/12h smoothed AQI |
| `06_aqi_change_rate.png` | AQI momentum feature distribution |
| `07_pollutant_timeseries.png` | Multi-panel pollutant concentrations |
| `08_aqi_vs_pm25.png` | AQI vs PM2.5 scatter by season |
| `09_annual_aqi_trend.png` | Annual AQI trend with confidence band |

---

## 🌐 API Usage

```bash
# Health check
curl http://localhost:8000/health

# 3-day AQI forecast
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"city": "Faisalabad", "date": "2024-06-01"}'

# Current AQI + forecast (shorthand)
curl http://localhost:8000/current/Faisalabad
```

**Example response:**
```json
{
  "city": "Faisalabad",
  "current_aqi": 187.0,
  "current_category": "Unhealthy",
  "forecast": [
    {"day": 1, "date": "2024-06-02", "predicted_aqi": 179.3, "category": "Unhealthy"},
    {"day": 2, "date": "2024-06-03", "predicted_aqi": 162.1, "category": "Unhealthy"},
    {"day": 3, "date": "2024-06-04", "predicted_aqi": 148.7, "category": "Unhealthy for Sensitive Groups"}
  ],
  "model_name": "RandomForest",
  "model_rmse": 12.4321
}
```

---

## 🚨 AQI Scale Reference

| Range | Category | Health Guidance |
|-------|----------|----------------|
| 0–50 | 🟢 Good | Air quality is satisfactory |
| 51–100 | 🟡 Moderate | Acceptable; some pollutants may affect sensitive groups |
| 101–150 | 🟠 Unhealthy for Sensitive Groups | Reduce prolonged outdoor exertion |
| 151–200 | 🔴 Unhealthy | Avoid prolonged outdoor exertion |
| 201–300 | 🟣 Very Unhealthy | Avoid outdoor activity |
| 301–500 | 🟤 Hazardous | Stay indoors; wear N95 if going outside |

---

## 🛠️ Development

```bash
# Run EDA
python -m src.eda

# Test feature pipeline (single run)
python -m src.feature_pipeline

# Backfill a custom date range
python -m src.backfill_pipeline --start 2023-01-01 --end 2023-06-01 --output data/test.parquet

# Train models
python -m src.training_pipeline

# Run all tests (if any)
pytest tests/
```

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.
