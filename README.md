# AQI Prediction System – Faisalabad, Pakistan

End-to-end ML pipeline for real-time 3-day Air Quality Index (AQI) forecasting.

## Project Structure

```
aqi_project/
├── .github/workflows/aqi_pipeline.yml   # GitHub Actions CI/CD
├── dags/aqi_feature_dag.py             # Apache Airflow DAG (hourly)
├── src/
│   ├── config.py                        # Central configuration
│   ├── feature_pipeline.py              # Live hourly feature ingestion
│   ├── backfill_pipeline.py             # 2-year historical backfill (CLI)
│   └── training_pipeline.py            # RF + Ridge + LSTM training
├── utils/
│   ├── api_client.py                    # AQICN + OpenWeather API with retry
│   ├── feature_engineering.py          # Time, rolling, change-rate features
│   └── store_client.py                 # Hopsworks / local Parquet store
├── app/
│   ├── dashboard.py                     # Streamlit dashboard
│   └── api.py                           # FastAPI /predict endpoint
├── models/                              # Saved model artifacts
├── data/                                # Local Parquet feature store
├── logs/                                # Pipeline log files
├── .env                                 # Your API keys (do not commit)
├── .env.example                         # Template for .env
└── requirements.txt
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Backfill 2 years of historical data
```bash
python -m src.backfill_pipeline --start 2022-01-01 --end 2024-01-01
```

### 3. Train models
```bash
python -m src.training_pipeline
```

### 4. Run live feature pipeline once (test)
```bash
python -m src.feature_pipeline
```

### 5. Launch dashboard
```bash
streamlit run app/dashboard.py
```

### 6. Launch prediction API
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Get 3-day forecast
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"city": "Faisalabad", "date": "2024-06-01"}'

# Current AQI + forecast
curl http://localhost:8000/current/Faisalabad
```

## GitHub Actions Secrets Required

| Secret              | Description                          |
|---------------------|--------------------------------------|
| `AQICN_KEY`         | AQICN API token                      |
| `OPENWEATHER_KEY`   | OpenWeatherMap API key               |
| `HOPSWORKS_KEY`     | Hopsworks API key                    |
| `HOPSWORKS_PROJECT` | Hopsworks project name               |
| `SLACK_WEBHOOK_URL` | Slack incoming webhook (for alerts)  |

## Features Engineered

- **Raw**: PM2.5, PM10, NO2, O3, CO, SO2, AQI
- **Time**: hour, day_of_week, month, season
- **Rolling**: 3h/6h/12h averages for all pollutants + AQI
- **Derived**: AQI_change_rate (momentum)

## Models

| Model         | Architecture                          | Input          |
|---------------|---------------------------------------|----------------|
| Random Forest | 200 trees, MultiOutput                | Current row    |
| Ridge Regression | L2 regularised, MultiOutput        | Current row    |
| LSTM          | 128→64 units, MC Dropout              | 24h sequence   |

Evaluation metrics: RMSE, MAE, R² on 15% held-out temporal test set.
