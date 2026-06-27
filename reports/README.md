# AQI Project – Reports

This directory contains auto-generated EDA figures and analysis reports.

## Generating Figures

Run the EDA pipeline to regenerate all figures:

```bash
python -m src.eda
```

Figures are saved to `reports/figures/`:

| File | Description |
|------|-------------|
| `01_aqi_distribution.png` | AQI count by US EPA category (bar + pie) |
| `02_seasonal_trends.png` | Monthly boxplot & seasonal violin plot |
| `03_time_heatmap.png` | Mean AQI heat-map by hour and day-of-week |
| `04_pollutant_correlation.png` | Pearson correlation matrix for all pollutants |
| `05_rolling_averages.png` | Raw vs 3h/6h/12h rolling AQI averages |
| `06_aqi_change_rate.png` | AQI momentum feature distribution |
| `07_pollutant_timeseries.png` | Multi-panel pollutant concentration time-series |
| `08_aqi_vs_pm25.png` | AQI vs PM2.5 scatter coloured by season |
| `09_annual_aqi_trend.png` | Annual AQI trend with monthly mean ± std band |
