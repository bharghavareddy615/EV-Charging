# Kavali EV Charging Demand Forecasting

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep these files in the same folder as `app.py`:
- kavali_ev_hourly_usage_synthetic.csv
- kavali_ev_sessions_synthetic_Oct2025-Sep2026.csv
- kavali_ev_stations.csv

The app can also accept the CSVs through the sidebar.

## Event CSV

Columns:
- date
- event_name
- event_type
- impact_pct
- start_hour
- end_hour

`impact_pct` is an explicit scenario assumption, not an automatically verified causal effect.

## Forecast behavior

The default forecast date is the computer's current date. Therefore opening the dashboard on a different day changes the forecast target automatically. Historical lags are used recursively for future hours.

## Important limitation

The supplied Kavali files are synthetic. Use them as a prototype/demo dataset and label them accordingly in reports.
