import os, io, math, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_OK = True
except Exception:
    TF_OK = False

st.set_page_config(
    page_title="EV Demand Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Styling -----------------------------
st.markdown("""
<style>
.main {background:#f6f8fb;}
.block-container {padding-top:1.2rem; padding-bottom:2rem;}
.hero {padding:1.4rem 1.6rem; border-radius:18px; background:linear-gradient(135deg,#071a2b,#123b58); color:white; margin-bottom:1rem; box-shadow:0 8px 30px rgba(0,0,0,.12);}
.hero h1 {font-size:2.2rem; margin:0;}
.hero p {opacity:.88; margin:.35rem 0 0;}
.card {background:white; border:1px solid #e8edf3; border-radius:16px; padding:1rem 1.1rem; box-shadow:0 4px 18px rgba(20,30,40,.05);}
.small {font-size:.84rem; color:#667085;}
.metric-title {font-size:.78rem; color:#667085; text-transform:uppercase; letter-spacing:.06em;}
.metric-value {font-size:1.55rem; font-weight:750; color:#101828;}
.badge {display:inline-block; padding:.25rem .55rem; border-radius:999px; font-size:.75rem; font-weight:700; background:#e8f7ee; color:#117a45;}
.warning {padding:.8rem 1rem; border-radius:12px; background:#fff7e6; border:1px solid #f5d48a; color:#7a5310;}
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SESSION = os.path.join(BASE, "kavali_ev_sessions_synthetic_Oct2025-Sep2026.csv")
DEFAULT_HOURLY = os.path.join(BASE, "kavali_ev_hourly_usage_synthetic.csv")
DEFAULT_STATIONS = os.path.join(BASE, "kavali_ev_stations.csv")


def read_csv(uploaded, default_path):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    return None


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def mape(y, p):
    y = np.asarray(y); p = np.asarray(p)
    mask = np.abs(y) > 1e-8
    return float(np.mean(np.abs((y[mask]-p[mask])/y[mask]))*100) if mask.any() else np.nan


def add_calendar_features(df):
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["date"]) + pd.to_timedelta(x["hour"], unit="h")
    x = x.sort_values("timestamp").reset_index(drop=True)
    x["hour_sin"] = np.sin(2*np.pi*x.timestamp.dt.hour/24)
    x["hour_cos"] = np.cos(2*np.pi*x.timestamp.dt.hour/24)
    x["dow_sin"] = np.sin(2*np.pi*x.timestamp.dt.dayofweek/7)
    x["dow_cos"] = np.cos(2*np.pi*x.timestamp.dt.dayofweek/7)
    x["month_sin"] = np.sin(2*np.pi*(x.timestamp.dt.month-1)/12)
    x["month_cos"] = np.cos(2*np.pi*(x.timestamp.dt.month-1)/12)
    x["dayofyear_sin"] = np.sin(2*np.pi*(x.timestamp.dt.dayofyear-1)/365.25)
    x["dayofyear_cos"] = np.cos(2*np.pi*(x.timestamp.dt.dayofyear-1)/365.25)
    x["is_weekend"] = (x.timestamp.dt.dayofweek >= 5).astype(int)
    x["trend_day"] = (x.timestamp.dt.normalize() - x.timestamp.dt.normalize().min()).dt.days.astype(float)
    return x


def make_hourly_station_grid(hourly, stations):
    h = hourly.copy()
    h["date"] = pd.to_datetime(h["date"])
    h["hour"] = h["hour"].astype(int)
    h = h.merge(stations[["station_id","num_chargers"]], on="station_id", how="left")
    # Keep the observed station-hour records; fill missing numerical fields.
    for c in ["sessions","energy_kwh","busy_min","num_chargers"]:
        if c in h: h[c] = pd.to_numeric(h[c], errors="coerce").fillna(0)
    h = add_calendar_features(h)
    return h


def aggregate_city(h):
    keys = ["timestamp","date","hour","dayofweek","month","dayofyear","is_weekend",
            "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
            "dayofyear_sin","dayofyear_cos","trend_day"]
    # Create date-time keys from station rows, then aggregate demand.
    g = h.groupby(["timestamp","date","hour"], as_index=False).agg(
        energy_kwh=("energy_kwh","sum"),
        sessions=("sessions","sum"),
        busy_min=("busy_min","sum"),
        active_stations=("station_id","nunique"),
    )
    g["dayofweek"] = g.timestamp.dt.dayofweek
    g["month"] = g.timestamp.dt.month
    g["dayofyear"] = g.timestamp.dt.dayofyear
    g["is_weekend"] = (g.dayofweek >= 5).astype(int)
    g["hour_sin"] = np.sin(2*np.pi*g.hour/24); g["hour_cos"] = np.cos(2*np.pi*g.hour/24)
    g["dow_sin"] = np.sin(2*np.pi*g.dayofweek/7); g["dow_cos"] = np.cos(2*np.pi*g.dayofweek/7)
    g["month_sin"] = np.sin(2*np.pi*(g.month-1)/12); g["month_cos"] = np.cos(2*np.pi*(g.month-1)/12)
    g["dayofyear_sin"] = np.sin(2*np.pi*(g.dayofyear-1)/365.25); g["dayofyear_cos"] = np.cos(2*np.pi*(g.dayofyear-1)/365.25)
    g["trend_day"] = (g.date - g.date.min()).dt.days.astype(float)
    return g.sort_values("timestamp").reset_index(drop=True)


def add_lag_features(g):
    x = g.copy().sort_values("timestamp").reset_index(drop=True)
    # City hourly sequence is intentionally reindexed to 24h so lag semantics remain meaningful.
    full_idx = pd.date_range(x.timestamp.min().floor("h"), x.timestamp.max().floor("h"), freq="h")
    x = x.set_index("timestamp").reindex(full_idx)
    x.index.name = "timestamp"
    x["date"] = x.index.date
    x["date"] = pd.to_datetime(x["date"])
    x["hour"] = x.index.hour
    x["dayofweek"] = x.index.dayofweek
    x["month"] = x.index.month
    x["dayofyear"] = x.index.dayofyear
    x["is_weekend"] = (x.dayofweek >= 5).astype(int)
    # Missing demand means no observed charging activity.
    for c in ["energy_kwh","sessions","busy_min","active_stations"]:
        x[c] = x[c].fillna(0)
    x["hour_sin"] = np.sin(2*np.pi*x.hour/24); x["hour_cos"] = np.cos(2*np.pi*x.hour/24)
    x["dow_sin"] = np.sin(2*np.pi*x.dayofweek/7); x["dow_cos"] = np.cos(2*np.pi*x.dayofweek/7)
    x["month_sin"] = np.sin(2*np.pi*(x.month-1)/12); x["month_cos"] = np.cos(2*np.pi*(x.month-1)/12)
    x["dayofyear_sin"] = np.sin(2*np.pi*(x.dayofyear-1)/365.25); x["dayofyear_cos"] = np.cos(2*np.pi*(x.dayofyear-1)/365.25)
    x["trend_day"] = (x.index.normalize() - x.index.normalize().min()).days.astype(float)
    for lag in [1,2,3,24,48,72,168,336]:
        x[f"lag_{lag}"] = x.energy_kwh.shift(lag)
    for w in [3,6,24,72,168]:
        x[f"roll_mean_{w}"] = x.energy_kwh.shift(1).rolling(w).mean()
        x[f"roll_std_{w}"] = x.energy_kwh.shift(1).rolling(w).std()
    # Growth proxy: current rolling baseline against the equivalent historical window.
    x["growth_30d"] = x.energy_kwh.shift(1).rolling(24*30, min_periods=24).mean() / (x.energy_kwh.shift(1).rolling(24*60, min_periods=24).mean().shift(24*30) + 1e-6)
    x["growth_30d"] = x.growth_30d.replace([np.inf,-np.inf],np.nan).fillna(1.0)
    return x.reset_index()


def load_events(uploaded):
    if uploaded is None:
        return pd.DataFrame(columns=["date","event_name","event_type","impact_pct","start_hour","end_hour"])
    e = pd.read_csv(uploaded)
    required = ["date","event_name","impact_pct"]
    for c in required:
        if c not in e.columns: raise ValueError(f"Events CSV needs column: {c}")
    e["date"] = pd.to_datetime(e["date"]).dt.date
    if "start_hour" not in e: e["start_hour"] = 0
    if "end_hour" not in e: e["end_hour"] = 23
    if "event_type" not in e: e["event_type"] = "General"
    e["impact_pct"] = pd.to_numeric(e["impact_pct"], errors="coerce").fillna(0)
    return e


def event_adjustment(future, events):
    out = future.copy(); out["event_impact_pct"] = 0.0; out["event_name"] = ""
    if events.empty: return out
    for i, r in out.iterrows():
        d = r.timestamp.date(); hr = r.timestamp.hour
        hit = events[(events.date == d) & (events.start_hour <= hr) & (events.end_hour >= hr)]
        if len(hit):
            out.at[i,"event_impact_pct"] = hit.impact_pct.sum()
            out.at[i,"event_name"] = ", ".join(hit.event_name.astype(str).tolist())
    return out


@st.cache_data(show_spinner=False)
def prepare_data(hourly_bytes, stations_bytes):
    h = pd.read_csv(io.BytesIO(hourly_bytes)); s = pd.read_csv(io.BytesIO(stations_bytes))
    hs = make_hourly_station_grid(h, s)
    city = aggregate_city(hs)
    city = add_lag_features(city)
    return hs, city, s


def xgb_train(train, features):
    X = train[features].replace([np.inf,-np.inf],np.nan).fillna(0)
    y = train["energy_kwh"].astype(float)
    model = XGBRegressor(
        n_estimators=700, max_depth=7, learning_rate=0.035,
        subsample=.85, colsample_bytree=.85, min_child_weight=3,
        reg_alpha=.05, reg_lambda=1.2, objective="reg:squarederror",
        random_state=42, n_jobs=-1
    )
    model.fit(X,y,verbose=False)
    return model


def make_lstm_sequences(df, feature_cols, target_col, lookback):
    arr = df[feature_cols].replace([np.inf,-np.inf],np.nan).fillna(0).values.astype("float32")
    target = df[target_col].values.astype("float32")
    scaler_x = MinMaxScaler(); scaler_y = MinMaxScaler()
    xs = scaler_x.fit_transform(arr); ys = scaler_y.fit_transform(target.reshape(-1,1)).ravel()
    X=[]; Y=[]
    for i in range(lookback, len(df)):
        X.append(xs[i-lookback:i]); Y.append(ys[i])
    return np.array(X), np.array(Y), scaler_x, scaler_y


def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(.20),
        LSTM(32),
        Dropout(.15),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_future_rows(history, forecast_date, events):
    # Recursive feature generation for 24 future hours. Features that are known at forecast time
    # are calendar/event features; lag/rolling features come from historical or earlier forecasts.
    hist = history.copy().sort_values("timestamp").reset_index(drop=True)
    future_times = pd.date_range(pd.Timestamp(forecast_date), periods=24, freq="h")
    work = hist.copy()
    rows=[]
    for ts in future_times:
        row = {"timestamp":ts,"date":ts.normalize(),"hour":ts.hour,
               "dayofweek":ts.dayofweek,"month":ts.month,"dayofyear":ts.dayofyear,
               "is_weekend":int(ts.dayofweek>=5)}
        row["hour_sin"] = np.sin(2*np.pi*ts.hour/24); row["hour_cos"] = np.cos(2*np.pi*ts.hour/24)
        row["dow_sin"] = np.sin(2*np.pi*ts.dayofweek/7); row["dow_cos"] = np.cos(2*np.pi*ts.dayofweek/7)
        row["month_sin"] = np.sin(2*np.pi*(ts.month-1)/12); row["month_cos"] = np.cos(2*np.pi*(ts.month-1)/12)
        row["dayofyear_sin"] = np.sin(2*np.pi*(ts.dayofyear-1)/365.25); row["dayofyear_cos"] = np.cos(2*np.pi*(ts.dayofyear-1)/365.25)
        row["trend_day"] = (ts.normalize()-hist.timestamp.min().normalize()).days
        # Demand lags from work, with direct calendar fallback where possible.
        for lag in [1,2,3,24,48,72,168,336]:
            target_ts = ts-pd.Timedelta(hours=lag)
            z = work.loc[work.timestamp == target_ts,"energy_kwh"]
            row[f"lag_{lag}"] = float(z.iloc[-1]) if len(z) else float(work.energy_kwh.tail(24).mean())
        for w in [3,6,24,72,168]:
            vals = work.loc[work.timestamp < ts,"energy_kwh"].tail(w)
            row[f"roll_mean_{w}"] = float(vals.mean()) if len(vals) else float(work.energy_kwh.mean())
            row[f"roll_std_{w}"] = float(vals.std()) if len(vals)>1 else 0.0
        vals30 = work.loc[work.timestamp < ts,"energy_kwh"].tail(24*30)
        vals60 = work.loc[work.timestamp < ts,"energy_kwh"].tail(24*60)
        row["growth_30d"] = float(vals30.mean()/(vals60.mean()+1e-6)) if len(vals30) else 1.0
        row["growth_30d"] = min(max(row["growth_30d"], .6), 1.8)
        rows.append(row)
        # placeholder gets replaced by predictions by caller
        work = pd.concat([work,pd.DataFrame([{**row,"energy_kwh":np.nan}])],ignore_index=True)
    return pd.DataFrame(rows)


def recursive_xgb_forecast(model, history, forecast_date, features, events):
    work = history.copy().sort_values("timestamp").reset_index(drop=True)
    future = build_future_rows(work, forecast_date, events)
    preds=[]
    for i, r in future.iterrows():
        x = pd.DataFrame([r])[features].replace([np.inf,-np.inf],np.nan).fillna(0)
        p = max(0.0, float(model.predict(x)[0]))
        if r.get("event_impact_pct",0): p *= (1+r.event_impact_pct/100)
        preds.append(p)
        work = pd.concat([work,pd.DataFrame([{**r,"energy_kwh":p}])],ignore_index=True)
    future["xgb_pred_kwh"] = preds
    return future


def lstm_fit_forecast(train_df, future, feature_cols, lookback=48, epochs=20):
    if not TF_OK: return None, None
    base = train_df.copy()
    X,Y,sx,sy = make_lstm_sequences(base, feature_cols, "energy_kwh", lookback)
    model=build_lstm((X.shape[1],X.shape[2]))
    es=EarlyStopping(monitor="val_loss",patience=4,restore_best_weights=True)
    model.fit(X,Y,epochs=epochs,batch_size=64,validation_split=.12,shuffle=False,callbacks=[es],verbose=0)
    # Recursive future sequence. For unknown future demand, use XGB prediction as an auxiliary
    # feature surrogate for the target column if target is among feature_cols; here it is not.
    combined=base.copy()
    preds=[]
    for _, r in future.iterrows():
        feat=r.copy()
        # LSTM features are all known calendar + lag/rolling features. No future target required.
        seq_rows = pd.concat([combined.tail(lookback), pd.DataFrame([feat])],ignore_index=True)
        vals=sx.transform(seq_rows[feature_cols].replace([np.inf,-np.inf],np.nan).fillna(0))[-lookback:]
        p=float(sy.inverse_transform(model.predict(vals[np.newaxis,:,:],verbose=0))[0,0])
        p=max(0,p); preds.append(p)
        combined=pd.concat([combined,pd.DataFrame([{**feat,"energy_kwh":p}])],ignore_index=True)
    return model, np.array(preds)


def validation_scores(city, features, lookback=48):
    # Last 7 complete days as holdout.
    cutoff=city.timestamp.max()-pd.Timedelta(days=7)
    tr=city[city.timestamp < cutoff].copy(); va=city[city.timestamp >= cutoff].copy()
    model=xgb_train(tr,features)
    px=model.predict(va[features].replace([np.inf,-np.inf],np.nan).fillna(0))
    return {"XGBoost MAE":mean_absolute_error(va.energy_kwh,px),"XGBoost RMSE":rmse(va.energy_kwh,px),"XGBoost MAPE":mape(va.energy_kwh,px)}, va, px


# ----------------------------- App -----------------------------
st.markdown('<div class="hero"><h1>⚡EV Charging Demand Intelligence</h1><p>Hybrid XGBoost + LSTM forecasting • full-day demand prediction • trend & event awareness • station analytics</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Data & Forecast")
    session_up = st.file_uploader("Session CSV (optional)", type="csv")
    hourly_up = st.file_uploader("Hourly usage CSV", type="csv")
    station_up = st.file_uploader("Station CSV", type="csv")
    event_up = st.file_uploader("Events CSV (optional)", type="csv", help="Columns: date,event_name,impact_pct and optional event_type,start_hour,end_hour")
    st.divider()
    forecast_mode = st.radio("Forecast date", ["Today", "Choose date"], index=0)
    chosen = st.date_input("Date", value=pd.Timestamp.today().date()) if forecast_mode=="Choose date" else pd.Timestamp.today().date()
    blend = st.slider("XGBoost weight", 0.0, 1.0, 0.60, 0.05)
    epochs = st.slider("LSTM epochs", 5, 40, 18, 1)
    run = st.button("🚀 Run / Refresh Forecast", type="primary", use_container_width=True)

hourly = read_csv(hourly_up, DEFAULT_HOURLY)
stations = read_csv(station_up, DEFAULT_STATIONS)
sessions = read_csv(session_up, DEFAULT_SESSION)
if hourly is None or stations is None:
    st.error("Upload the hourly usage CSV and station CSV, or place the supplied Kavali files beside app.py.")
    st.stop()

try:
    hourly_bytes = hourly.to_csv(index=False).encode(); stations_bytes=stations.to_csv(index=False).encode()
    hs, city, stations = prepare_data(hourly_bytes,stations_bytes)
    events = load_events(event_up)
except Exception as e:
    st.error(f"Data preparation failed: {e}"); st.stop()

# Limit forecast date to a meaningful operational horizon; future dates are allowed.
forecast_date=pd.Timestamp(chosen).normalize()
last_hist=city.timestamp.max().normalize()

# ----------------------------- Overview -----------------------------
min_date=city.date.min().date(); max_date=city.date.max().date()
mean_daily=city.groupby("date").energy_kwh.sum().mean()
latest_daily=city[city.date==city.date.max()].energy_kwh.sum()

c1,c2,c3,c4,c5=st.columns(5)
for c,title,val,sub in [
    (c1,"DATA PERIOD",f"{min_date} → {max_date}","1 year historical coverage"),
    (c2,"SESSIONS",f"{int(sessions.shape[0]):,}" if sessions is not None else "—","charging sessions"),
    (c3,"STATIONS",str(hs.station_id.nunique()),"Kavali charging sites"),
    (c4,"AVG DAILY ENERGY",f"{mean_daily:,.0f} kWh","historical mean"),
    (c5,"LATEST DAY",f"{latest_daily:,.0f} kWh","observed historical day")]:
    c.markdown(f'<div class="card"><div class="metric-title">{title}</div><div class="metric-value">{val}</div><div class="small">{sub}</div></div>',unsafe_allow_html=True)

st.write("")

tab1,tab2,tab3,tab4=st.tabs(["🔮 Forecast","📊 Analytics","🏢 Stations","🧠 Model & Data"])

# Features used by both models. No target included: all are known/constructible at forecast time.
FEATURES=["hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos","dayofyear_sin","dayofyear_cos",
          "is_weekend","trend_day","lag_1","lag_2","lag_3","lag_24","lag_48","lag_72","lag_168","lag_336",
          "roll_mean_3","roll_mean_6","roll_mean_24","roll_mean_72","roll_mean_168",
          "roll_std_3","roll_std_6","roll_std_24","roll_std_72","roll_std_168","growth_30d"]

with tab1:
    st.subheader(f"Full-day demand forecast — {forecast_date.strftime('%A, %d %B %Y')}")
    if forecast_date <= last_hist:
        st.info("Selected date is inside the historical period. Forecast is generated as a model simulation; actual values remain available for comparison.")
    elif forecast_date > last_hist + pd.Timedelta(days=30):
        st.warning("This date is beyond the historical range. Forecast uncertainty will increase because the model must extrapolate further into the future.")

    # Event overlay is applied before final display.
    with st.spinner("Training XGBoost and preparing the hybrid forecast…"):
        xgb=xgb_train(city,FEATURES)
        fut=build_future_rows(city,forecast_date,events)
        fut=event_adjustment(fut,events)
        xgb_pred=[]
        work=city.copy()
        for _,r in fut.iterrows():
            xx=pd.DataFrame([r])[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0)
            p=max(0,float(xgb.predict(xx)[0]))
            p*=max(0.0,1+r.event_impact_pct/100)
            xgb_pred.append(p)
            work=pd.concat([work,pd.DataFrame([{**r,"energy_kwh":p}])],ignore_index=True)
        fut["xgb_pred_kwh"]=xgb_pred
        lstm_model,lstm_pred=lstm_fit_forecast(city,fut,FEATURES,lookback=48,epochs=epochs)
        if lstm_pred is None:
            fut["lstm_pred_kwh"]=fut.xgb_pred_kwh.values
            st.warning("TensorFlow/Keras is not installed, so the current run uses XGBoost for the forecast. Install the requirements file to activate LSTM.")
        else:
            fut["lstm_pred_kwh"]=lstm_pred*(1+fut.event_impact_pct.values/100)
        fut["forecast_kwh"]=blend*fut.xgb_pred_kwh+(1-blend)*fut.lstm_pred_kwh
        fut["forecast_kwh"]=fut.forecast_kwh.clip(lower=0)

    total=fut.forecast_kwh.sum(); peak=fut.loc[fut.forecast_kwh.idxmax()]; avg=fut.forecast_kwh.mean()
    event_hours=fut[fut.event_impact_pct!=0]
    m1,m2,m3,m4=st.columns(4)
    for c,t,v,s in [(m1,"PREDICTED DAILY ENERGY",f"{total:,.1f} kWh","24-hour total"),(m2,"PEAK HOUR",peak.timestamp.strftime("%I:%M %p"),f"{peak.forecast_kwh:,.1f} kWh"),(m3,"AVG HOURLY",f"{avg:,.1f} kWh","daily mean"),(m4,"EVENT HOURS",str(len(event_hours)),"event-adjusted hours")]:
        c.markdown(f'<div class="card"><div class="metric-title">{t}</div><div class="metric-value">{v}</div><div class="small">{s}</div></div>',unsafe_allow_html=True)

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=fut.timestamp,y=fut.forecast_kwh,mode="lines+markers",name="Hybrid forecast",line=dict(width=4),fill="tozeroy"))
    fig.add_trace(go.Scatter(x=fut.timestamp,y=fut.xgb_pred_kwh,mode="lines",name="XGBoost",line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=fut.timestamp,y=fut.lstm_pred_kwh,mode="lines",name="LSTM",line=dict(dash="dot")))
    if len(event_hours):
        fig.add_trace(go.Scatter(x=event_hours.timestamp,y=event_hours.forecast_kwh,mode="markers",name="Event impact",marker=dict(size=11,symbol="star")))
    fig.update_layout(height=430,margin=dict(l=10,r=10,t=35,b=10),xaxis_title="Time",yaxis_title="Energy demand (kWh)",hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)

    left,right=st.columns([1.35,1])
    with left:
        st.markdown("### Hour-by-hour forecast")
        table=fut[["timestamp","xgb_pred_kwh","lstm_pred_kwh","forecast_kwh","event_impact_pct","event_name"]].copy()
        table.columns=["Time","XGBoost (kWh)","LSTM (kWh)","Hybrid (kWh)","Event impact %","Event"]
        table["Time"]=table.Time.dt.strftime("%I:%M %p")
        st.dataframe(table.style.format({"XGBoost (kWh)":"{:.1f}","LSTM (kWh)":"{:.1f}","Hybrid (kWh)":"{:.1f}","Event impact %":"{:.1f}%"}),use_container_width=True,hide_index=True)
    with right:
        st.markdown("### Forecast interpretation")
        if len(event_hours):
            st.success(f"Event-aware forecast: {len(event_hours)} hours receive an event adjustment. The event inputs are scenario assumptions and should be replaced with verified local event information for deployment.")
        st.markdown(f"**Peak:** {peak.timestamp.strftime('%A %I:%M %p')} at **{peak.forecast_kwh:,.1f} kWh**.")
        st.markdown(f"**Day total:** approximately **{total:,.1f} kWh** across the four monitored stations.")
        st.markdown("The model combines recent lags, same-week patterns, calendar seasonality and a long-term trend proxy rather than simply copying the same date from the previous year.")

with tab2:
    st.subheader("📊 Historical charging analytics")
    daily=city.groupby("date",as_index=False).agg(energy_kwh=("energy_kwh","sum"),sessions=("sessions","sum"),busy_min=("busy_min","sum"))
    monthly=daily.assign(month=daily.date.dt.to_period("M")).groupby("month",as_index=False).agg(energy_kwh=("energy_kwh","sum"),sessions=("sessions","sum"))
    fig=make_subplots(rows=2,cols=1,shared_xaxes=False,vertical_spacing=.12,subplot_titles=("Daily energy demand","Monthly energy demand"))
    fig.add_trace(go.Scatter(x=daily.date,y=daily.energy_kwh,mode="lines",name="Daily"),row=1,col=1)
    fig.add_trace(go.Bar(x=monthly.month.astype(str),y=monthly.energy_kwh,name="Monthly"),row=2,col=1)
    fig.update_layout(height=650,margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig,use_container_width=True)
    a,b,c=st.columns(3)
    peak_day=daily.loc[daily.energy_kwh.idxmax()]
    a.metric("Peak historical day",str(peak_day.date.date()),f"{peak_day.energy_kwh:,.1f} kWh")
    b.metric("Highest month",str(monthly.loc[monthly.energy_kwh.idxmax(),"month"]),f"{monthly.energy_kwh.max():,.0f} kWh")
    c.metric("Average sessions/day",f"{daily.sessions.mean():,.1f}")
    st.markdown("### Hour × day-of-week pattern")
    pivot=city.pivot_table(index="hour",columns="dayofweek",values="energy_kwh",aggfunc="mean")
    pivot.columns=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    heat=go.Figure(data=go.Heatmap(z=pivot.values,x=pivot.columns,y=pivot.index,colorscale="Blues",colorbar_title="kWh"))
    heat.update_layout(height=500,xaxis_title="Day",yaxis_title="Hour")
    st.plotly_chart(heat,use_container_width=True)

with tab3:
    st.subheader("🏢 Station-level intelligence")
    station_daily=hs.groupby(["station_id","date"],as_index=False).agg(energy_kwh=("energy_kwh","sum"),sessions=("sessions","sum"),busy_min=("busy_min","sum"))
    station_summary=station_daily.groupby("station_id",as_index=False).agg(total_energy_kwh=("energy_kwh","sum"),avg_daily_kwh=("energy_kwh","mean"),total_sessions=("sessions","sum"),avg_busy_min=("busy_min","mean"))
    station_summary=station_summary.merge(stations,on="station_id",how="left")
    st.dataframe(station_summary.sort_values("total_energy_kwh",ascending=False).style.format({"total_energy_kwh":"{:,.0f}","avg_daily_kwh":"{:,.1f}","total_sessions":"{:,.0f}","avg_busy_min":"{:,.1f}"}),use_container_width=True,hide_index=True)
    selected=st.selectbox("Inspect station",station_summary.station_id.tolist())
    sd=station_daily[station_daily.station_id==selected]
    fig=go.Figure(go.Scatter(x=sd.date,y=sd.energy_kwh,mode="lines",name=selected,fill="tozeroy"))
    fig.update_layout(height=350,xaxis_title="Date",yaxis_title="Daily energy (kWh)")
    st.plotly_chart(fig,use_container_width=True)

with tab4:
    st.subheader("🧠 Model, data quality & methodology")
    st.markdown("**Primary model:** XGBoost gradient-boosted regression. **Secondary model:** LSTM sequence model. **Final forecast:** weighted hybrid of both models.")
    st.markdown("**Target:** hourly station-network energy demand (`energy_kwh`).")
    st.markdown("**Key predictors:** hour/day/month seasonality, weekend status, lag-1/2/3, lag-24/48/72, lag-168/336, rolling demand statistics, and a 30-day growth proxy.")
    st.markdown("**Why not random train/test splitting?** Time-series data must be split chronologically so future information does not leak into training.")
    st.markdown("**Event layer:** optional event CSV allows transparent scenario adjustments. Event impact is an explicit input rather than an invented hidden assumption.")
    if sessions is not None:
        st.markdown(f"**Session dataset:** {len(sessions):,} records, {sessions.date.min()} to {sessions.date.max()}; used for contextual/session-level analytics.")
    st.markdown(f"**Hourly dataset:** {len(hs):,} station-hour observations across {hs.station_id.nunique()} stations.")
    if not TF_OK:
        st.warning("TensorFlow/Keras is unavailable in this environment. Install requirements.txt to enable the LSTM branch.")

    # Chronological validation for transparency.
    with st.spinner("Calculating chronological XGBoost validation metrics…"):
        scores,va,px=validation_scores(city,FEATURES)
    vc1,vc2,vc3=st.columns(3)
    vc1.metric("Validation MAE",f"{scores['XGBoost MAE']:.2f} kWh")
    vc2.metric("Validation RMSE",f"{scores['XGBoost RMSE']:.2f} kWh")
    vc3.metric("Validation MAPE",f"{scores['XGBoost MAPE']:.2f}%")
    st.caption("Validation uses the last 7 historical days as a chronological holdout. These metrics are for the XGBoost component; the displayed daily forecast is the hybrid model.")

# ----------------------------- Footer -----------------------------
st.divider()
st.caption("Prototype note: the supplied Kavali dataset is synthetic. It is suitable for demonstrating the forecasting architecture, but model results should not be presented as measured real-world Kavali demand. Verify local event schedules and operational data before real deployment.")
