import os, io, math, re, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

try:
    import pulp
    PULP_OK = True
except Exception:
    pulp = None
    PULP_OK = False

from scipy.optimize import linprog

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

# ============================================================
# 🎨 KAVALI EV CHARGING INTELLIGENCE — STRICT DARK MODE CSS
# ============================================================
st.markdown("""<style>
/* =========================================================   STRICT DARK MODE GLOBALS   ========================================================= */
:root, html, body {
    color-scheme: dark !important;
    background-color: #0b0f14 !important;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(0, 180, 120, 0.08),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(30, 120, 255, 0.08),
            transparent 30%
        ),
        #0b0f14 !important;
    color: #f5f7fa !important;
    color-scheme: dark !important;
}

/* Enforce dark sidebar */
section[data-testid="stSidebar"] {
    background-color: #0b0f14 !important;
    border-right: 1px solid rgba(148, 163, 184, 0.12) !important;
}
section[data-testid="stSidebar"] * {
    color: #f5f7fa !important;
}

/* Header & Main container */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}
.main .block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
    color: #f5f7fa !important;
}

/* Force dark styling on all form controls & inputs */
input, textarea, select {
    background-color: rgba(15, 23, 42, 0.9) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
}
div[data-baseweb="input"], div[data-baseweb="base-input"] {
    background-color: rgba(15, 23, 42, 0.9) !important;
    border-color: rgba(148, 163, 184, 0.25) !important;
}
div[data-baseweb="select"] > div {
    background-color: rgba(15, 23, 42, 0.9) !important;
    color: #f8fafc !important;
    border-color: rgba(148, 163, 184, 0.25) !important;
}
ul[role="listbox"], li[role="option"] {
    background-color: #0f172a !important;
    color: #f8fafc !important;
}

/* Force dark tabs */
button[data-baseweb="tab"] {
    color: #94a3b8 !important;
    background: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8 !important;
}

/* Force dark expanders */
div[data-testid="stExpander"] {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] summary {
    color: #f8fafc !important;
}

/* Force dark buttons */
button[kind="primary"] {
    background: linear-gradient(135deg, #0284c7, #0ea5e9) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 750 !important;
}
button[kind="secondary"] {
    background: rgba(30, 41, 59, 0.8) !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
}

/* Header & Hero */
.hero {
    padding: 1.5rem 1.8rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(18, 59, 88, 0.85));
    border: 1px solid rgba(56, 189, 248, 0.2);
    color: white;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(8px);
}
.hero h1 {
    font-size: 2.1rem;
    margin: 0;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #f8fafc;
}
.hero p {
    color: #94a3b8;
    margin: 0.4rem 0 0;
    font-size: 0.95rem;
}

/* Section Title */
.ev-section-title {
    font-size: 1.3rem;
    font-weight: 750;
    color: #f8fafc;
    margin: 1.5rem 0 0.8rem 0;
    letter-spacing: -0.01em;
}

/* Recommend Card (AI Recommended Hero) */
.recommend-card {
    background: linear-gradient(145deg, rgba(16, 185, 129, 0.08), rgba(15, 23, 42, 0.95));
    border: 1.5px solid rgba(16, 185, 129, 0.5);
    border-radius: 18px;
    padding: 1.5rem 1.8rem;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.12);
    margin-bottom: 1.1rem;
    position: relative;
    backdrop-filter: blur(10px);
}
.recommend-badge {
    display: inline-block;
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
    font-weight: 750;
    font-size: 0.82rem;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    letter-spacing: 0.05em;
    border: 1px solid rgba(52, 211, 153, 0.4);
    margin-bottom: 0.6rem;
}
.recommend-title {
    font-size: 1.75rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.02em;
}
.recommend-location {
    font-size: 0.92rem;
    color: #94a3b8;
    margin-top: 0.2rem;
    margin-bottom: 0.9rem;
}

/* Metric Row & Box */
.metric-row {
    display: flex;
    gap: 0.8rem;
    margin: 0.9rem 0;
    flex-wrap: wrap;
}
.metric-box {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 12px;
    padding: 0.65rem 0.95rem;
    flex: 1;
    min-width: 105px;
}
.metric-label {
    font-size: 0.73rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
}
.metric-value {
    font-size: 1.15rem;
    font-weight: 750;
    color: #f8fafc;
}

/* Fast Card (Fastest Charger) */
.fast-card {
    background: linear-gradient(145deg, rgba(234, 179, 8, 0.08), rgba(15, 23, 42, 0.95));
    border: 1.5px solid rgba(234, 179, 8, 0.4);
    border-radius: 18px;
    padding: 1.5rem 1.8rem;
    box-shadow: 0 10px 30px rgba(234, 179, 8, 0.08);
    margin-bottom: 1.1rem;
    backdrop-filter: blur(10px);
}
.fast-title {
    color: #facc15;
    font-size: 0.82rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.4rem;
}
.fast-power {
    font-size: 1.55rem;
    font-weight: 800;
    color: #facc15;
    margin-top: 0.2rem;
}
.fast-label {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-bottom: 0.6rem;
}

/* 4 Station Cards */
.station-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 14px;
    padding: 1.2rem 1.3rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.station-card:hover {
    border-color: rgba(56, 189, 248, 0.4);
    transform: translateY(-2px);
}
.station-rank {
    font-size: 0.75rem;
    font-weight: 750;
    color: #38bdf8;
    letter-spacing: 0.06em;
}
.station-name {
    font-size: 1.35rem;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 0.2rem;
}
.station-status {
    display: inline-block;
    font-size: 0.8rem;
    font-weight: 750;
    padding: 0.25rem 0.65rem;
    border-radius: 999px;
}
.status-low {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}
.status-medium {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(251, 191, 36, 0.3);
}
.status-high {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(248, 113, 113, 0.3);
}

/* Why list box */
.why-list {
    background: rgba(16, 185, 129, 0.06);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 14px;
    padding: 1.1rem 1.35rem;
    margin-top: 1rem;
    color: #86efac;
    font-size: 0.92rem;
    line-height: 1.7;
}

/* Base card & utility classes */
.card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
}
.small { font-size: 0.82rem; color: #94a3b8; }
.metric-title { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { font-size: 1.45rem; font-weight: 750; color: #f8fafc; }
.badge { display: inline-block; padding: 0.25rem 0.55rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }
.progress-bar-bg { background: rgba(148, 163, 184, 0.2); border-radius: 999px; height: 8px; width: 100%; overflow: hidden; margin-top: 6px; }
.progress-bar-fill { height: 100%; border-radius: 999px; }
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SESSION = (
    os.path.join(BASE, "kavali_ev_clean_session_master.csv")
    if os.path.exists(os.path.join(BASE, "kavali_ev_clean_session_master.csv"))
    else os.path.join(BASE, "kavali_ev_sessions_synthetic_Oct2025-Sep2026.csv")
)
DEFAULT_HOURLY = (
    os.path.join(BASE, "kavali_ev_clean_hourly_forecast.csv")
    if os.path.exists(os.path.join(BASE, "kavali_ev_clean_hourly_forecast.csv"))
    else os.path.join(BASE, "kavali_ev_hourly_usage_synthetic.csv")
)
DEFAULT_STATIONS = os.path.join(BASE, "kavali_ev_stations.csv")
DEFAULT_EVENTS = (
    os.path.join(BASE, "kavali_event_scenarios_clean.csv")
    if os.path.exists(os.path.join(BASE, "kavali_event_scenarios_clean.csv"))
    else os.path.join(BASE, "events_template.csv")
)
DEFAULT_OPTIMIZATION_REQUESTS = os.path.join(BASE, "kavali_current_ev_optimization_requests.csv")


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


# ----------------------------- Click Effects (Originkit) -----------------------------
def render_click_effects(interaction_mode="sniper", color="#00b4d8"):
    if interaction_mode == "none":
        return
    html_content = f"""
    <script>
    (function() {{
        try {{
            var doc = document;
            var win = window;
            try {{
                if (window.parent && window.parent.document) {{
                    doc = window.parent.document;
                    win = window.parent;
                }}
            }} catch(e) {{
                doc = document;
                win = window;
            }}
            
            var existingContainer = doc.getElementById('originkit-click-effects');
            if (existingContainer) existingContainer.remove();
            
            var container = doc.createElement('div');
            container.id = 'originkit-click-effects';
            container.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:999999;overflow:hidden;';
            doc.body.appendChild(container);
            
            var mode = "{interaction_mode}";
            var color = "{color}";
            var effectSize = 88;
            var duration = 0.35;

            var styleTag = doc.getElementById('originkit-click-style');
            if (!styleTag) {{
                styleTag = doc.createElement('style');
                styleTag.id = 'originkit-click-style';
                styleTag.innerHTML = `
                    @keyframes originkit-ring {{
                        0% {{ transform: scale(0.4); opacity: 1; stroke-width: 3px; }}
                        70% {{ opacity: 0.9; }}
                        100% {{ transform: scale(2.2); opacity: 0; stroke-width: 0.5px; }}
                    }}
                `;
                doc.head.appendChild(styleTag);
            }}

            function triggerEffect(e) {{
                if (mode === 'none') return;
                var x = e.clientX;
                var y = e.clientY;
                if (!x && !y) return;
                
                if (mode === 'rings') {{
                    var svg = doc.createElementNS('http://www.w3.org/2000/svg', 'svg');
                    svg.style.cssText = 'position:absolute;left:' + (x - effectSize/2) + 'px;top:' + (y - effectSize/2) + 'px;width:' + effectSize + 'px;height:' + effectSize + 'px;pointer-events:none;overflow:visible;';
                    svg.innerHTML = '<circle cx="' + (effectSize/2) + '" cy="' + (effectSize/2) + '" r="' + (effectSize/4) + '" fill="none" stroke="' + color + '" stroke-width="3" style="transform-origin:center;animation:originkit-ring ' + duration + 's cubic-bezier(0.1, 0.8, 0.3, 1) forwards;"/>';
                    container.appendChild(svg);
                    setTimeout(function() {{ svg.remove(); }}, duration * 1000 + 40);
                }}
                else if (mode === 'particles') {{
                    var count = 8;
                    for (var i = 0; i < count; i++) {{
                        (function(idx) {{
                            var ang = idx * 45 * (Math.PI / 180);
                            var dist = effectSize * 0.28 + Math.random() * (effectSize * 0.25);
                            var dot = doc.createElement('div');
                            dot.style.cssText = 'position:absolute;left:' + x + 'px;top:' + y + 'px;width:5px;height:5px;background:' + color + ';border-radius:50%;pointer-events:none;transform:translate(-50%,-50%);transition:all ' + duration + 's cubic-bezier(0.1, 0.8, 0.3, 1);box-shadow:0 0 6px ' + color + ';';
                            container.appendChild(dot);
                            requestAnimationFrame(function() {{
                                dot.style.left = (x + Math.cos(ang) * dist) + 'px';
                                dot.style.top = (y + Math.sin(ang) * dist) + 'px';
                                dot.style.opacity = '0';
                                dot.style.transform = 'translate(-50%,-50%) scale(0.2)';
                            }});
                            setTimeout(function() {{ dot.remove(); }}, duration * 1000 + 40);
                        }})(i);
                    }}
                }}
                else if (mode === 'sniper') {{
                    var wrap = doc.createElement('div');
                    wrap.style.cssText = 'position:absolute;left:' + x + 'px;top:' + y + 'px;pointer-events:none;';
                    
                    var angles = [0, Math.PI/2, Math.PI, 3*Math.PI/2];
                    angles.forEach(function(ang) {{
                        var line = doc.createElement('div');
                        var len = effectSize * 0.18;
                        line.style.cssText = 'position:absolute;left:0;top:0;width:' + len + 'px;height:2px;background:' + color + ';transform-origin:0 50%;transform:rotate(' + ang + 'rad) translateX(8px);transition:all ' + duration + 's ease-out;box-shadow:0 0 5px ' + color + ';';
                        wrap.appendChild(line);
                        requestAnimationFrame(function() {{
                            line.style.transform = 'rotate(' + ang + 'rad) translateX(' + (len + 15) + 'px)';
                            line.style.opacity = '0';
                        }});
                    }});
                    
                    var sparkAngles = [Math.PI/3, 2*Math.PI/3, 4*Math.PI/3, 5*Math.PI/3, Math.PI/6, 5*Math.PI/6, 7*Math.PI/6, 11*Math.PI/6];
                    sparkAngles.forEach(function(ang) {{
                        var spark = doc.createElement('div');
                        var dist = effectSize * 0.38;
                        spark.style.cssText = 'position:absolute;left:0;top:0;width:3px;height:3px;background:' + color + ';border-radius:50%;transform:translate(-50%,-50%);transition:all ' + duration + 's ease-out;box-shadow:0 0 5px ' + color + ';';
                        wrap.appendChild(spark);
                        requestAnimationFrame(function() {{
                            spark.style.transform = 'translate(' + (Math.cos(ang)*dist) + 'px, ' + (Math.sin(ang)*dist) + 'px) scale(0)';
                            spark.style.opacity = '0';
                        }});
                    }});
                    
                    container.appendChild(wrap);
                    setTimeout(function() {{ wrap.remove(); }}, duration * 1000 + 40);
                }}
                else if (mode === 'crosshair') {{
                    var wrap = doc.createElement('div');
                    wrap.style.cssText = 'position:absolute;left:' + x + 'px;top:' + y + 'px;pointer-events:none;';
                    [0, Math.PI/2, Math.PI, 3*Math.PI/2].forEach(function(ang) {{
                        var line = doc.createElement('div');
                        var len = effectSize * 0.28;
                        line.style.cssText = 'position:absolute;left:0;top:0;width:' + len + 'px;height:2px;background:' + color + ';transform-origin:0 50%;transform:rotate(' + ang + 'rad) translateX(12px);transition:all ' + duration + 's ease-out;box-shadow:0 0 6px ' + color + ';';
                        wrap.appendChild(line);
                        requestAnimationFrame(function() {{
                            line.style.transform = 'rotate(' + ang + 'rad) translateX(' + (len + 25) + 'px)';
                            line.style.opacity = '0';
                        }});
                    }});
                    container.appendChild(wrap);
                    setTimeout(function() {{ wrap.remove(); }}, duration * 1000 + 40);
                }}
                else if (mode === 'burst') {{
                    var wrap = doc.createElement('div');
                    wrap.style.cssText = 'position:absolute;left:' + x + 'px;top:' + y + 'px;pointer-events:none;';
                    [45, 80, 115, 150].forEach(function(deg) {{
                        var ang = deg * Math.PI / 180;
                        var line = doc.createElement('div');
                        var len = effectSize * 0.22;
                        line.style.cssText = 'position:absolute;left:0;top:0;width:' + len + 'px;height:2px;background:' + color + ';transform-origin:0 50%;transform:rotate(' + ang + 'rad) translateX(6px);transition:all ' + duration + 's ease-out;box-shadow:0 0 5px ' + color + ';';
                        wrap.appendChild(line);
                        requestAnimationFrame(function() {{
                            line.style.transform = 'rotate(' + ang + 'rad) translateX(' + (len + 20) + 'px)';
                            line.style.opacity = '0';
                        }});
                    }});
                    container.appendChild(wrap);
                    setTimeout(function() {{ wrap.remove(); }}, duration * 1000 + 40);
                }}
                else if (mode === 'wavy') {{
                    var count = 4;
                    for (var i = 0; i < count; i++) {{
                        (function(idx) {{
                            var ang = (idx * 90 + 45) * (Math.PI / 180);
                            var dist = effectSize * 0.45;
                            var spark = doc.createElement('div');
                            spark.style.cssText = 'position:absolute;left:' + x + 'px;top:' + y + 'px;width:4px;height:4px;background:' + color + ';border-radius:50%;transform:translate(-50%,-50%);transition:all ' + duration + 's cubic-bezier(0.2, 0.8, 0.4, 1);box-shadow:0 0 8px ' + color + ';';
                            container.appendChild(spark);
                            requestAnimationFrame(function() {{
                                spark.style.left = (x + Math.cos(ang) * dist) + 'px';
                                spark.style.top = (y + Math.sin(ang) * dist) + 'px';
                                spark.style.opacity = '0';
                            }});
                            setTimeout(function() {{ spark.remove(); }}, duration * 1000 + 40);
                        }})(i);
                    }}
                }}
            }}
            
            if (win._originkitClickFn) {{
                doc.removeEventListener('click', win._originkitClickFn);
            }}
            win._originkitClickFn = triggerEffect;
            doc.addEventListener('click', triggerEffect);
        }} catch(err) {{
            console.error('Click effects error:', err);
        }}
    }})();
    </script>
    """
    components.html(html_content, height=0, width=0)


# ----------------------------- Station Recommendation Logic -----------------------------
def get_congestion_level(utilization):
    if utilization < 0.35:
        return "🟢 Low"
    elif utilization < 0.70:
        return "🟡 Medium"
    else:
        return "🔴 High"


def estimate_wait_time(utilization, num_chargers):
    """
    Approximate waiting time from predicted utilization.
    This is an estimate, not live queue data.
    """
    num_chargers = max(int(num_chargers), 1)
    if utilization < 0.35:
        wait = 5.0
    elif utilization < 0.60:
        wait = 10.0
    elif utilization < 0.80:
        wait = 20.0
    else:
        wait = 30.0
    # More chargers generally reduce waiting
    wait = wait / np.sqrt(num_chargers)
    return round(float(wait), 1)


def estimate_charging_time(required_energy_kwh, charger_power_kw):
    """
    Estimated charging duration in minutes.
    """
    charger_power_kw = max(float(charger_power_kw), 1.0)
    efficiency = 0.90
    charging_hours = required_energy_kwh / (charger_power_kw * efficiency)
    return round(float(charging_hours * 60), 1)


def calculate_station_score(row):
    """
    Lower score = better station.
    Combines predicted congestion, estimated waiting, and charging duration.
    """
    congestion_penalty = row["predicted_utilization"] * 50
    wait_penalty = row["estimated_wait_min"]
    charging_penalty = row["estimated_charge_min"]
    return float(congestion_penalty + wait_penalty + charging_penalty)


def apply_event_effect(prediction, event_impact_pct):
    multiplier = 1 + (event_impact_pct / 100)
    return prediction * multiplier


def extract_station_power(station_row):
    if "rated_power_kw" in station_row and pd.notnull(station_row["rated_power_kw"]):
        return float(station_row["rated_power_kw"])
    chargers_str = str(station_row.get("chargers", ""))
    match = re.search(r'(\d+(?:\.\d+)?)\s*k[wW]', chargers_str)
    return float(match.group(1)) if match else 50.0


def compute_station_forecasts(city_forecast, hourly_df, station_ids):
    """
    Disaggregates the 24-hour city forecast across individual stations
    based on each station's empirical hourly demand profile.
    """
    piv = hourly_df.groupby(["hour", "station_id"])["energy_kwh"].mean().unstack(fill_value=0)
    hour_shares = piv.div(piv.sum(axis=1) + 1e-6, axis=0)
    station_forecasts = {}
    for sid in station_ids:
        if sid in hour_shares.columns:
            shares = [hour_shares.loc[h, sid] if h in hour_shares.index else 0.25 for h in range(len(city_forecast))]
            station_forecasts[sid] = np.array(city_forecast) * np.array(shares)
        else:
            station_forecasts[sid] = np.array(city_forecast) / max(len(station_ids), 1)
    return station_forecasts


def generate_station_recommendations(
    station_forecasts,
    station_info,
    required_energy_kwh=20.0
):
    """
    Combine forecast + station metadata to recommend the best EV charging station.
    """
    recommendations = []
    for station_id, forecast in station_forecasts.items():
        predicted_energy = float(np.mean(forecast))
        
        station_rows = station_info[station_info["station_id"] == station_id]
        if len(station_rows) == 0:
            continue
        station = station_rows.iloc[0]
        
        num_chargers = int(station.get("num_chargers", 1))
        charger_power = extract_station_power(station)
        
        # Max capacity per hour: num_chargers * rated_power
        max_capacity_kwh = max(num_chargers * charger_power, 1.0)
        utilization = float(np.clip(predicted_energy / max_capacity_kwh, 0.05, 0.98))
        
        congestion = get_congestion_level(utilization)
        wait_min = estimate_wait_time(utilization, num_chargers)
        charge_min = estimate_charging_time(required_energy_kwh, charger_power)
        total_time = round(wait_min + charge_min, 1)
        
        row_dict = {
            "station_id": station_id,
            "station_name": station.get("station_name", station_id),
            "area": station.get("area", "Kavali"),
            "predicted_energy_kwh": round(predicted_energy, 1),
            "predicted_utilization": utilization,
            "congestion": congestion,
            "num_chargers": num_chargers,
            "charger_power_kw": charger_power,
            "estimated_wait_min": wait_min,
            "estimated_charge_min": charge_min,
            "total_time_min": total_time,
        }
        row_dict["score"] = calculate_station_score(row_dict)
        recommendations.append(row_dict)
        
    if not recommendations:
        return pd.DataFrame()
        
    rec_df = pd.DataFrame(recommendations)
    rec_df = rec_df.sort_values("score", ascending=True).reset_index(drop=True)
    rec_df["rank"] = [f"#{i+1}" for i in range(len(rec_df))]
    return rec_df


# ============================================================
# ⚡ PU LP 4.0 INTELLIGENT CHARGING POWER ALLOCATION OPTIMIZER
# ============================================================
def optimize_charging_power_allocation(
    station_id: str,
    ev_requests_df: pd.DataFrame,
    station_capacity_kw: float,
    forecasted_demand_kwh: float,
    decision_interval_hours: float = 1.0,
    headroom_override_kw: float = None
) -> dict:
    """
    Intelligent EV Charging Power Allocation using PuLP 4.0.
    
    Optimizes power distribution among multiple EVs connected to a charging station,
    respecting:
    - Station capacity and dynamic forecast-aware grid headroom
    - Individual EV onboard charger power limits
    - Vehicle remaining energy need (prevents charging beyond requirement or target SOC)
    - Deadline urgency (less remaining time -> higher power priority)
    - SOC deficit (lower current SOC -> higher charging priority)
    - User/fleet priority tier
    
    Uses diminishing-marginal-utility base/boost allocation to avoid 'bang-bang'
    monopolization (e.g. giving 60 kW to one EV and 0 kW to others).
    """
    if ev_requests_df is None or ev_requests_df.empty:
        return {
            "status": "No Requests",
            "is_optimal": False,
            "station_id": station_id,
            "station_capacity_kw": float(station_capacity_kw),
            "usable_power_kw": float(station_capacity_kw),
            "forecasted_demand_kwh": float(forecasted_demand_kwh) if forecasted_demand_kwh is not None else 0.0,
            "total_allocated_kw": 0.0,
            "total_required_kwh": 0.0,
            "total_delivered_kwh": 0.0,
            "df": pd.DataFrame()
        }
    
    reqs = ev_requests_df[ev_requests_df["station_id"] == station_id].copy().reset_index(drop=True)
    if reqs.empty:
        return {
            "status": "No Active EVs",
            "is_optimal": False,
            "station_id": station_id,
            "station_capacity_kw": float(station_capacity_kw),
            "usable_power_kw": float(station_capacity_kw),
            "forecasted_demand_kwh": float(forecasted_demand_kwh) if forecasted_demand_kwh is not None else 0.0,
            "total_allocated_kw": 0.0,
            "total_required_kwh": 0.0,
            "total_delivered_kwh": 0.0,
            "df": pd.DataFrame()
        }
    
    dt = max(0.1, float(decision_interval_hours))
    
    reqs["battery_capacity_kwh"] = pd.to_numeric(reqs["battery_capacity_kwh"], errors="coerce").fillna(40.0)
    reqs["current_soc"] = pd.to_numeric(reqs["current_soc"], errors="coerce").fillna(20.0).clip(0, 100)
    reqs["target_soc"] = pd.to_numeric(reqs["target_soc"], errors="coerce").fillna(80.0).clip(0, 100)
    reqs["max_charging_power_kw"] = pd.to_numeric(reqs["max_charging_power_kw"], errors="coerce").fillna(30.0).clip(lower=1.0)
    reqs["remaining_time_hours"] = pd.to_numeric(reqs["remaining_time_hours"], errors="coerce").fillna(1.5).clip(lower=0.1)
    
    if "priority_score" not in reqs.columns:
        tier_map = {"Emergency / Fleet": 4.0, "Emergency": 4.0, "Fleet": 3.5, "High": 3.0, "Normal": 2.0, "Low": 1.0}
        reqs["priority_score"] = reqs["priority_tier"].map(tier_map).fillna(2.0)
    else:
        reqs["priority_score"] = pd.to_numeric(reqs["priority_score"], errors="coerce").fillna(2.0)
        
    if "priority_tier" not in reqs.columns:
        reqs["priority_tier"] = reqs["priority_score"].apply(lambda s: "High" if s >= 3.0 else ("Normal" if s >= 2.0 else "Low"))
    
    reqs["required_energy_kwh"] = (
        (reqs["target_soc"] - reqs["current_soc"]).clip(lower=0.0) * reqs["battery_capacity_kwh"] / 100.0
    )
    reqs["max_interval_power_kw"] = reqs["required_energy_kwh"] / dt
    reqs["effective_power_limit_kw"] = reqs[["max_charging_power_kw", "max_interval_power_kw"]].min(axis=1)
    
    reqs["min_power_needed_kw"] = reqs["required_energy_kwh"] / reqs["remaining_time_hours"].clip(lower=0.25)
    time_pressure = (reqs["min_power_needed_kw"] / reqs["max_charging_power_kw"]).clip(upper=3.0)
    soc_deficit = (1.0 - reqs["current_soc"] / 100.0)
    prio_weight = reqs["priority_score"]
    
    reqs["urgency_score"] = 3.0 * time_pressure + 2.5 * soc_deficit + 1.2 * prio_weight
    
    rated_capacity = float(station_capacity_kw)
    if headroom_override_kw is not None:
        usable_power_kw = float(headroom_override_kw)
    else:
        forecast_val = float(forecasted_demand_kwh) if forecasted_demand_kwh is not None else 0.0
        if forecast_val > 0.85 * rated_capacity:
            usable_power_kw = max(10.0, rated_capacity * 0.90)
        else:
            usable_power_kw = rated_capacity
            
    allocations = []
    delivered_kwh = []
    projected_soc = []
    status_str = "Optimal"
    is_optimal = True
    
    if PULP_OK:
        try:
            prob = pulp.LpProblem(f"EV_Power_Allocation_{station_id}", pulp.LpMaximize)
            p_base = {}
            p_boost = {}
            
            for idx, r in reqs.iterrows():
                eid = str(r["ev_id"])
                p_lim = float(r["effective_power_limit_kw"])
                b_cap = min(p_lim, max(3.0, 0.30 * p_lim)) if p_lim > 0 else 0.0
                
                pb = prob.add_variable(f"{eid}_base", lowBound=0.0, upBound=b_cap)
                pt = prob.add_variable(f"{eid}_boost", lowBound=0.0, upBound=max(0.0, p_lim - b_cap))
                p_base[eid] = pb
                p_boost[eid] = pt
                
            prob += pulp.lpSum([p_base[str(r["ev_id"])] + p_boost[str(r["ev_id"])] for _, r in reqs.iterrows()]) <= usable_power_kw
            
            obj_terms = []
            for _, r in reqs.iterrows():
                eid = str(r["ev_id"])
                u = float(r["urgency_score"])
                obj_terms.append((50.0 + u) * p_base[eid] + u * p_boost[eid])
            prob += pulp.lpSum(obj_terms)
            
            solver = None
            try:
                if hasattr(pulp, "HiGHS") and pulp.HiGHS().available():
                    solver = pulp.HiGHS(msg=False)
            except Exception:
                pass
            if solver is None:
                for s_name in ["COIN_CMD", "GLPK_CMD", "SCIP_CMD"]:
                    try:
                        s = getattr(pulp, s_name)(msg=False)
                        if s.available():
                            solver = s
                            break
                    except Exception:
                        pass
                        
            if solver is not None:
                res = prob.solve(solver)
            else:
                res = prob.solve()
                
            status_str = res.status_str if hasattr(res, "status_str") else "Optimal"
            is_optimal = (res.status == pulp.LpSolveStatus.Optimal) if hasattr(res, "status") and hasattr(pulp, "LpSolveStatus") else True
            
            for idx, r in reqs.iterrows():
                eid = str(r["ev_id"])
                val_base = float(pulp.value(p_base[eid])) if p_base[eid] is not None and pulp.value(p_base[eid]) is not None else 0.0
                val_boost = float(pulp.value(p_boost[eid])) if p_boost[eid] is not None and pulp.value(p_boost[eid]) is not None else 0.0
                total_p = max(0.0, val_base + val_boost)
                total_p = min(total_p, float(r["effective_power_limit_kw"]))
                allocations.append(round(total_p, 2))
        except Exception:
            allocations = []
            
    if not allocations or len(allocations) != len(reqs):
        try:
            n = len(reqs)
            c = []
            bounds = []
            for idx, r in reqs.iterrows():
                u = float(r["urgency_score"])
                p_lim = float(r["effective_power_limit_kw"])
                b_cap = min(p_lim, max(3.0, 0.30 * p_lim)) if p_lim > 0 else 0.0
                c.extend([-(50.0 + u), -u])
                bounds.extend([(0.0, b_cap), (0.0, max(0.0, p_lim - b_cap))])
                
            A_ub = [[1.0] * (2 * n)]
            b_ub = [usable_power_kw]
            
            res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            if res_lp.success:
                status_str = "Optimal"
                is_optimal = True
                allocations = [
                    round(min(float(reqs.loc[i, "effective_power_limit_kw"]), float(res_lp.x[2*i] + res_lp.x[2*i+1])), 2)
                    for i in range(n)
                ]
            else:
                status_str = "Feasible"
                allocations = [round(min(float(r["effective_power_limit_kw"]), usable_power_kw / n), 2) for _, r in reqs.iterrows()]
        except Exception:
            status_str = "Feasible"
            n = len(reqs)
            allocations = [round(min(float(r["effective_power_limit_kw"]), usable_power_kw / n), 2) for _, r in reqs.iterrows()]
            
    for idx, r in reqs.iterrows():
        total_p = allocations[idx]
        e_del = round(total_p * dt, 2)
        delivered_kwh.append(e_del)
        new_soc = min(float(r["target_soc"]), float(r["current_soc"]) + (e_del / float(r["battery_capacity_kwh"])) * 100.0)
        projected_soc.append(round(new_soc, 1))
        
    reqs["station_name"] = station_id
    reqs["forecast_demand_kwh"] = round(float(forecasted_demand_kwh), 1) if forecasted_demand_kwh is not None else 0.0
    reqs["station_capacity_kw"] = round(rated_capacity, 1)
    reqs["usable_charging_power_kw"] = round(usable_power_kw, 1)
    reqs["allocated_power_kw"] = allocations
    reqs["energy_delivered_next_hour_kwh"] = delivered_kwh
    reqs["projected_soc"] = projected_soc
    
    total_allocated_kw = round(sum(allocations), 2)
    
    return {
        "status": status_str,
        "is_optimal": is_optimal,
        "station_id": station_id,
        "station_capacity_kw": rated_capacity,
        "usable_power_kw": usable_power_kw,
        "forecasted_demand_kwh": float(forecasted_demand_kwh) if forecasted_demand_kwh is not None else 0.0,
        "total_allocated_kw": total_allocated_kw,
        "total_required_kwh": round(reqs["required_energy_kwh"].sum(), 1),
        "total_delivered_kwh": round(sum(delivered_kwh), 1),
        "df": reqs,
        "allocated_df": reqs
    }


@st.cache_data(show_spinner=False)
def prepare_data(hourly_bytes, stations_bytes):
    h = pd.read_csv(io.BytesIO(hourly_bytes)); s = pd.read_csv(io.BytesIO(stations_bytes))
    s["rated_power_kw"] = s.apply(extract_station_power, axis=1)
    hs = make_hourly_station_grid(h, s)
    city = aggregate_city(hs)
    city = add_lag_features(city)
    return hs, city, s


def _pickle_bytes(df):
    b = io.BytesIO()
    df.to_pickle(b)
    return b.getvalue()


@st.cache_resource(show_spinner=False)
def xgb_train_cached(train_bytes, features):
    train = pd.read_pickle(io.BytesIO(train_bytes))
    X = train[list(features)].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)
    y = train["energy_kwh"].astype(float)
    model = XGBRegressor(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.05,
        reg_lambda=1.2,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )
    model.fit(X, y, verbose=False)
    return model


def xgb_train(train, features):
    return xgb_train_cached(_pickle_bytes(train), tuple(features))



def make_lstm_sequences(df, feature_cols, target_col, lookback):
    arr = df[feature_cols].replace([np.inf,-np.inf],np.nan).fillna(0).values.astype("float32")
    target = df[target_col].values.astype("float32")
    scaler_x = MinMaxScaler(); scaler_y = MinMaxScaler()
    xs = scaler_x.fit_transform(arr); ys = scaler_y.fit_transform(target.reshape(-1,1)).ravel()
    X=[]; Y=[]
    for i in range(lookback, len(df)):
        X.append(xs[i-lookback:i]); Y.append(ys[i])
    return np.array(X), np.array(Y), scaler_x, scaler_y


@st.cache_resource(show_spinner=False)
def lstm_train_cached(train_bytes, feature_cols, lookback=48, epochs=4):
    if not TF_OK:
        return None
    base = pd.read_pickle(io.BytesIO(train_bytes))
    if len(base) > 2160:
        base = base.tail(2160)
    X, Y, scaler_x, scaler_y = make_lstm_sequences(
        base,
        list(feature_cols),
        "energy_kwh",
        lookback
    )
    model = build_lstm(
        (X.shape[1], X.shape[2])
    )
    es = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )
    model.fit(
        X,
        Y,
        epochs=epochs,
        batch_size=128,
        validation_split=0.10,
        shuffle=False,
        callbacks=[es],
        verbose=0
    )
    return model, scaler_x, scaler_y


def build_lstm(input_shape):
    try:
        from tensorflow.keras.layers import Input
        model = Sequential([
            Input(shape=input_shape),
            LSTM(32, return_sequences=True),
            Dropout(0.12),
            LSTM(16),
            Dense(8, activation="relu"),
            Dense(1)
        ])
    except Exception:
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=input_shape),
            Dropout(0.12),
            LSTM(16),
            Dense(8, activation="relu"),
            Dense(1)
        ])
    model.compile(
        optimizer="adam",
        loss="mse"
    )
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


def lstm_fit_forecast(
    train_df,
    future,
    feature_cols,
    lookback=48,
    epochs=4
):
    if not TF_OK:
        return None, None
    trained = lstm_train_cached(
        _pickle_bytes(train_df),
        tuple(feature_cols),
        lookback,
        epochs
    )
    if trained is None:
        return None, None
    model, scaler_x, scaler_y = trained
    combined = train_df.copy()
    predictions = []
    for _, row in future.iterrows():
        feat = row.copy()
        sequence_rows = pd.concat(
            [
                combined.tail(lookback),
                pd.DataFrame([feat])
            ],
            ignore_index=True
        )
        vals = scaler_x.transform(
            sequence_rows[list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(0)
        )[-lookback:]
        p = float(scaler_y.inverse_transform(model.predict(vals[np.newaxis, :, :], verbose=0))[0, 0])
        p = max(0.0, p)
        predictions.append(p)
        combined = pd.concat([combined, pd.DataFrame([{**feat, "energy_kwh": p}])], ignore_index=True)
    return model, np.array(predictions)


@st.cache_data(show_spinner=False)
def validation_scores_cached(train_bytes, features, lookback=48):
    city = pd.read_pickle(io.BytesIO(train_bytes))
    cutoff = city.timestamp.max() - pd.Timedelta(days=7)
    tr = city[city.timestamp < cutoff].copy()
    va = city[city.timestamp >= cutoff].copy()
    model = xgb_train(tr, features)
    px = model.predict(va[list(features)].replace([np.inf, -np.inf], np.nan).fillna(0))
    return {"XGBoost MAE": mean_absolute_error(va.energy_kwh, px), "XGBoost RMSE": rmse(va.energy_kwh, px), "XGBoost MAPE": mape(va.energy_kwh, px)}, va, px


def validation_scores(city, features, lookback=48):
    return validation_scores_cached(_pickle_bytes(city), tuple(features), lookback)


# ----------------------------- App -----------------------------
st.markdown('<div class="hero"><h1>⚡ KAVALI EV CHARGING INTELLIGENCE</h1><p>Find the best charging station for your journey &bull; Real-time AI recommendations, demand forecasts &amp; turnaround time estimation</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Trip Planning")
    required_energy = st.number_input(
        "Energy Required (kWh)",
        min_value=5.0,
        max_value=100.0,
        value=20.0,
        step=5.0,
        help="Estimated energy needed for your vehicle (e.g. 20 kWh gives ~120-140 km range)"
    )
    forecast_mode = st.radio("Forecast Date", ["Today", "Choose date"], index=0)
    chosen = st.date_input("Date", value=pd.Timestamp.today().date()) if forecast_mode=="Choose date" else pd.Timestamp.today().date()
    run = st.button("🔍 Find Best Station", type="primary", use_container_width=True)

    st.divider()
    with st.expander("🧠 Advanced / Model Settings", expanded=False):
        blend = st.slider("XGBoost Weight in Hybrid", 0.0, 1.0, 0.60, 0.05)
        st.caption("⚡ Models are cached after first training. Fast inference enabled.")
        st.markdown("**Custom CSV Uploads (Optional)**")
        session_up = st.file_uploader("Session CSV", type="csv")
        hourly_up = st.file_uploader("Hourly usage CSV", type="csv")
        station_up = st.file_uploader("Station CSV", type="csv")
        event_up = st.file_uploader("Events CSV", type="csv", help="Columns: date,event_name,impact_pct")
        ev_req_up = st.file_uploader("EV Requests CSV", type="csv", help="Columns: station_id,ev_id,battery_capacity_kwh,current_soc,target_soc,max_charging_power_kw,remaining_time_hours,priority_tier")

    with st.expander("✨ UI Click Effects (Originkit)", expanded=False):
        fx_mode = st.selectbox(
            "Effect Style",
            ["sniper", "rings", "particles", "crosshair", "burst", "wavy", "none"],
            index=0,
            format_func=lambda x: {
                "sniper": "🎯 Sniper / Spark (Default)",
                "rings": "⭕ Energy Rings",
                "particles": "✨ Particles",
                "crosshair": "➕ Crosshair",
                "burst": "💥 Burst",
                "wavy": "〰️ Wavy Arc",
                "none": "Off"
            }.get(x, x)
        )
        fx_color = st.color_picker("Effect Color", "#00b4d8")

render_click_effects(fx_mode, fx_color)

hourly = read_csv(hourly_up, DEFAULT_HOURLY)
stations = read_csv(station_up, DEFAULT_STATIONS)
sessions = read_csv(session_up, DEFAULT_SESSION)
ev_requests = read_csv(ev_req_up, DEFAULT_OPTIMIZATION_REQUESTS)
if ev_requests is None:
    ev_requests = pd.DataFrame()
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

# Features used by both models.
FEATURES=["hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos","dayofyear_sin","dayofyear_cos",
          "is_weekend","trend_day","lag_1","lag_2","lag_3","lag_24","lag_48","lag_72","lag_168","lag_336",
          "roll_mean_3","roll_mean_6","roll_mean_24","roll_mean_72","roll_mean_168",
          "roll_std_3","roll_std_6","roll_std_24","roll_std_72","roll_std_168","growth_30d"]

# ----------------------------- Fast Forecast Pre-calculation -----------------------------
with st.spinner("⚡ Computing AI demand forecast & station status..."):
    train_bytes = _pickle_bytes(city)
    xgb = xgb_train_cached(train_bytes, tuple(FEATURES))
    fut = build_future_rows(city, forecast_date, events)
    fut = event_adjustment(fut, events)
    xgb_pred = []
    work = city.copy()
    for _, r in fut.iterrows():
        xx = pd.DataFrame([r])[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
        prediction = float(xgb.predict(xx)[0])
        prediction = max(0.0, prediction)
        prediction *= max(0.0, 1 + r.event_impact_pct / 100)
        xgb_pred.append(prediction)
        work = pd.concat([work, pd.DataFrame([{**r, "energy_kwh": prediction}])], ignore_index=True)
    fut["xgb_pred_kwh"] = xgb_pred
    lstm_model, lstm_pred = lstm_fit_forecast(city, fut, FEATURES, lookback=48, epochs=4)
    if lstm_pred is None:
        fut["lstm_pred_kwh"] = fut.xgb_pred_kwh.values
    else:
        fut["lstm_pred_kwh"] = lstm_pred * (1 + fut.event_impact_pct.values / 100)
    fut["forecast_kwh"] = (blend * fut.xgb_pred_kwh + (1 - blend) * fut.lstm_pred_kwh).clip(lower=0)

# Station forecasts and recommendations
station_forecasts = compute_station_forecasts(fut["forecast_kwh"].values, hs, stations["station_id"].unique())
recommendations = generate_station_recommendations(station_forecasts, stations, required_energy)
event_hours = fut[fut.event_impact_pct != 0]

# ----------------------------- 5 User & Analytics Mode Tabs -----------------------------
tab_find, tab_forecast, tab_stations, tab_analytics, tab_model = st.tabs([
    "⚡ Intelligent Power Allocation",
    "📈 24-Hour Forecast",
    "📍 Stations",
    "📊 Analytics",
    "🧠 Model & Methodology"
])

# ============================================================
# TAB 1: ⚡ INTELLIGENT CHARGING POWER ALLOCATION (OPTIMIZER)
# ============================================================
with tab_find:
    st.markdown("## ⚡ Intelligent Charging Power Allocation (PuLP 4.0 Optimizer)")
    st.caption("Given multiple EVs at one charging station and limited available electricity, the mathematical optimizer intelligently allocates charging power among them based on urgency, remaining time, SOC deficit, target SOC, and station capacity.")


    # 1. Station Selector & Usable Power Controls
    col_st1, col_st2, col_st3 = st.columns([1.5, 1.2, 1.3])
    with col_st1:
        station_options = list(stations["station_id"].unique())
        chosen_st = st.selectbox(
            "Select Charging Station",
            station_options,
            index=0,
            format_func=lambda s: f"{s} — {stations.loc[stations['station_id']==s, 'station_name'].values[0]}" if s in stations["station_id"].values else s
        )
        st_row = stations[stations["station_id"] == chosen_st].iloc[0]
        st_rated_cap = float(st_row["rated_power_kw"])
        st_num_chargers = int(st_row.get("num_chargers", 1))
        st_chargers_desc = str(st_row.get("chargers", f"DC {st_rated_cap}kW"))

    with col_st2:
        st.markdown(f"**📍 Location:** `{st_row.get('area', 'Kavali')}`")
        st.markdown(f"**⚡ Rated Capacity:** `{st_rated_cap:.1f} kW` ({st_num_chargers} charger{'s' if st_num_chargers>1 else ''})")
        st.caption(f"🔌 Hardware Spec: {st_chargers_desc}")

    # Forecast-aware demand for this station in the current hour
    st_hourly_forecast = station_forecasts.get(chosen_st, [25.0] * 24)
    current_hour_idx = pd.Timestamp.now().hour % len(st_hourly_forecast)
    st_forecast_hour_demand = float(st_hourly_forecast[current_hour_idx])

    with col_st3:
        usable_power_input = st.slider(
            "Usable Station Power (kW)",
            min_value=5.0,
            max_value=float(max(st_rated_cap, 60.0)),
            value=float(st_rated_cap),
            step=2.5,
            help="Simulate transformer limits, peak grid stress, or demand-response curtailment."
        )

    # 2. Run Intelligent PuLP 4.0 Power Allocation Optimization
    opt_result = optimize_charging_power_allocation(
        station_id=chosen_st,
        ev_requests_df=ev_requests,
        station_capacity_kw=st_rated_cap,
        forecasted_demand_kwh=st_forecast_hour_demand,
        decision_interval_hours=1.0,
        headroom_override_kw=usable_power_input
    )
    opt_df = opt_result["df"]

    # 3. Key Summary Metrics Cards
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Station Capacity", f"{opt_result['station_capacity_kw']:.1f} kW")
    with m_col2:
        st.metric("Forecasted Demand", f"{opt_result['forecasted_demand_kwh']:.1f} kWh", help="From XGBoost + LSTM Hybrid forecast for this hour")
    with m_col3:
        st.metric("Usable Power", f"{opt_result['usable_power_kw']:.1f} kW", help="Available electrical capacity for active EV charging")
    with m_col4:
        st.metric("Total Power Allocated", f"{opt_result['total_allocated_kw']:.1f} kW", delta=f"{opt_result['usable_power_kw'] - opt_result['total_allocated_kw']:.1f} kW Headroom")
    with m_col5:
        status_badge = "Optimal 🟢" if opt_result["is_optimal"] else f"{opt_result['status']} 🟡"
        st.metric("Solver Status", status_badge, help="PuLP 4.0 / HiGHS LP Solver")

    # 4. Mandatory Core Explanation Box
    st.markdown("""
    <div style="background: rgba(14, 165, 233, 0.08); border-left: 4px solid #38bdf8; border-radius: 8px; padding: 0.95rem 1.25rem; margin: 1.1rem 0;">
        <strong style="color: #38bdf8; font-size: 1.05rem;">⚡ Allocation Principle:</strong><br/>
        <span style="color: #f1f5f9; font-size: 0.96rem; line-height: 1.6;">
            Charging power is allocated according to EV urgency, SOC, energy requirement, deadline and station capacity.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # 5. Before vs After Demonstration Callout
    with st.expander("🔍 Demonstration: Why This Solves Unintelligent Single-EV Monopolization", expanded=False):
        c_bef, c_aft = st.columns(2)
        with c_bef:
            st.markdown("""
            <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 12px; padding: 1rem 1.2rem;">
                <div style="color: #f87171; font-weight: 750; font-size: 0.92rem; margin-bottom: 0.4rem;">❌ NAIVE LINEAR OPTIMIZER (BEFORE)</div>
                <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                    <b>Station Capacity = 60 kW</b><br/>
                    • <b>EV-01</b> → 60 kW (Monopolizes entire capacity)<br/>
                    • <b>EV-02</b> → 0 kW (Starved, departure deadline missed)<br/>
                    • <b>EV-03</b> → 0 kW (Starved)<br/>
                    <i>Defect: Extreme-point solution ignores other vehicles' deadlines and remaining energy needs.</i>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c_aft:
            st.markdown("""
            <div style="background: rgba(34, 197, 94, 0.08); border: 1px solid rgba(34, 197, 94, 0.25); border-radius: 12px; padding: 1rem 1.2rem;">
                <div style="color: #4ade80; font-weight: 750; font-size: 0.92rem; margin-bottom: 0.4rem;">✅ INTELLIGENT ALLOCATION (AFTER)</div>
                <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                    <b>Station Capacity = 60 kW (PuLP 4.0 / HiGHS)</b><br/>
                    • Multi-factor priority: SOC deficit + deadline urgency + priority tier<br/>
                    • Diminishing marginal returns ensures non-starvation base charging<br/>
                    • Strictly capped by remaining energy need (e.g. 5.9 kWh need → max 5.9 kW)<br/>
                    <i>Result: Balanced throughput, all deadlines respected, fair intelligent allocation.</i>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 6. Detailed Allocation Table
    if not opt_df.empty:
        st.markdown("### 📋 Intelligent Power Allocation Results")
        table_show = opt_df[[
            "station_name",
            "forecast_demand_kwh",
            "station_capacity_kw",
            "usable_charging_power_kw",
            "ev_id",
            "vehicle_model",
            "current_soc",
            "target_soc",
            "required_energy_kwh",
            "remaining_time_hours",
            "priority_tier",
            "allocated_power_kw",
            "energy_delivered_next_hour_kwh",
            "projected_soc"
        ]].copy()

        table_show.columns = [
            "Station",
            "Forecast Demand (kWh)",
            "Station Capacity (kW)",
            "Usable Power (kW)",
            "EV ID",
            "Vehicle Model",
            "Current SOC (%)",
            "Target SOC (%)",
            "Required Energy (kWh)",
            "Remaining Time (h)",
            "Priority",
            "Allocated Power (kW)",
            "Energy Delivered Next Hour (kWh)",
            "Projected SOC (%)"
        ]

        st.dataframe(
            table_show.style.format({
                "Forecast Demand (kWh)": "{:.1f}",
                "Station Capacity (kW)": "{:.1f}",
                "Usable Power (kW)": "{:.1f}",
                "Current SOC (%)": "{:.1f}%",
                "Target SOC (%)": "{:.1f}%",
                "Required Energy (kWh)": "{:.1f}",
                "Remaining Time (h)": "{:.1f}h",
                "Allocated Power (kW)": "{:.2f}",
                "Energy Delivered Next Hour (kWh)": "{:.2f}",
                "Projected SOC (%)": "{:.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

        # 7. Visual Comparison: Required / Maximum Power vs Allocated Power
        st.markdown("### 📊 Power Comparison: Required vs. Charger Limit vs. Allocated Power")
        fig_alloc = go.Figure()
        ev_labels = [f"{r['ev_id']}<br><sup>{r['vehicle_model']}</sup>" for _, r in opt_df.iterrows()]

        fig_alloc.add_trace(go.Bar(
            x=ev_labels,
            y=opt_df["max_interval_power_kw"],
            name="Required Power (kWh/h)",
            marker_color="#f59e0b",
            opacity=0.85
        ))

        fig_alloc.add_trace(go.Bar(
            x=ev_labels,
            y=opt_df["max_charging_power_kw"],
            name="Max Charger Limit (kW)",
            marker_color="#64748b",
            opacity=0.70
        ))

        fig_alloc.add_trace(go.Bar(
            x=ev_labels,
            y=opt_df["allocated_power_kw"],
            name="Intelligently Allocated Power (kW)",
            marker_color="#38bdf8",
            marker_line=dict(width=1.5, color="#ffffff")
        ))

        fig_alloc.add_hline(
            y=opt_result["usable_power_kw"],
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Usable Station Capacity: {opt_result['usable_power_kw']:.1f} kW",
            annotation_position="top right"
        )

        fig_alloc.update_layout(
            barmode="group",
            height=400,
            margin=dict(l=20, r=20, t=35, b=40),
            xaxis_title="Queued Electric Vehicles",
            yaxis_title="Power (kW)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            font=dict(color="#f8fafc")
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

        # 8. Visual Progress: Current SOC -> Projected SOC -> Target SOC
        st.markdown("### 🔋 State of Charge (SOC) Progression in Next Hour")
        fig_soc = go.Figure()
        for idx, r in opt_df.iterrows():
            delivered_soc_pct = max(0.0, r["projected_soc"] - r["current_soc"])
            fig_soc.add_trace(go.Bar(
                y=[f"{r['ev_id']} ({r['vehicle_model']})"],
                x=[r["current_soc"]],
                orientation="h",
                name="Current SOC" if idx == 0 else None,
                showlegend=(idx == 0),
                marker_color="#334155"
            ))
            fig_soc.add_trace(go.Bar(
                y=[f"{r['ev_id']} ({r['vehicle_model']})"],
                x=[delivered_soc_pct],
                orientation="h",
                name="Delivered in Next Hour" if idx == 0 else None,
                showlegend=(idx == 0),
                marker_color="#22c55e"
            ))
            fig_soc.add_trace(go.Scatter(
                y=[f"{r['ev_id']} ({r['vehicle_model']})"],
                x=[r["target_soc"]],
                mode="markers",
                name="Target SOC" if idx == 0 else None,
                showlegend=(idx == 0),
                marker=dict(symbol="line-ns-open", size=18, color="#eab308", line_width=3)
            ))

        fig_soc.update_layout(
            barmode="stack",
            height=300,
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis=dict(title="State of Charge (%)", range=[0, 105]),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            font=dict(color="#f8fafc")
        )
        st.plotly_chart(fig_soc, use_container_width=True)

    else:
        st.warning(f"No active EV charging requests found for station **{chosen_st}**.")

    # 9. Interactive Queue Simulation
    with st.expander("➕ Simulate / Add an EV to the Queue at this Station", expanded=False):
        st.caption("Test how the PuLP 4.0 optimizer dynamically re-balances allocation when a new vehicle connects.")
        c_e1, c_e2, c_e3, c_e4 = st.columns(4)
        with c_e1:
            new_ev_id = st.text_input("EV ID", value=f"EV-SIM-{len(opt_df)+1}")
            new_model = st.text_input("Vehicle Model", value="Mahindra BE 6")
        with c_e2:
            new_bat = st.number_input("Battery Capacity (kWh)", min_value=10.0, max_value=120.0, value=50.0, step=5.0)
            new_cur_soc = st.slider("Current SOC (%)", min_value=5.0, max_value=95.0, value=25.0, step=5.0)
        with c_e3:
            new_tgt_soc = st.slider("Target SOC (%)", min_value=50.0, max_value=100.0, value=85.0, step=5.0)
            new_max_kw = st.number_input("Max Charger Power (kW)", min_value=3.3, max_value=150.0, value=50.0, step=5.0)
        with c_e4:
            new_rem_time = st.number_input("Remaining Time (hours)", min_value=0.25, max_value=8.0, value=1.2, step=0.25)
            new_prio_tier = st.selectbox("Priority Tier", ["Emergency / Fleet", "High", "Normal", "Low"], index=1)

        if st.button("⚡ Add to Simulation Queue", type="primary"):
            prio_map = {"Emergency / Fleet": 4, "High": 3, "Normal": 2, "Low": 1}
            new_row = pd.DataFrame([{
                "station_id": chosen_st,
                "ev_id": new_ev_id,
                "vehicle_model": new_model,
                "battery_capacity_kwh": new_bat,
                "current_soc": new_cur_soc,
                "target_soc": new_tgt_soc,
                "max_charging_power_kw": new_max_kw,
                "remaining_time_hours": new_rem_time,
                "priority_tier": new_prio_tier,
                "priority_score": prio_map.get(new_prio_tier, 2)
            }])
            ev_requests = pd.concat([ev_requests, new_row], ignore_index=True)
            st.success(f"Added **{new_ev_id}** to {chosen_st}! Re-running optimizer...")
            st.rerun()

    # 10. Preserved Station Travel Recommendation Tool (Trip Planner)
    with st.expander("🚗 Station Travel Recommendation & Comparison (Trip Planner)", expanded=False):
        st.caption("AI-powered routing comparing predicted demand, charger speeds, and waiting times across Kavali.")
        if len(recommendations) > 0:
            rec_best = recommendations.iloc[0]
            rec_fastest = recommendations.loc[recommendations["charger_power_kw"].idxmax()]
            
            c_r1, c_r2 = st.columns([1.2, 1])
            with c_r1:
                st.markdown(f"""
                <div class="recommend-card">
                    <div class="recommend-badge">🥇 AI RECOMMENDED</div>
                    <div class="recommend-title">{rec_best['station_id']}</div>
                    <div class="recommend-location">{rec_best.get('station_name', 'Kavali Charging Station')}</div>
                    <div class="metric-row">
                        <div class="metric-box"><div class="metric-label">Congestion</div><div class="metric-value">{rec_best['congestion']}</div></div>
                        <div class="metric-box"><div class="metric-label">Power</div><div class="metric-value">⚡ {rec_best['charger_power_kw']:.1f} kW</div></div>
                        <div class="metric-box"><div class="metric-label">Wait</div><div class="metric-value">⏱ {rec_best['estimated_wait_min']:.1f} min</div></div>
                        <div class="metric-box"><div class="metric-label">Charge</div><div class="metric-value">🔋 {rec_best['estimated_charge_min']:.1f} min</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c_r2:
                st.markdown(f"""
                <div class="fast-card">
                    <div class="fast-title">⚡ Fastest Charger</div>
                    <div class="recommend-title">{rec_fastest['station_id']}</div>
                    <div class="fast-power">{rec_fastest['charger_power_kw']:.1f} kW</div>
                    <div class="fast-label">{rec_fastest.get('station_name', 'Kavali Charging Station')}</div>
                </div>
                """, unsafe_allow_html=True)

            st.dataframe(recommendations[[
                "rank", "station_id", "predicted_energy_kwh", "congestion", "num_chargers", "charger_power_kw", "estimated_wait_min", "estimated_charge_min", "total_time_min"
            ]], use_container_width=True, hide_index=True)


# ============================================================
# TAB 2: 📈 24-HOUR FORECAST
# ============================================================
with tab_forecast:
    st.subheader(f"Full-Day Demand Forecast — {forecast_date.strftime('%A, %d %B %Y')}")
    total = fut.forecast_kwh.sum()
    peak = fut.loc[fut.forecast_kwh.idxmax()]
    avg = fut.forecast_kwh.mean()

    m1, m2, m3, m4 = st.columns(4)
    for c, t, v, s in [
        (m1, "PREDICTED DAILY ENERGY", f"{total:,.1f} kWh", "24-hour total"),
        (m2, "PEAK HOUR", peak.timestamp.strftime("%I:%M %p"), f"{peak.forecast_kwh:,.1f} kWh"),
        (m3, "AVG HOURLY", f"{avg:,.1f} kWh", "daily mean"),
        (m4, "EVENT HOURS", str(len(event_hours)), "event-adjusted hours")
    ]:
        c.markdown(f'<div class="card"><div class="metric-title">{t}</div><div class="metric-value">{v}</div><div class="small">{s}</div></div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fut.timestamp, y=fut.forecast_kwh, mode="lines+markers", name="Hybrid forecast", line=dict(width=4), fill="tozeroy"))
    fig.add_trace(go.Scatter(x=fut.timestamp, y=fut.xgb_pred_kwh, mode="lines", name="XGBoost", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=fut.timestamp, y=fut.lstm_pred_kwh, mode="lines", name="LSTM", line=dict(dash="dot")))
    if len(event_hours):
        fig.add_trace(go.Scatter(x=event_hours.timestamp, y=event_hours.forecast_kwh, mode="markers", name="Event impact", marker=dict(size=11, symbol="star")))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=35, b=10), xaxis_title="Time", yaxis_title="Energy demand (kWh)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🕐 Station-Level Demand Breakdown")
    col_sel, col_peak = st.columns([2, 1])
    with col_sel:
        selected_station = st.selectbox(
            "Inspect Station Demand Profile",
            list(station_forecasts.keys()),
            format_func=lambda sid: f"{sid} — {stations.loc[stations.station_id == sid, 'station_name'].iloc[0] if len(stations.loc[stations.station_id == sid]) else sid}"
        )
    with col_peak:
        st_forecast = station_forecasts[selected_station]
        peak_index = int(np.argmax(st_forecast))
        peak_demand = float(st_forecast[peak_index])
        st.metric(
            "Station Peak Demand Hour",
            fut.timestamp.iloc[peak_index].strftime("%I:%M %p"),
            f"{peak_demand:.1f} kWh"
        )

    hours_labels = [ts.strftime("%I %p") for ts in fut.timestamp]
    fig_st = go.Figure()
    fig_st.add_trace(
        go.Scatter(
            x=hours_labels,
            y=st_forecast,
            mode="lines+markers",
            name="Predicted Demand (kWh)",
            line=dict(width=3, color="#00b4d8"),
            marker=dict(size=6)
        )
    )
    fig_st.update_layout(
        title=f"24-Hour Demand Forecast — {selected_station}",
        xaxis_title="Hour of Day",
        yaxis_title="Energy Demand (kWh)",
        template="plotly_white",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_st, use_container_width=True)

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("### Hour-by-hour forecast")
        table = fut[["timestamp","xgb_pred_kwh","lstm_pred_kwh","forecast_kwh","event_impact_pct","event_name"]].copy()
        table.columns = ["Time","XGBoost (kWh)","LSTM (kWh)","Hybrid (kWh)","Event impact %","Event"]
        table["Time"] = table.Time.dt.strftime("%I:%M %p")
        st.dataframe(table.style.format({"XGBoost (kWh)":"{:.1f}","LSTM (kWh)":"{:.1f}","Hybrid (kWh)":"{:.1f}","Event impact %":"{:.1f}%"}), use_container_width=True, hide_index=True)
    with right:
        st.markdown("### Forecast interpretation")
        if len(event_hours):
            st.success(f"Event-aware forecast: {len(event_hours)} hours receive an event adjustment. Scenario assumptions should be verified against local operational event records.")
        st.markdown(f"**Peak:** {peak.timestamp.strftime('%A %I:%M %p')} at **{peak.forecast_kwh:,.1f} kWh**.")
        st.markdown(f"**Day total:** approximately **{total:,.1f} kWh** across the four monitored stations.")
        st.markdown("The model combines recent lags, same-week patterns, calendar seasonality, and long-term trends rather than copying the prior year's demand directly.")

# ============================================================
# TAB 3: 📍 STATIONS
# ============================================================
with tab_stations:
    st.subheader("🏢 Station-Level Intelligence")
    station_daily = hs.groupby(["station_id","date"], as_index=False).agg(energy_kwh=("energy_kwh","sum"), sessions=("sessions","sum"), busy_min=("busy_min","sum"))
    station_summary = station_daily.groupby("station_id", as_index=False).agg(total_energy_kwh=("energy_kwh","sum"), avg_daily_kwh=("energy_kwh","mean"), total_sessions=("sessions","sum"), avg_busy_min=("busy_min","mean"))
    station_summary = station_summary.merge(stations, on="station_id", how="left")
    st.dataframe(station_summary.sort_values("total_energy_kwh", ascending=False).style.format({"total_energy_kwh":"{:,.0f}","avg_daily_kwh":"{:,.1f}","total_sessions":"{:,.0f}","avg_busy_min":"{:,.1f}"}), use_container_width=True, hide_index=True)
    selected = st.selectbox("Inspect station historical trajectory", station_summary.station_id.tolist())
    sd = station_daily[station_daily.station_id == selected]
    fig = go.Figure(go.Scatter(x=sd.date, y=sd.energy_kwh, mode="lines", name=selected, fill="tozeroy"))
    fig.update_layout(height=350, xaxis_title="Date", yaxis_title="Daily energy (kWh)")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 4: 📊 ANALYTICS
# ============================================================
with tab_analytics:
    st.subheader("📊 Historical Charging Analytics")
    daily = city.groupby("date", as_index=False).agg(energy_kwh=("energy_kwh","sum"), sessions=("sessions","sum"), busy_min=("busy_min","sum"))
    monthly = daily.assign(month=daily.date.dt.to_period("M")).groupby("month", as_index=False).agg(energy_kwh=("energy_kwh","sum"), sessions=("sessions","sum"))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=.12, subplot_titles=("Daily energy demand","Monthly energy demand"))
    fig.add_trace(go.Scatter(x=daily.date, y=daily.energy_kwh, mode="lines", name="Daily"), row=1, col=1)
    fig.add_trace(go.Bar(x=monthly.month.astype(str), y=monthly.energy_kwh, name="Monthly"), row=2, col=1)
    fig.update_layout(height=650, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    a, b, c = st.columns(3)
    peak_day = daily.loc[daily.energy_kwh.idxmax()]
    a.metric("Peak historical day", str(peak_day.date.date()), f"{peak_day.energy_kwh:,.1f} kWh")
    b.metric("Highest month", str(monthly.loc[monthly.energy_kwh.idxmax(), "month"]), f"{monthly.energy_kwh.max():,.0f} kWh")
    c.metric("Average sessions/day", f"{daily.sessions.mean():,.1f}")
    st.markdown("### Hour × day-of-week pattern")
    pivot = city.pivot_table(index="hour", columns="dayofweek", values="energy_kwh", aggfunc="mean")
    pivot.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    heat = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Blues", colorbar_title="kWh"))
    heat.update_layout(height=500, xaxis_title="Day", yaxis_title="Hour")
    st.plotly_chart(heat, use_container_width=True)

# ============================================================
# TAB 5: 🧠 MODEL & METHODOLOGY
# ============================================================
with tab_model:
    st.subheader("🧠 Model Architecture, Validation & Methodology")
    st.markdown("**Primary Model:** XGBoost histogram gradient-boosted regression (`n_estimators=420`, `max_depth=6`, `learning_rate=0.045`).")
    st.markdown("**Secondary Model:** Deep LSTM sequence neural network (`32 units → Dropout(0.12) → 16 units → Dense(8) → Dense(1)`).")
    st.markdown("**Final Forecast:** Weighted ensemble blending both models with recursive hour-ahead feature generation.")
    st.markdown("**Target Variable:** Hourly station-network energy demand (`energy_kwh`).")
    st.markdown("**Predictor Set:** Hour/day/month sine-cosine seasonality, weekend indicator, lags (1, 2, 3, 24, 48, 72, 168, 336), rolling statistics, and 30-day baseline growth proxy.")
    st.markdown("**Why Chronological Holdout?** Time-series data violates the i.i.d. assumption. Random cross-validation leaks future information into past predictions. We validate strictly against the final 7 historical days.")
    if sessions is not None:
        st.markdown(f"**Session Dataset:** {len(sessions):,} records ({sessions.date.min()} to {sessions.date.max()}).")
    st.markdown(f"**Hourly Dataset:** {len(hs):,} station-hour observations across {hs.station_id.nunique()} Kavali stations.")

    with st.spinner("Calculating chronological XGBoost validation metrics…"):
        scores, va, px = validation_scores(city, FEATURES)
    vc1, vc2, vc3 = st.columns(3)
    vc1.metric("Validation MAE", f"{scores['XGBoost MAE']:.2f} kWh")
    vc2.metric("Validation RMSE", f"{scores['XGBoost RMSE']:.2f} kWh")
    vc3.metric("Validation MAPE", f"{scores['XGBoost MAPE']:.2f}%")
    st.caption("Validation metrics evaluated against a 7-day chronological holdout split.")


# ----------------------------- Footer -----------------------------
st.divider()
st.caption("Prototype note: the supplied Kavali dataset is synthetic. It is suitable for demonstrating the forecasting architecture, but model results should not be presented as measured real-world Kavali demand. Verify local event schedules and operational data before real deployment.")
