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
    "🚗 Find Station",
    "📈 24-Hour Forecast",
    "📍 Stations",
    "📊 Analytics",
    "🧠 Model & Methodology"
])

# ============================================================
# TAB 1: 🚗 FIND STATION (User Decision Hub)
# ============================================================
with tab_find:
    st.markdown("## 🔌 Find Your Best Charging Station")
    st.caption("AI-powered routing comparing predicted demand, charger speeds, and waiting times across Kavali.")

    col_inp1, col_inp2 = st.columns([1, 1.6])
    with col_inp1:
        st.markdown(f"**⚡ Target Energy:** `{required_energy:.0f} kWh` &nbsp;•&nbsp; **📅 Date:** `{forecast_date.strftime('%d %b %Y')}`")
    with col_inp2:
        if len(recommendations) > 0:
            best_st = recommendations.iloc[0]
            worst_st = recommendations.iloc[-1]
            st.info(
                f"💡 **AI Planning Tip**: **{best_st['station_id']}** currently has the lowest predicted turnaround ({best_st['total_time_min']:.0f} min). "
                f"**{worst_st['station_id']}** is expected to experience {worst_st['congestion']} congestion ({worst_st['total_time_min']:.0f} min). Plan ahead to avoid queues."
            )

    if len(event_hours):
        avg_event_impact = event_hours.event_impact_pct.mean()
        if avg_event_impact != 0:
            st.warning(f"🎉 **Scenario Event Active**: {avg_event_impact:+.0f}% average demand shift factored into station utilization.")

    # 1. 🥇 AI RECOMMENDED & ⚡ FASTEST CHARGER DUAL CARDS
    if len(recommendations) > 0:
        best = recommendations.iloc[0]
        fastest = recommendations.loc[recommendations["charger_power_kw"].idxmax()]
        
        col_rec, col_fast = st.columns([1.2, 1])
        with col_rec:
            st.markdown(
                f"""
                <div class="recommend-card">
                    <div class="recommend-badge">
                        🥇 AI RECOMMENDED
                    </div>
                    <div class="recommend-title">
                        {best['station_id']}
                    </div>
                    <div class="recommend-location">
                        {best.get('station_name', 'Kavali Charging Station')}
                    </div>
                    <div class="metric-row">
                        <div class="metric-box">
                            <div class="metric-label">Congestion</div>
                            <div class="metric-value">
                                {best['congestion']}
                            </div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Charger</div>
                            <div class="metric-value">
                                ⚡ {best['charger_power_kw']:.1f} kW
                            </div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Wait</div>
                            <div class="metric-value">
                                ⏱ {best['estimated_wait_min']:.1f} min
                            </div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Charging</div>
                            <div class="metric-value">
                                🔋 {best['estimated_charge_min']:.1f} min
                            </div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Total Time</div>
                            <div class="metric-value" style="color:#38bdf8;">
                                {best['total_time_min']:.0f} min
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_fast:
            st.markdown(
                f"""
                <div class="fast-card">
                    <div class="fast-title">
                        ⚡ Fastest Charger
                    </div>
                    <div class="recommend-title">
                        {fastest['station_id']}
                    </div>
                    <div class="fast-power">
                        {fastest['charger_power_kw']:.1f} kW
                    </div>
                    <div class="fast-label">
                        Charger rated power &bull; {fastest.get('station_name', 'Kavali Charging Station')}
                    </div>
                    <div class="metric-row">
                        <div class="metric-box">
                            <div class="metric-label">Wait</div>
                            <div class="metric-value">
                                ⏱ {fastest['estimated_wait_min']:.1f} min
                            </div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Charging</div>
                            <div class="metric-value">
                                🔋 {fastest['estimated_charge_min']:.1f} min
                            </div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Congestion</div>
                            <div class="metric-value">
                                {fastest['congestion']}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # 2. 💡 WHY THIS STATION?
        st.markdown(f"""
        <div class="why-list">
            <div style="font-weight:750; font-size:1.05rem; margin-bottom:0.4rem; color:#4ade80;">
                Why {best['station_id']}? ✓
            </div>
            <div style="font-size:0.92rem; line-height:1.8; color:#dcfce7;">
                {best['congestion']} predicted congestion<br/>
                ⚡ Highest charger power ({best['charger_power_kw']:.0f} kW)<br/>
                ⏱ Short estimated waiting time ({best['estimated_wait_min']:.1f} min)<br/>
                🔋 Short estimated charging duration ({best['estimated_charge_min']:.1f} min)
            </div>
            <div style="margin-top:0.6rem; font-size:0.82rem; color:#94a3b8; border-top:1px dashed rgba(52,211,153,0.3); padding-top:0.45rem;">
                <strong>AI recommendation based on:</strong><br/>
                Demand forecast + congestion + charger power + estimated wait
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 3. 📍 COMPARE 4 CHARGING STATIONS (Cards)
    st.markdown(
        '<div class="ev-section-title">📍 Compare Charging Stations</div>',
        unsafe_allow_html=True
    )
    cols = st.columns(2)
    for i, (_, station) in enumerate(recommendations.iterrows()):
        congestion = str(station["congestion"])
        if "Low" in congestion:
            status_class = "status-low"
        elif "Medium" in congestion:
            status_class = "status-medium"
        else:
            status_class = "status-high"
        with cols[i % 2]:
            rank_str = str(station['rank']).replace("#", "")
            st.markdown(
                f"""
                <div class="station-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div class="station-rank">
                            RANK #{rank_str}
                        </div>
                        <div class="station-status {status_class}">
                            {congestion}
                        </div>
                    </div>
                    <div class="station-name">
                        {station['station_id']}
                    </div>
                    <div style="font-size:0.86rem; color:#94a3b8; margin:0.2rem 0 0.5rem;">
                        {station['station_name']} ({station['area']})
                    </div>
                    <div class="metric-row">
                        <div class="metric-box">
                            <div class="metric-label">Power</div>
                            <div class="metric-value">⚡ {station['charger_power_kw']:.1f} kW</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Wait</div>
                            <div class="metric-value">⏱ {station['estimated_wait_min']:.1f} min</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Charge</div>
                            <div class="metric-value">🔋 {station['estimated_charge_min']:.1f} min</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Total</div>
                            <div class="metric-value" style="color:#38bdf8; font-weight:750;">{station['total_time_min']:.0f} min</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 4. 📋 DETAILED STATION DATA EXPANDER
    with st.expander("📋 View detailed station data", expanded=False):
        display_df = recommendations[
            [
                "rank",
                "station_id",
                "predicted_energy_kwh",
                "congestion",
                "num_chargers",
                "charger_power_kw",
                "estimated_wait_min",
                "estimated_charge_min",
                "total_time_min"
            ]
        ].copy()
        display_df.columns = [
            "Rank",
            "Station",
            "Predicted Demand (kWh)",
            "Congestion",
            "Chargers",
            "Power (kW)",
            "Wait (min)",
            "Charge (min)",
            "Total Time (min)"
        ]
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    # 5. 📊 CURRENT PREDICTED CONGESTION (Visual Meter & Progress Cards)
    st.markdown("### 📊 Current Predicted Status")
    meter_html = '<div style="background:#0f172a; color:#f8fafc; padding:1.1rem 1.4rem; border-radius:14px; font-family:monospace; font-size:0.98rem; margin-bottom:1.1rem; box-shadow:0 4px 15px rgba(0,0,0,0.08);">'
    meter_html += '<div style="color:#94a3b8; font-weight:750; font-size:0.8rem; letter-spacing:0.08em; margin-bottom:0.75rem;">CURRENT PREDICTED STATUS</div>'
    for _, r_st in recommendations.iterrows():
        u = r_st['predicted_utilization']
        filled = max(1, min(10, int(round(u * 10))))
        blocks = "█" * filled + "░" * (10 - filled)
        color = "#10b981" if u < 0.35 else ("#f59e0b" if u < 0.70 else "#ef4444")
        lvl_word = "Low" if u < 0.35 else ("Medium" if u < 0.70 else "High")
        meter_html += f'<div style="display:flex; justify-content:space-between; margin-bottom:0.4rem; align-items:center;">'
        meter_html += f'<span><strong>{r_st["station_id"]}</strong> &nbsp; <span style="color:{color}; letter-spacing:2px;">{blocks}</span></span>'
        meter_html += f'<span style="color:{color}; font-weight:750;">{r_st["congestion"]}</span>'
        meter_html += '</div>'
    meter_html += '</div>'
    st.markdown(meter_html, unsafe_allow_html=True)

    # 6. 🕒 NEXT 3 HOURS LOOKAHEAD (Traffic Light Grid)
    st.markdown("### 🕒 Next 3 Hours Lookahead")
    next3_rows = []
    warning_stations = []
    for sid in stations["station_id"]:
        st_row = stations[stations["station_id"] == sid].iloc[0]
        pwr = extract_station_power(st_row)
        nc = int(st_row.get("num_chargers", 1))
        cap = max(pwr * nc, 1.0)
        fc = station_forecasts[sid]
        u0 = np.clip(fc[0] / cap, 0.05, 0.98)
        u1 = np.clip(fc[1] / cap, 0.05, 0.98)
        u2 = np.clip(fc[2] / cap, 0.05, 0.98)
        c0 = get_congestion_level(u0)
        c1 = get_congestion_level(u1)
        c2 = get_congestion_level(u2)
        if ("Low" in c0 and ("Medium" in c2 or "High" in c2)) or ("Medium" in c0 and "High" in c2):
            warning_stations.append(sid)
        next3_rows.append({
            "Station ID": sid,
            "Station Name": st_row["station_name"],
            "Now": c0,
            f"+1h ({fut.timestamp.iloc[1].strftime('%I %p')})": c1,
            f"+2h ({fut.timestamp.iloc[2].strftime('%I %p')})": c2
        })
    st.dataframe(pd.DataFrame(next3_rows), use_container_width=True, hide_index=True)
    if warning_stations:
        st.warning(f"⚠️ **Congestion Alert**: {', '.join(warning_stations)} is expected to experience increasing congestion over the next 2 hours.")

    # 7. 📈 24-HOUR FORECAST PREVIEW EXPANDER
    with st.expander("📈 Quick Preview: 24-Hour City Demand Forecast Curve", expanded=False):
        fig_mini = go.Figure()
        fig_mini.add_trace(go.Scatter(x=fut.timestamp, y=fut.forecast_kwh, mode="lines+markers", name="Hybrid Forecast", line=dict(width=3, color="#00b4d8"), fill="tozeroy"))
        fig_mini.update_layout(height=300, margin=dict(l=10, r=10, t=25, b=10), xaxis_title="Hour", yaxis_title="Demand (kWh)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_mini, use_container_width=True)
        st.caption("👉 For complete model evaluation, drilldowns, and tables, switch to the **📈 24-Hour Forecast** tab above.")

    st.caption("ℹ️ *Notice: Predicted Congestion and Estimated Waiting Time are calculated using hybrid demand forecasts and queuing approximations on the Kavali dataset. They serve as planning intelligence rather than real-time hardware queue telemetry.*")

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
