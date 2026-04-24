import base64
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path
 
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components
 
# =========================
# CONFIG
# =========================
API_BASE = "https://satpi-backend.onrender.com"
 
MAX_HISTORIAL         = 300
REFRESH_SECONDS       = 2        # interval base (s) — pujat de 1→2 per reduir càrrega
REFRESH_ERROR_SECONDS = 4        # interval si l'API falla
API_TIMEOUT           = 6        # timeout petició HTTP (s)
API_MAX_RETRIES       = 2        # reintents en cas de fallada de xarxa
GRAF_MAX_PUNTS        = 100      # submostreig gràfiques (menys punts = molt més ràpid)
 
SMOOTH_WINDOW         = 5
ASCENS_CONFIRM_POINTS = 4
ASCENS_THRESHOLD      = 0.8
ASCENS_GAIN_MIN       = 3.0
 
FASE_WINDOW           = 10
FASE_V_UP             = 0.8
FASE_V_DOWN           = -0.8
FASE_LAND_V_ABS       = 0.25
FASE_LAND_ALTURA_MAX  = 5.0
 
MAP_HEIGHT                 = 650
MAP_ZOOM                   = 18
MAP_MOVE_THRESHOLD_METERS  = 15.0
MAP_FORCE_REFRESH_SECONDS  = 30
 
TAULA_COLUMNS = [
    "temps_txt", "temps", "lat", "lon", "alt", "alt_press", "alt_suav",
    "altura_guanyada", "altura_maxima_total", "vel", "vel_calc",
    "vel_lineal_calc", "temp", "press", "retard_s", "camX", "camY", "pc_rebut_ts",
]
 
PLOTLY_CONFIG = {"displayModeBar": False, "scrollZoom": False, "responsive": True}
 
# =========================
# UI BASE
# =========================
st.set_page_config(page_title="Estació de terra", layout="wide")
 
ASSETS_DIR = Path(__file__).parent / "assets"
if not ASSETS_DIR.exists():
    ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
 
SATPI_LOGO    = ASSETS_DIR / "satpi_logo.png"
INSTITUT_LOGO = ASSETS_DIR / "institut_logo.png"
 
 
@st.cache_data(show_spinner=False)
def _logo_b64(path_str: str):
    """Carrega i codifica el logo una sola vegada (cached)."""
    try:
        return base64.b64encode(Path(path_str).read_bytes()).decode("utf-8")
    except Exception:
        return None
 
 
st.markdown(
    """
    <style>
    :root {
        --bg-main:    #040b18;
        --bg-soft:    #091529;
        --card-bg:    rgba(10,23,43,0.92);
        --card-border:rgba(119,170,255,0.16);
        --text-main:  #f8fbff;
        --text-soft:  #aac0dc;
        --shadow:     0 8px 24px rgba(0,0,0,0.28);
    }
 
    .stApp {
        background:
            radial-gradient(circle at top left,  rgba(40,85,160,0.18), transparent 32%),
            radial-gradient(circle at top right, rgba(0,170,220,0.10), transparent 28%),
            linear-gradient(180deg, #030814 0%, #07111f 55%, #040a14 100%);
    }
 
    .block-container { padding-top:0.8rem; padding-bottom:1.3rem; max-width:none; }
 
    /* HEADER */
    .header-shell { margin-bottom:1.1rem; }
 
    .header-logo-box {
        height:122px; display:flex; align-items:center; justify-content:center;
        border-radius:22px; background:rgba(255,255,255,0.02);
        border:1px solid rgba(255,255,255,0.05); box-shadow:var(--shadow);
    }
    .header-logo-img {
        display:block; width:auto; height:auto; object-fit:contain;
        filter:drop-shadow(0 3px 8px rgba(0,0,0,0.22));
    }
    .institut-logo { max-width:220px; max-height:76px; }
    .satpi-logo    { max-width:92px;  max-height:92px; }
 
    .top-header {
        min-height:122px; display:flex; flex-direction:column; justify-content:center;
        padding:20px 30px; border-radius:24px;
        background:linear-gradient(135deg,rgba(14,29,52,0.97),rgba(4,29,54,0.97));
        border:1px solid rgba(130,180,255,0.14); box-shadow:var(--shadow);
        position:relative; overflow:hidden;
    }
    .top-header::before {
        content:""; position:absolute; inset:0; pointer-events:none;
        background:
            radial-gradient(circle at 18% 50%,rgba(86,182,255,0.10),transparent 26%),
            radial-gradient(circle at 82% 35%,rgba(124,227,200,0.07),transparent 22%);
    }
    .top-header-title    { position:relative; font-size:2.55rem; font-weight:800; line-height:1.02; letter-spacing:-0.03em; color:var(--text-main); margin-bottom:10px; }
    .top-header-subtitle { position:relative; font-size:1.02rem; color:var(--text-soft); letter-spacing:0.01em; }
 
    /* BANNERS CONNEXIÓ */
    .banner-error {
        border-radius:12px; padding:10px 18px; margin-bottom:10px;
        font-size:0.92rem; font-weight:600; display:flex; align-items:center; gap:10px;
    }
    .banner-warn  { background:rgba(251,191,36,0.12);  border:1px solid rgba(251,191,36,0.4);  color:#fbbf24; }
    .banner-crit  { background:rgba(248,113,113,0.10); border:1px solid rgba(248,113,113,0.4); color:#f87171; }
    .banner-stale { background:rgba(56,189,248,0.08);  border:1px solid rgba(56,189,248,0.3);  color:#7dd3fc; }
 
    /* INFO CARD */
    .info-card { background:#0f1724; border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:18px 20px; margin-bottom:14px; color:#f8fafc; }
    .info-card h3 { margin-top:0; margin-bottom:14px; font-size:1.35rem; color:#ffffff; }
    .info-grid    { display:grid; grid-template-columns:1fr 1fr; gap:8px 20px; }
    .info-item    { padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.05); font-size:1rem; line-height:1.35; color:#e5e7eb; }
    .info-item b  { color:#ffffff; font-weight:700; }
 
    .map-wrap { background:#0f1724; border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:10px; }
 
    @media (max-width:1100px) {
        .top-header-title    { font-size:2rem; }
        .top-header-subtitle { font-size:0.95rem; }
        .institut-logo { max-width:180px; max-height:62px; }
        .satpi-logo    { max-width:82px;  max-height:82px; }
    }
    @media (max-width:900px) {
        .top-header-title { font-size:1.55rem; }
        .header-logo-box, .top-header { min-height:100px; height:100px; }
    }
 
    /* FASE CARD */
    .fase-card {
        border-radius:20px; padding:26px 28px; margin-bottom:0;
        display:flex; flex-direction:column; gap:10px; min-height:220px;
        position:relative; overflow:hidden;
    }
    .fase-icon { font-size:2.8rem; line-height:1; }
    .fase-nom  { font-size:2.1rem; font-weight:800; letter-spacing:-0.02em; line-height:1.05; }
    .fase-desc { font-size:1.05rem; opacity:0.82; line-height:1.5; margin-top:4px; }
    .fase-retard {
        display:inline-flex; align-items:center; gap:7px; font-size:0.88rem; font-weight:600;
        border-radius:999px; padding:5px 14px; margin-top:6px; align-self:flex-start; letter-spacing:0.02em;
    }
    .retard-dot { width:8px; height:8px; border-radius:50%; display:inline-block; flex-shrink:0; }
 
    /* MOVIMENT CARD fila */
    .mov-fila {
        display:flex; align-items:center; gap:12px; padding:10px 14px;
        border-radius:12px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
def renderitzar_header():
    st.markdown('<div class="header-shell">', unsafe_allow_html=True)
    col_left, col_center, col_right = st.columns([1.25, 4.9, 1.25], gap="medium")
 
    with col_left:
        b64 = _logo_b64(str(INSTITUT_LOGO))
        if b64:
            st.markdown(
                f'<div class="header-logo-box"><img src="data:image/png;base64,{b64}" class="header-logo-img institut-logo"></div>',
                unsafe_allow_html=True,
            )
 
    with col_center:
        st.markdown(
            """<div class="top-header">
                <div class="top-header-title">Estació de terra SATPI26</div>
                <div class="top-header-subtitle">Institut Bernat el Ferrer · CanSat · Telemetria en temps real</div>
            </div>""",
            unsafe_allow_html=True,
        )
 
    with col_right:
        b64 = _logo_b64(str(SATPI_LOGO))
        if b64:
            st.markdown(
                f'<div class="header-logo-box"><img src="data:image/png;base64,{b64}" class="header-logo-img satpi-logo"></div>',
                unsafe_allow_html=True,
            )
 
    st.markdown("</div>", unsafe_allow_html=True)
 
 
renderitzar_header()
 
 
# =========================
# STATE
# =========================
def init_state():
    if "init" in st.session_state:
        return
    st.session_state.init             = True
    st.session_state.session_id       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state.historial        = deque(maxlen=MAX_HISTORIAL)
    st.session_state.altura_base      = None
    st.session_state.ha_descendit     = False
    st.session_state.last_valid_gps   = None
    st.session_state.launch_temps     = None
    # Fase
    st.session_state.fase_confirmada    = "Esperant enlairament"
    st.session_state.fase_candidata     = None
    st.session_state.fase_candidata_n   = 0
    st.session_state.last_data_wall_time = 0.0
    # Mapa
    st.session_state.map_html_cached      = ""
    st.session_state.map_last_render_time = 0.0
    st.session_state.map_lat_render       = None
    st.session_state.map_lon_render       = None
    # Connexió
    st.session_state.api_error_count    = 0
    st.session_state.api_last_ok_time   = 0.0
    st.session_state.api_last_error_msg = ""
    # Cache DataFrame
    st.session_state.df_cache     = None
    st.session_state.df_cache_len = 0
 
 
def reset_missio():
    st.session_state.historial        = deque(maxlen=MAX_HISTORIAL)
    st.session_state.altura_base      = None
    st.session_state.ha_descendit     = False
    st.session_state.last_valid_gps   = None
    st.session_state.launch_temps     = None
    st.session_state.fase_confirmada    = "Esperant enlairament"
    st.session_state.fase_candidata     = None
    st.session_state.fase_candidata_n   = 0
    st.session_state.last_data_wall_time = 0.0
    st.session_state.map_html_cached      = ""
    st.session_state.map_last_render_time = 0.0
    st.session_state.map_lat_render       = None
    st.session_state.map_lon_render       = None
    st.session_state.df_cache     = None
    st.session_state.df_cache_len = 0
 
 
init_state()
 
if not isinstance(st.session_state.historial, deque):
    st.session_state.historial = deque(st.session_state.historial, maxlen=MAX_HISTORIAL)
 
 
# =========================
# API — RESILIENT
# =========================
def _get_api(url: str):
    """GET amb reintents i backoff. Retorna el JSON o None si falla."""
    for intent in range(API_MAX_RETRIES):
        try:
            r = requests.get(url, timeout=API_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            st.session_state.api_last_error_msg = f"HTTP {r.status_code}"
            return None
        except requests.exceptions.Timeout:
            st.session_state.api_last_error_msg = "Timeout"
        except requests.exceptions.ConnectionError:
            st.session_state.api_last_error_msg = "Connexió refusada"
        except Exception as e:
            st.session_state.api_last_error_msg = str(e)[:60]
        if intent < API_MAX_RETRIES - 1:
            time.sleep(0.3 * (intent + 1))
    return None
 
 
def processar_lectura_api():
    data_raw = _get_api(f"{API_BASE}/telemetry/latest")
 
    if data_raw is None or "temps" not in data_raw:
        st.session_state.api_error_count += 1
        return
 
    st.session_state.api_error_count    = 0
    st.session_state.api_last_ok_time   = time.time()
    st.session_state.api_last_error_msg = ""
 
    try:
        data = {
            "lat":        float(data_raw["lat"]),
            "lon":        float(data_raw["lon"]),
            "alt":        float(data_raw["alt"]),
            "vel":        float(data_raw["vel"]),
            "temp":       float(data_raw["temp"]),
            "press":      float(data_raw["press"]),
            "alt_press":  float(data_raw["alt_press"]),
            "temps_txt":  str(data_raw.get("temps_txt", "")),
            "temps":      float(data_raw["temps"]),
            "camX":       str(data_raw.get("camX", "center")),
            "camY":       str(data_raw.get("camY", "center")),
            "pc_rebut_ts": float(data_raw["pc_rebut_ts"]) if data_raw.get("pc_rebut_ts") is not None else None,
        }
    except (KeyError, ValueError, TypeError):
        st.session_state.api_error_count += 1
        return
 
    if st.session_state.historial:
        ultim = st.session_state.historial[-1]["temps"]
        if data["temps"] == ultim:
            return
        if data["temps"] < ultim:
            reset_missio()
 
    retard = calcular_retard_segons(data.get("pc_rebut_ts"))
    data["retard_s"] = float(retard) if retard is not None else 0.0
    st.session_state.last_data_wall_time = time.time()
 
    if coords_valides(data["lat"], data["lon"]):
        st.session_state.last_valid_gps = {"lat": data["lat"], "lon": data["lon"], "temps": data["temps"]}
 
    st.session_state.historial.append(data)
    st.session_state.df_cache_len = 0  # invalida la caché
 
 
def calcular_retard_segons(pc_rebut_ts):
    try:
        return max(0.0, time.time() - float(pc_rebut_ts))
    except Exception:
        return None
 
 
# =========================
# BANNER CONNEXIÓ
# =========================
def renderitzar_banner_connexio():
    errors  = st.session_state.api_error_count
    last_ok = st.session_state.api_last_ok_time
    msg_err = st.session_state.api_last_error_msg
    ara     = time.time()
 
    if errors == 0:
        return
 
    # Banner de dades obsoletes (si en tenim)
    if st.session_state.historial and last_ok > 0:
        secs = ara - last_ok
        if secs > 5:
            st.markdown(
                f'<div class="banner-error banner-stale">📡 Mostrant dades de fa {secs:.0f}s — servidor sense resposta</div>',
                unsafe_allow_html=True,
            )
 
    # Banner d'error principal
    if errors <= 2:
        cls, ico, txt = "banner-warn", "⚠️", f"Reintentant connexió… ({msg_err})"
    elif errors <= 6:
        cls, ico, txt = "banner-crit", "🔴", f"Servidor lent o no disponible ({msg_err})"
    else:
        cls, ico, txt = "banner-crit", "🔴", f"Sense connexió · {errors} errors consecutius · {msg_err}"
 
    st.markdown(
        f'<div class="banner-error {cls}">{ico} {txt}</div>',
        unsafe_allow_html=True,
    )
 
 
# =========================
# GPS / DISTÀNCIA
# =========================
def coords_valides(lat, lon):
    try:
        lat, lon = float(lat), float(lon)
        return np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180
    except Exception:
        return False
 
 
def metres_per_grau(lat):
    return 111320.0, 111320.0 * math.cos(math.radians(lat))
 
 
def distancia_metres(lat1, lon1, lat2, lon2):
    if not (coords_valides(lat1, lon1) and coords_valides(lat2, lon2)):
        return 0.0
    m_lon = 111320.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return float(math.hypot((lon2 - lon1) * m_lon, (lat2 - lat1) * 111320.0))
 
 
# =========================
# CÀLCULS
# =========================
def calcular_velocitat_lineal_df(df):
    if len(df) < 2:
        return pd.Series(0.0, index=df.index)
    dt    = df["temps"].diff()
    vm_a  = df["lat"].between(-90, 90) & df["lon"].between(-180, 180)
    valid = vm_a & vm_a.shift(fill_value=False) & (dt > 0)
    m_lon = 111320.0 * np.cos(np.radians(df["lat"]))
    vel   = pd.Series(np.hypot(df["lon"].diff() * m_lon, df["lat"].diff() * 111320.0) / dt, index=df.index)
    return vel.where(valid, 0.0).replace([np.inf, -np.inf], 0).fillna(0.0)
 
 
def obtenir_df() -> pd.DataFrame:
    """Construeix (o retorna de caché) el DataFrame amb totes les variables calculades."""
    n = len(st.session_state.historial)
    if st.session_state.df_cache is not None and st.session_state.df_cache_len == n:
        return st.session_state.df_cache
 
    df = pd.DataFrame(st.session_state.historial)
    for col, default in [("camX", "center"), ("camY", "center"),
                         ("pc_rebut_ts", np.nan), ("retard_s", 0.0)]:
        if col not in df.columns:
            df[col] = default
 
    df["alt_suav"]        = df["alt"].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
    df["vel_calc"]        = (df["alt_suav"].diff() / df["temps"].diff()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["vel_lineal_calc"] = calcular_velocitat_lineal_df(df)
 
    if st.session_state.altura_base is None and len(df) >= ASCENS_CONFIRM_POINTS + 1:
        ultimes_vels   = df["vel_calc"].tail(ASCENS_CONFIRM_POINTS)
        guany_finestra = float(df["alt_suav"].iloc[-1] - df["alt_suav"].iloc[-ASCENS_CONFIRM_POINTS - 1])
        if (ultimes_vels > ASCENS_THRESHOLD).all() and guany_finestra >= ASCENS_GAIN_MIN:
            idx_ref = max(0, len(df) - ASCENS_CONFIRM_POINTS - 1)
            st.session_state.altura_base  = float(df.iloc[idx_ref]["alt_suav"])
            if st.session_state.launch_temps is None:
                st.session_state.launch_temps = float(df.iloc[idx_ref]["temps"])
 
    df["altura_guanyada"]     = (df["alt_suav"] - st.session_state.altura_base).clip(lower=0) \
                                if st.session_state.altura_base is not None else 0.0
    df["altura_maxima_total"] = df["alt"].cummax()
 
    st.session_state.df_cache     = df
    st.session_state.df_cache_len = n
    return df
 
 
def calcular_velocitat_vertical(df) -> float:
    if len(df) < 2:
        return 0.0
    recent = df.tail(min(6, len(df)))
    dt     = recent["temps"].diff()
    v      = (recent["alt_suav"].diff() / dt).replace([np.inf, -np.inf], np.nan).dropna()
    return float(v.median()) if len(v) else 0.0
 
 
def obtenir_fase_intelligent(df) -> str:
    if len(df) < 2:
        return "Esperant enlairament"
    dt    = df["temps"].diff()
    v_alt = (df["alt_suav"].diff() / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w     = min(FASE_WINDOW, len(df))
    v_mean     = float(v_alt.tail(w).mean())
    v_abs_mean = float(v_alt.tail(w).abs().mean())
    h_guanyada = float(df.iloc[-1].get("altura_guanyada", 0.0))
    vel_lineal  = float(df.iloc[-1].get("vel_lineal_calc", 0.0))
 
    if st.session_state.altura_base is None:
        if h_guanyada >= ASCENS_GAIN_MIN and v_mean >= FASE_V_UP * 0.6:
            return "Ascens"
        return "Esperant enlairament"
    if v_mean >= FASE_V_UP:
        return "Ascens"
    if v_mean <= FASE_V_DOWN:
        st.session_state.ha_descendit = True
        return "Descens"
    if st.session_state.ha_descendit:
        if h_guanyada <= FASE_LAND_ALTURA_MAX and v_abs_mean <= FASE_LAND_V_ABS and vel_lineal <= 0.5:
            return "Aterrat"
        return "Vol actiu"
    return "Vol actiu"
 
 
def calcular_moviment_i_velocitat_lineal(df) -> dict:
    base = {"mov_x": "X estable", "mov_y": "Y estable", "mov_z": "Altitud estable",
            "vel_lineal": 0.0, "direccio": "sense moviment"}
    if len(df) < 2:
        return base
    a, b  = df.iloc[-1], df.iloc[-2]
    dt    = a["temps"] - b["temps"]
    if dt <= 0:
        return base
    d_lon = a["lon"] - b["lon"]
    d_lat = a["lat"] - b["lat"]
    d_alt = a["alt_suav"] - b["alt_suav"]
    TG, TA = 0.00001, 0.3
    mov_x = "X: cap a l'est"  if d_lon >  TG else "X: cap a l'oest" if d_lon < -TG else "X estable"
    mov_y = "Y: cap al nord"  if d_lat >  TG else "Y: cap al sud"    if d_lat < -TG else "Y estable"
    mov_z = "Z: pujant"       if d_alt >  TA else "Z: baixant"       if d_alt < -TA else "Altitud estable"
 
    if coords_valides(a["lat"], a["lon"]) and coords_valides(b["lat"], b["lon"]):
        _, m_lon  = metres_per_grau(a["lat"])
        dx_m      = d_lon * m_lon
        dy_m      = d_lat * 111320.0
        vel_lineal = float(a["vel_lineal_calc"])
    else:
        dx_m = dy_m = vel_lineal = 0.0
 
    comp = []
    if dy_m >  0.3: comp.append("nord")
    elif dy_m < -0.3: comp.append("sud")
    if dx_m >  0.3: comp.append("est")
    elif dx_m < -0.3: comp.append("oest")
 
    return {"mov_x": mov_x, "mov_y": mov_y, "mov_z": mov_z,
            "vel_lineal": vel_lineal, "direccio": "-".join(comp) if comp else "sense moviment"}
 
 
def calcular_temps_aprox_aterratge(df, h_guanyada, fase):
    if fase != "Descens" or h_guanyada <= 0 or len(df) < 5:
        return None
    vels = df["vel_calc"].tail(8)
    vels_neg = vels[vels < -0.25]
    if len(vels_neg) < 2:
        return None
    vd = abs(float(vels_neg.mean()))
    return h_guanyada / vd if vd > 0 else None
 
 
def format_temps_aprox(segons) -> str:
    if segons is None:
        return "-"
    total = max(0, int(round(segons)))
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h {m:02d}m"
    if m: return f"{m}m {s:02d}s"
    return f"{s}s"
 
 
# =========================
# GRÀFIQUES
# =========================
def _submostreig(df: pd.DataFrame, max_punts: int) -> pd.DataFrame:
    if len(df) <= max_punts:
        return df
    step = len(df) // max_punts
    idx  = list(range(0, len(df), step))
    if idx[-1] != len(df) - 1:
        idx.append(len(df) - 1)
    return df.iloc[idx]
 
 
def mini_grafic(df, y, title):
    dfs = _submostreig(df, GRAF_MAX_PUNTS)
    fig = px.line(dfs, x="temps", y=y, title=title)
    fig.update_layout(
        height=260, margin={"l": 20, "r": 20, "t": 50, "b": 20},
        transition={"duration": 0}, uirevision=f"graf-{y}",
    )
    return fig
 
 
# =========================
# MAPA LEAFLET
# =========================
def generar_html_mapa_leaflet(lat, lon, zoom=18, height=650):
    return f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
html,body{{margin:0;padding:0;background:transparent;}}
#map{{width:100%;height:{height}px;border-radius:14px;overflow:hidden;background:#d9d9d9;}}
.pm{{width:16px;height:16px;background:#ff3b30;border:3px solid rgba(255,255,255,0.95);border-radius:50%;}}
</style></head><body>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map=L.map("map",{{zoomControl:true,preferCanvas:true,zoomAnimation:false,fadeAnimation:false,markerZoomAnimation:false}}).setView([{lat},{lon}],{zoom});
L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png",{{maxZoom:19,maxNativeZoom:19,detectRetina:false,attribution:"&copy; OpenStreetMap contributors"}}).addTo(map);
const icon=L.divIcon({{className:"",html:'<div class="pm"></div>',iconSize:[22,22],iconAnchor:[11,11]}});
L.marker([{lat},{lon}],{{icon}}).addTo(map);
L.circle([{lat},{lon}],{{color:"#ff3b30",weight:2,fillColor:"#ff3b30",fillOpacity:0.10,radius:8}}).addTo(map);
setTimeout(()=>map.invalidateSize(),150);
</script></body></html>"""
 
 
def renderitzar_mapa():
    gps = st.session_state.last_valid_gps
    if gps is None:
        st.warning("No hi ha coordenades GPS vàlides per mostrar el mapa.")
        return
    lat, lon = float(gps["lat"]), float(gps["lon"])
    if not coords_valides(lat, lon):
        st.warning("Coordenades invàlides.")
        return
 
    now      = time.time()
    prev_lat = st.session_state.map_lat_render
    prev_lon = st.session_state.map_lon_render
    dist_m   = distancia_metres(prev_lat, prev_lon, lat, lon) if prev_lat is not None else 999999.0
    elapsed  = now - st.session_state.map_last_render_time
 
    if (st.session_state.map_html_cached == ""
            or dist_m >= MAP_MOVE_THRESHOLD_METERS
            or elapsed >= MAP_FORCE_REFRESH_SECONDS):
        st.session_state.map_lat_render       = lat
        st.session_state.map_lon_render       = lon
        st.session_state.map_last_render_time = now
        st.session_state.map_html_cached      = generar_html_mapa_leaflet(lat, lon, MAP_ZOOM, MAP_HEIGHT)
 
    st.markdown('<div class="map-wrap">', unsafe_allow_html=True)
    components.html(st.session_state.map_html_cached, height=MAP_HEIGHT, scrolling=False)
    st.markdown("</div>", unsafe_allow_html=True)
 
 
# =========================
# HTML CARDS
# =========================
_CARD = (
    "background:rgba(8,18,36,0.92);"
    "border:1px solid rgba(86,142,255,0.15);"
    "border-radius:18px;padding:20px 24px;"
    "box-shadow:0 4px 20px rgba(0,0,0,0.30),inset 0 1px 0 rgba(255,255,255,0.04);"
    "height:100%;box-sizing:border-box;"
)
 
def _sec(t):
    return (f'<div style="font-size:0.65rem;font-weight:700;letter-spacing:0.14em;color:#334155;'
            f'text-transform:uppercase;border-top:1px solid rgba(255,255,255,0.07);'
            f'padding-top:12px;margin-top:6px;margin-bottom:10px;">{t}</div>')
 
def _m(label, value, color="#f1f5f9", size="1.8rem"):
    return (f'<div style="margin-bottom:14px;">'
            f'<div style="font-size:0.67rem;font-weight:700;letter-spacing:0.11em;'
            f'color:#475569;text-transform:uppercase;margin-bottom:3px;">{label}</div>'
            f'<div style="font-size:{size};font-weight:700;color:{color};'
            f'line-height:1.05;letter-spacing:-0.01em;">{value}</div></div>')
 
def _m2(items):
    return (f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0 20px;">'
            + "".join(f'<div>{_m(l,v,c)}</div>' for l,v,c in items) + '</div>')
 
def _m4(items):
    return (f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0 16px;">'
            + "".join(f'<div>{_m(l,v,c)}</div>' for l,v,c in items) + '</div>')
 
def _vc(v):
    return "#34d399" if v > 0.5 else "#fb923c" if v < -0.5 else "#94a3b8"
 
def _html_card_left(hora_txt, retard_s, estat_txt):
    ec = "#34d399" if estat_txt == "OK" else "#fbbf24" if estat_txt == "RETARD" else "#f87171"
    rc = "#34d399" if retard_s <= 3 else "#fbbf24" if retard_s <= 10 else "#f87171"
    return (f'<div style="{_CARD}">'
            + _m("Hora de missió", hora_txt, "#f8fbff", "2rem")
            + _sec("COMUNICACIÓ") + _m("Retard", f"{retard_s:.0f} s", rc)
            + _sec("ESTAT") + _m("Sistema", estat_txt, ec) + '</div>')
 
def _html_card_mid(alt, alt_max, h_guanyada, alt_press, temp, press, temps_aterr):
    return (f'<div style="{_CARD}">'
            f'<div style="font-size:0.65rem;font-weight:700;letter-spacing:0.14em;'
            f'color:#334155;text-transform:uppercase;margin-bottom:10px;">POSICIO I ALTITUD</div>'
            + _m4([("Altitud actual",  f"{alt:.1f} m",        "#38bdf8"),
                   ("Altitud maxima",  f"{alt_max:.1f} m",     "#f1f5f9"),
                   ("Altura guanyada", f"{h_guanyada:.1f} m",  "#f1f5f9"),
                   ("Altitud pressio", f"{alt_press:.1f} m",   "#94a3b8")])
            + _sec("CONDICIONS AMBIENTALS")
            + _m2([("Temperatura", f"{temp:.1f} °C",   "#f1f5f9"),
                   ("Pressio",     f"{press:.1f} hPa",  "#f1f5f9")])
            + _sec("TEMPS APROX. ATERRATGE")
            + _m("Aterratge estimat", temps_aterr, "#fb923c" if temps_aterr != "-" else "#475569")
            + '</div>')
 
def _html_card_right(vel_env, vel_vert, vel_lin, met_txt):
    return (f'<div style="{_CARD}">'
            f'<div style="font-size:0.65rem;font-weight:700;letter-spacing:0.14em;'
            f'color:#334155;text-transform:uppercase;margin-bottom:10px;">VELOCITAT</div>'
            + _m2([("Enviada",  f"{vel_env:.2f} m/s",  "#f1f5f9"),
                   ("Vertical", f"{vel_vert:+.2f} m/s", _vc(vel_vert))])
            + _m("Lineal (horitzontal)", f"{vel_lin:.2f} m/s", "#38bdf8")
            + _sec("T+ MISSIO")
            + _m("MET · Mission Elapsed Time", met_txt, "#a78bfa", "2rem")
            + '</div>')
 
 
_FASES = {
    "Esperant enlairament": {
        "icon": "", "bg": "linear-gradient(135deg,#0f172a,#1e293b)",
        "border": "rgba(100,116,139,0.55)", "glow": "rgba(100,116,139,0.15)", "color": "#94a3b8",
        "desc": "Sistemes actius. Esperant el moment de l'enlairament.",
    },
    "Ascens": {
        "icon": "", "bg": "linear-gradient(135deg,#052e16,#064e3b)",
        "border": "rgba(52,211,153,0.7)", "glow": "rgba(52,211,153,0.18)", "color": "#34d399",
        "desc": "El cohet s'està enlairant. Altitud en augment constant.",
    },
    "Vol actiu": {
        "icon": "", "bg": "linear-gradient(135deg,#0c1a35,#0c2a4a)",
        "border": "rgba(56,189,248,0.6)", "glow": "rgba(56,189,248,0.15)", "color": "#38bdf8",
        "desc": "En vol. Monitoritzant posició, alçada i condicions.",
    },
    "Descens": {
        "icon": "🪂", "bg": "linear-gradient(135deg,#2c0a00,#431407)",
        "border": "rgba(249,115,22,0.7)", "glow": "rgba(249,115,22,0.18)", "color": "#fb923c",
        "desc": "Descens actiu. Calculant temps d'aterratge.",
    },
    "Aterrat": {
        "icon": "", "bg": "linear-gradient(135deg,#0f172a,#1c1917)",
        "border": "rgba(168,162,158,0.55)", "glow": "rgba(168,162,158,0.12)", "color": "#a8a29e",
        "desc": "Missió completada. Cohet recuperat a terra.",
    },
}
 
def _html_card_fase(fase, retard_s):
    cfg = _FASES.get(fase, _FASES["Vol actiu"])
    if   retard_s <= 3:  rd,rb,rbrd,rt = "#34d399","rgba(52,211,153,0.12)","rgba(52,211,153,0.35)",  f"Temps real · retard {retard_s:.0f} s"
    elif retard_s <= 10: rd,rb,rbrd,rt = "#fbbf24","rgba(251,191,36,0.12)","rgba(251,191,36,0.35)",  f"Petit retard · {retard_s:.0f} s"
    else:                rd,rb,rbrd,rt = "#f87171","rgba(248,113,113,0.12)","rgba(248,113,113,0.35)",f"Retard important · {retard_s:.0f} s"
    return f"""<div class="fase-card" style="background:{cfg['bg']};border:1.5px solid {cfg['border']};
    box-shadow:0 0 28px {cfg['glow']},0 8px 20px rgba(0,0,0,0.35);color:{cfg['color']};">
  <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.13em;opacity:0.7;text-transform:uppercase;margin-bottom:2px;">ETAPA DE LA MISSIÓ</div>
  <div class="fase-icon">{cfg['icon']}</div>
  <div class="fase-nom">{fase}</div>
  <div class="fase-desc">{cfg['desc']}</div>
  <div class="fase-retard" style="background:{rb};border:1px solid {rbrd};color:{rd};">
    <span class="retard-dot" style="background:{rd};box-shadow:0 0 6px {rd};"></span>{rt}
  </div>
</div>"""
 
 
def _html_card_moviment(moviment, vel_vertical, vel_lineal, direccio, temps_aterratge_txt, fase):
    SC = ("background:rgba(8,18,36,0.92);border:1.5px solid rgba(56,189,248,0.22);"
          "border-radius:20px;padding:22px 24px;display:flex;flex-direction:column;"
          "gap:14px;min-height:220px;box-shadow:0 8px 20px rgba(0,0,0,0.3);")
    SL = "font-size:0.72rem;font-weight:700;letter-spacing:0.13em;color:#64748b;text-transform:uppercase;margin-bottom:2px;"
    SP = "font-size:1.55rem;font-weight:800;color:#f8fbff;line-height:1.2;"
    SF = "display:flex;align-items:center;gap:12px;padding:10px 14px;border-radius:12px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);"
    SI = "font-size:1.35rem;flex-shrink:0;"
    ST = "font-size:0.97rem;color:#cbd5e1;line-height:1.35;flex:1;"
    SV = "font-size:1.05rem;font-weight:800;flex-shrink:0;"
 
    if vel_vertical > 0.5:
        vi, vc, vw, vd = "", "#34d399", "PUJANT", "pujant"
    elif vel_vertical < -0.5:
        vi, vc, vw, vd = "", "#fb923c", "BAIXANT", "baixant"
    else:
        vi, vc, vw, vd = "", "#94a3b8", "ALTITUD ESTABLE", "estable"
 
    vp = (f"El cohet està <span style='color:{vc};font-weight:900;'>{vw}</span>"
          f" a <span style='color:#38bdf8;font-weight:900;'>{abs(vel_vertical):.1f} m/s</span>"
          if vw != "ALTITUD ESTABLE"
          else "El cohet està <span style='color:#94a3b8;font-weight:900;'>ALTITUD ESTABLE</span>")
 
    dc = direccio.upper() if direccio != "sense moviment" else "SENSE MOVIMENT"
    hi = "" if vel_lineal > 0.3 else ""
    hl = (f"Cap al <strong style='color:#f1f5f9;'>{dc}</strong>" if vel_lineal > 0.3
          else "<strong style='color:#94a3b8;'>Sense desplaçament horitzontal</strong>")
    v3d = (vel_vertical**2 + vel_lineal**2)**0.5
 
    lf = ""
    if fase == "Descens" and temps_aterratge_txt != "-":
        lf = (f'<div style="{SF}"><span style="{SI}"></span>'
              f'<span style="{ST}">Temps aprox. d\'aterratge</span>'
              f'<span style="{SV}color:#fb923c;">{temps_aterratge_txt}</span></div>')
 
    return (f'<div style="{SC}"><div style="{SL}">MOVIMENT</div>'
            f'<div style="{SP}">{vp}</div>'
            f'<div style="{SF}"><span style="{SI}">{vi}</span>'
            f'<span style="{ST}"><strong style="color:#f1f5f9;">Velocitat vertical</strong> · {vd}</span>'
            f'<span style="{SV}color:{vc};">{vel_vertical:+.2f} m/s</span></div>'
            f'<div style="{SF}"><span style="{SI}">{hi}</span>'
            f'<span style="{ST}">{hl}</span>'
            f'<span style="{SV}color:#38bdf8;">{vel_lineal:.2f} m/s</span></div>'
            f'<div style="{SF}"><span style="{SI}"></span>'
            f'<span style="{ST}"><strong style="color:#f1f5f9;">Velocitat total</strong> (3D)</span>'
            f'<span style="{SV}color:#e2e8f0;">{v3d:.2f} m/s</span></div>'
            f'{lf}</div>')
 
 
# =========================
# RENDER GPS + MAPA
# =========================
def renderitzar_bloc_gps_i_mapa(dada, fase, h_guanyada, h_max, vel_lineal, direccio, temps_aterr, h_base):
    st.subheader("Posició GPS en temps real")
    gps = st.session_state.last_valid_gps
    col_info, col_map = st.columns([1, 1.55], gap="large")
 
    with col_info:
        if gps is None:
            st.warning("No hi ha coordenades GPS vàlides.")
        else:
            cam_map = {"left": ("", "left"), "right": ("", "right"),
                       "center": ("⏺", "center"), "up": ("", "up"), "down": ("", "down")}
            cxi, cxt = cam_map.get(str(dada.get("camX","center")).strip().lower(), ("⏺","center"))
            cyi, cyt = cam_map.get(str(dada.get("camY","center")).strip().lower(), ("⏺","center"))
            st.markdown(f"""<div class="info-card"><h3>Posició actual</h3><div class="info-grid">
                <div class="info-item"><b>Hora:</b> {dada['temps_txt']}</div>
                <div class="info-item"><b>Retard:</b> {dada['retard_s']:.0f} s</div>
                <div class="info-item"><b>Latitud:</b> {gps['lat']:.6f}</div>
                <div class="info-item"><b>Longitud:</b> {gps['lon']:.6f}</div>
                <div class="info-item"><b>Etapa:</b> {fase}</div>
                <div class="info-item"><b>Altitud:</b> {dada['alt']:.1f} m</div>
                <div class="info-item"><b>Altitud per pressió:</b> {dada['alt_press']:.1f} m</div>
                <div class="info-item"><b>Altura guanyada:</b> {h_guanyada:.1f} m</div>
                <div class="info-item"><b>Altura màxima:</b> {h_max:.1f} m</div>
                <div class="info-item"><b>Velocitat enviada:</b> {dada['vel']:.2f} m/s</div>
                <div class="info-item"><b>Velocitat lineal:</b> {vel_lineal:.2f} m/s</div>
                <div class="info-item"><b>Direcció:</b> {direccio}</div>
                <div class="info-item"><b>Temperatura:</b> {dada['temp']:.1f} °C</div>
                <div class="info-item"><b>Pressió:</b> {dada['press']:.1f} hPa</div>
                <div class="info-item"><b>Càmera X:</b> {cxi} {cxt}</div>
                <div class="info-item"><b>Càmera Y:</b> {cyi} {cyt}</div>
                <div class="info-item"><b>Temps aterratge:</b> {temps_aterr}</div>
                <div class="info-item"><b>Altura llançament:</b> {"pendent" if h_base is None else f"{h_base:.1f} m"}</div>
            </div></div>""", unsafe_allow_html=True)
 
    with col_map:
        renderitzar_mapa()
 
 
# =========================
# DASHBOARD
# =========================
def renderitzar_dashboard():
    if not st.session_state.historial:
        if st.session_state.api_error_count > 0:
            st.info("Esperant dades del servidor… Les últimes dades apareixeran aquí tan aviat com es rebi la primera resposta.")
        else:
            st.write("Esperant dades…")
        return
 
    df           = obtenir_df()
    dada         = df.iloc[-1]
    fase         = obtenir_fase_intelligent(df)
    vel_vertical = calcular_velocitat_vertical(df)
    moviment     = calcular_moviment_i_velocitat_lineal(df)
 
    h_guanyada = float(dada["altura_guanyada"])
    h_max      = float(dada["altura_maxima_total"])
    vel_lineal = moviment["vel_lineal"]
    dir_lineal = moviment["direccio"]
    h_base     = st.session_state.altura_base
 
    temps_aterr_s   = calcular_temps_aprox_aterratge(df, h_guanyada, fase)
    temps_aterr_txt = format_temps_aprox(temps_aterr_s)
 
    # MET
    if st.session_state.get("launch_temps") is not None:
        met_s    = max(0, int(float(dada["temps"]) - st.session_state.launch_temps))
        h_m, rem = divmod(met_s, 3600)
        m_m, s_m = divmod(rem, 60)
        met_txt  = f"{h_m:02d}:{m_m:02d}:{s_m:02d}"
    else:
        met_txt = "–"
 
    retard_s  = float(dada["retard_s"])
    estat_txt = "OK" if retard_s <= 3 else "RETARD" if retard_s <= 10 else "NO OK"
 
    # Fila superior
    c1, c2, c3 = st.columns([1.05, 3.4, 2.1], gap="large")
    with c1: st.markdown(_html_card_left(dada["temps_txt"], retard_s, estat_txt), unsafe_allow_html=True)
    with c2: st.markdown(_html_card_mid(float(dada["alt"]), h_max, h_guanyada, float(dada["alt_press"]),
                                        float(dada["temp"]), float(dada["press"]), temps_aterr_txt), unsafe_allow_html=True)
    with c3: st.markdown(_html_card_right(float(dada["vel"]), vel_vertical, vel_lineal, met_txt), unsafe_allow_html=True)
 
    # Fase + Moviment
    ce, cm = st.columns(2)
    with ce: st.markdown(_html_card_fase(fase, retard_s), unsafe_allow_html=True)
    with cm: st.markdown(_html_card_moviment(moviment, vel_vertical, vel_lineal, dir_lineal, temps_aterr_txt, fase), unsafe_allow_html=True)
 
    # GPS + Mapa
    renderitzar_bloc_gps_i_mapa(dada, fase, h_guanyada, h_max, vel_lineal, dir_lineal, temps_aterr_txt, h_base)
 
    # Gràfiques
    if len(df) >= 2:
        st.subheader("Gràfiques principals")
        st.plotly_chart(mini_grafic(df, "alt", "Altitud vs Temps"),
                        use_container_width=True, config=PLOTLY_CONFIG, key="fig_alt")
        a1, a2, a3 = st.columns(3)
        with a1: st.plotly_chart(mini_grafic(df, "alt_press",  "Altitud per pressió"),         use_container_width=True, config=PLOTLY_CONFIG, key="fig_alt_press")
        with a2: st.plotly_chart(mini_grafic(df, "temp",       "Temperatura"),                 use_container_width=True, config=PLOTLY_CONFIG, key="fig_temp")
        with a3: st.plotly_chart(mini_grafic(df, "press",      "Pressió"),                     use_container_width=True, config=PLOTLY_CONFIG, key="fig_press")
        b1, b2, b3 = st.columns(3)
        with b1: st.plotly_chart(mini_grafic(df, "vel",            "Velocitat enviada"),           use_container_width=True, config=PLOTLY_CONFIG, key="fig_vel_real")
        with b2: st.plotly_chart(mini_grafic(df, "vel_calc",       "Velocitat vertical calculada"), use_container_width=True, config=PLOTLY_CONFIG, key="fig_vel_calc")
        with b3: st.plotly_chart(mini_grafic(df, "vel_lineal_calc","Velocitat lineal"),            use_container_width=True, config=PLOTLY_CONFIG, key="fig_vlin")
 
    st.subheader("Últimes dades")
    cols_ok = [c for c in TAULA_COLUMNS if c in df.columns]
    st.dataframe(df[cols_ok].tail(10), use_container_width=True)
 
 
# =========================
# LOOP
# =========================
if hasattr(st, "fragment"):
 
    @st.fragment(run_every=f"{REFRESH_SECONDS}s")
    def bloc_temps_real():
        renderitzar_banner_connexio()
        processar_lectura_api()
        renderitzar_dashboard()
 
    bloc_temps_real()
 
else:
    renderitzar_banner_connexio()
    processar_lectura_api()
    renderitzar_dashboard()
    interval = REFRESH_ERROR_SECONDS if st.session_state.api_error_count > 0 else REFRESH_SECONDS
    time.sleep(interval)
    st.rerun()
