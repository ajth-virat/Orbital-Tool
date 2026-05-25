"""
app.py — Streamlit web interface for the Orbital GNC Design Tool.

Logic:
This file is the only entry point a user touches — everything else runs behind it.
Input forms collect all orbital parameters and spacecraft properties in one place.
On submit, validator.py runs first — errors are shown before any calculation starts.
The propagator and all output modules are called in sequence by the orchestrator.
Results are displayed in tabs so each output module has its own clean panel.
Plots use Plotly for interactive 3D orbit and ground track visualisation.
All computation stays in Python — the UI only handles display and input collection.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone

# ── Local modules ──────────────────────────────────────────────────
from constants import CONST
from validator import validate_all_keplerian, validate_all_cartesian
from converter import kep_to_cart, cart_to_kep, orbital_period, circular_velocity
from propagator import propagate
from deltav import hohmann, bielliptic, combined_maneuver, phasing, recommend_transfer
from perturbations import atmospheric_density
from coverage import compute_coverage, compute_ground_track, compute_eclipse
from lifetime import compute_lifetime, SOLAR_SCENARIOS
from environment import compute_environment
from adcs_timeline import compute_adcs_timeline, compute_slew, ADCS_MODES


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title  = "Orbital GNC Design Tool",
    page_icon   = "🛰️",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global font and background ── */
    html, body, [class*="css"], .stApp {
        font-family: Georgia, 'Times New Roman', serif !important;
        background-color: #F5F0E8 !important;
        color: #1A2B4A !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #EDE8DC !important;
        border-right: 2px solid #C8BFA8 !important;
        min-width: 320px !important;
        max-width: 320px !important;
        padding: 24px 20px !important;
    }
    [data-testid="stSidebar"] * {
        font-family: Georgia, serif !important;
        color: #1A2B4A !important;
        font-size: 1rem !important;
    }
    [data-testid="stSidebar"] label {
        font-size: 1rem !important;
        font-weight: bold !important;
        color: #1A2B4A !important;
        letter-spacing: 0.3px !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select {
        font-size: 1rem !important;
        background-color: #FAF7F2 !important;
        border: 1px solid #B8A98A !important;
        border-radius: 5px !important;
        color: #1A2B4A !important;
        padding: 6px 10px !important;
    }
    [data-testid="stSidebar"] .stSlider label {
        font-size: 1rem !important;
    }

    /* ── Main content area ── */
    .main .block-container {
        background-color: #F5F0E8 !important;
        padding: 2rem 2.5rem !important;
        max-width: 1400px !important;
    }

    /* ── Title ── */
    .main-title {
        font-family: Georgia, serif !important;
        font-size: 2.4rem;
        font-weight: bold;
        color: #1A2B4A;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .sub-title {
        font-family: Georgia, serif !important;
        font-size: 1rem;
        color: #4A6B8A;
        letter-spacing: 1px;
        margin-top: 0;
        font-style: italic;
    }

    /* ── Section headers ── */
    .section-header {
        font-family: Georgia, serif !important;
        font-size: 0.9rem;
        font-weight: bold;
        color: #1A2B4A;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 2px solid #C8BFA8;
        padding-bottom: 6px;
        margin-bottom: 16px;
        margin-top: 20px;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #EDE8DC 0%, #E4DDD0 100%);
        border: 1px solid #C8BFA8;
        border-radius: 8px;
        padding: 18px 12px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(26,43,74,0.08);
        transition: box-shadow 0.2s;
    }
    .metric-card:hover {
        box-shadow: 0 4px 14px rgba(26,43,74,0.15);
    }
    .metric-value {
        font-family: Georgia, serif !important;
        font-size: 1.5rem;
        font-weight: bold;
        color: #1A2B4A;
        line-height: 1.2;
    }
    .metric-label {
        font-family: Georgia, serif !important;
        font-size: 0.78rem;
        color: #4A6B8A;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 6px;
    }

    /* ── Run button ── */
    .stButton > button {
        background: linear-gradient(135deg, #1A2B4A, #2A4A6B);
        color: #F5F0E8 !important;
        border: none;
        border-radius: 6px;
        font-family: Georgia, serif !important;
        font-size: 1rem;
        letter-spacing: 1px;
        padding: 0.7rem 2rem;
        width: 100%;
        transition: all 0.2s;
        box-shadow: 0 2px 8px rgba(26,43,74,0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2A4A6B, #1A2B4A);
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(26,43,74,0.3);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #EDE8DC !important;
        border-radius: 8px 8px 0 0;
        border-bottom: 2px solid #C8BFA8;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: Georgia, serif !important;
        font-size: 0.9rem !important;
        color: #4A6B8A !important;
        background-color: transparent !important;
        padding: 10px 18px !important;
        border-radius: 6px 6px 0 0 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F5F0E8 !important;
        color: #1A2B4A !important;
        font-weight: bold !important;
        border-top: 3px solid #1A2B4A !important;
    }

    /* ── Alert boxes ── */
    .warning-box {
        background: #FFF8E6;
        border-left: 4px solid #C49A00;
        padding: 12px 16px;
        border-radius: 4px;
        font-family: Georgia, serif !important;
        font-size: 0.92rem;
        color: #6B4E00;
        margin: 8px 0;
    }
    .error-box {
        background: #FFF0EE;
        border-left: 4px solid #B22222;
        padding: 12px 16px;
        border-radius: 4px;
        font-family: Georgia, serif !important;
        font-size: 0.92rem;
        color: #8B0000;
        margin: 8px 0;
    }
    .success-box {
        background: #F0F7EE;
        border-left: 4px solid #2D6A2D;
        padding: 12px 16px;
        border-radius: 4px;
        font-family: Georgia, serif !important;
        font-size: 0.92rem;
        color: #1A4A1A;
        margin: 8px 0;
    }
    .info-box {
        background: #EEF3FA;
        border-left: 4px solid #1A2B4A;
        padding: 12px 16px;
        border-radius: 4px;
        font-family: Georgia, serif !important;
        font-size: 0.92rem;
        color: #1A2B4A;
        margin: 8px 0;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #C8BFA8 !important;
        border-radius: 6px !important;
    }

    /* ── Divider ── */
    hr {
        border-color: #C8BFA8 !important;
        margin: 20px 0 !important;
    }

    /* ── Toggles ── */
    [data-testid="stSidebar"] .stToggle label {
        font-size: 1rem !important;
        font-weight: normal !important;
    }

    /* ── Metric widget override ── */
    [data-testid="stMetric"] {
        background: #EDE8DC;
        border: 1px solid #C8BFA8;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] {
        font-family: Georgia, serif !important;
        color: #4A6B8A !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stMetricValue"] {
        font-family: Georgia, serif !important;
        color: #1A2B4A !important;
        font-size: 1.3rem !important;
        font-weight: bold !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #1A2B4A !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════

col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown('<div class="main-title">🛰️ Orbital GNC Design Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Guidance · Navigation · Control &nbsp;|&nbsp; Full Mission Analysis Suite</div>', unsafe_allow_html=True)

st.divider()


# ══════════════════════════════════════════════════════════════════
# SIDEBAR — ALL INPUTS
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="section-header">Input Mode</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Select input type",
        ["Keplerian Elements", "Cartesian State Vector"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown('<div class="section-header">Orbital State</div>', unsafe_allow_html=True)

    if input_mode == "Keplerian Elements":
        a_km   = st.number_input("Semi-major axis a [km]",    value=6778.0,  step=10.0,  format="%.3f")
        e      = st.number_input("Eccentricity e",             value=0.0001,  step=0.001, format="%.4f")
        i_deg  = st.number_input("Inclination i [deg]",        value=51.6,    step=1.0,   format="%.2f")
        raan   = st.number_input("RAAN Ω [deg]",               value=120.0,   step=5.0,   format="%.2f")
        argp   = st.number_input("Arg. of Perigee ω [deg]",    value=90.0,    step=5.0,   format="%.2f")
        nu     = st.number_input("True Anomaly ν [deg]",       value=0.0,     step=5.0,   format="%.2f")
    else:
        st.markdown("**Position [km]**")
        col_x, col_y, col_z = st.columns(3)
        with col_x: rx = st.number_input("X", value=-181.776, format="%.3f", label_visibility="visible")
        with col_y: ry = st.number_input("Y", value=-5638.771, format="%.3f", label_visibility="visible")
        with col_z: rz = st.number_input("Z", value=3755.797, format="%.3f", label_visibility="visible")
        st.markdown("**Velocity [km/s]**")
        col_u, col_v, col_w = st.columns(3)
        with col_u: vx = st.number_input("U", value=5.6286, format="%.4f", label_visibility="visible")
        with col_v: vy = st.number_input("V", value=-3.0126, format="%.4f", label_visibility="visible")
        with col_w: vz = st.number_input("W", value=-4.2496, format="%.4f", label_visibility="visible")

    st.markdown('<div class="section-header">Spacecraft</div>', unsafe_allow_html=True)

    mass  = st.number_input("Mass [kg]",                value=500.0,  step=10.0,  format="%.1f")
    cd    = st.number_input("Drag coefficient Cd",      value=2.2,    step=0.1,   format="%.2f")
    area  = st.number_input("Cross-section area [m²]",  value=4.0,    step=0.5,   format="%.2f")
    cr    = st.number_input("SRP coefficient Cr",       value=1.3,    step=0.1,   format="%.2f")

    st.markdown('<div class="section-header">Propagation</div>', unsafe_allow_html=True)

    duration_hr = st.slider("Duration [hr]",  min_value=1,  max_value=720, value=24,  step=1)
    step_min    = st.slider("Timestep [min]", min_value=1,  max_value=60,  value=1,   step=1)

    st.markdown('<div class="section-header">Perturbations</div>', unsafe_allow_html=True)

    use_j2   = st.toggle("J2 Oblateness",           value=True)
    use_drag = st.toggle("Atmospheric Drag",         value=True)
    use_srp  = st.toggle("Solar Radiation Pressure", value=True)

    st.markdown('<div class="section-header">Delta-V Analysis</div>', unsafe_allow_html=True)

    target_alt_km = st.number_input("Target altitude [km]", value=800.0, step=50.0, format="%.1f")
    delta_i_dv    = st.number_input("Inclination change Δi [deg]", value=0.0, step=1.0, format="%.1f")
    isp           = st.number_input("Engine Isp [s]",  value=311.0, step=10.0, format="%.1f")

    st.markdown('<div class="section-header">Coverage Analysis</div>', unsafe_allow_html=True)

    gs_lat    = st.number_input("Ground station lat [deg]", value=28.6,  step=1.0, format="%.2f")
    gs_lon    = st.number_input("Ground station lon [deg]", value=-80.6, step=1.0, format="%.2f")
    min_el    = st.number_input("Min elevation [deg]",      value=5.0,   step=1.0, format="%.1f")

    st.markdown('<div class="section-header">Mission & ADCS</div>', unsafe_allow_html=True)

    mission_days    = st.number_input("Mission lifetime [days]", value=365.0, step=30.0, format="%.0f")
    mission_profile = st.selectbox("Mission profile", ["earth_observation", "comms", "technology_demo"])
    sc_size_m       = st.number_input("Spacecraft size [m]", value=0.6, step=0.1, format="%.2f")
    slew_angle      = st.number_input("Typical slew angle [deg]", value=30.0, step=5.0, format="%.1f")
    init_rate       = st.number_input("Initial tumble rate [deg/s]", value=10.0, step=1.0, format="%.1f")

    st.divider()
    run_btn = st.button("▶  RUN ANALYSIS", type="primary")


# ══════════════════════════════════════════════════════════════════
# HELPER: BUILD EARTH SPHERE FOR 3D PLOT
# ══════════════════════════════════════════════════════════════════

def earth_sphere():
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    Re = CONST.R_EARTH / 1e3   # km
    x  = Re * np.outer(np.cos(u), np.sin(v))
    y  = Re * np.outer(np.sin(u), np.sin(v))
    z  = Re * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, "#1A3A6B"], [0.5, "#2E6DA4"], [1, "#4A90C4"]],
        showscale=False, opacity=0.90,
        hoverinfo="skip",
        name="Earth"
    )


# ══════════════════════════════════════════════════════════════════
# MAIN ANALYSIS — runs when button is pressed
# ══════════════════════════════════════════════════════════════════

if run_btn:

    # ── Convert inputs to SI ───────────────────────────────────────
    duration_s = duration_hr * 3600.0
    step_s     = step_min * 60.0
    jd0        = 2451545.0   # J2000 epoch

    if input_mode == "Keplerian Elements":
        a_m = a_km * 1e3
        val = validate_all_keplerian(
            a_m, e, i_deg, raan, argp, nu,
            mass, cd, area, duration_s, step_s
        )
    else:
        r_m = np.array([rx, ry, rz]) * 1e3
        v_ms = np.array([vx, vy, vz]) * 1e3
        val = validate_all_cartesian(r_m, v_ms, mass, cd, area, duration_s, step_s)

    # ── Show validation results ────────────────────────────────────
    if not val.valid:
        st.markdown("### ❌ Validation Failed")
        for err in val.errors:
            st.markdown(f'<div class="error-box">{err}</div>', unsafe_allow_html=True)
        st.stop()

    if val.warnings:
        for w in val.warnings:
            st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)

    # ── Convert to Cartesian if needed ────────────────────────────
    if input_mode == "Keplerian Elements":
        r0, v0 = kep_to_cart(a_m, e, i_deg, raan, argp, nu)
        kep_input = {"a": a_m, "e": e, "i": i_deg, "raan": raan, "argp": argp, "nu": nu}
    else:
        r0, v0 = r_m, v_ms
        kep_input = cart_to_kep(r0, v0)

    # ── Run propagator ─────────────────────────────────────────────
    with st.spinner("Propagating orbit..."):
        result = propagate(
            r0, v0, duration_s, step_s,
            mass, cd, area, cr, jd0,
            use_j2=use_j2, use_drag=use_drag, use_srp=use_srp
        )

    if not result.success:
        st.error(f"Propagation failed: {result.message}")
        st.stop()

    if result.message and "Re-entry" in result.message:
        st.markdown(f'<div class="warning-box">⚠️ {result.message}</div>', unsafe_allow_html=True)

    # ── Run all Phase 2 & 3 modules ────────────────────────────────
    with st.spinner("Computing coverage, lifetime, environment, ADCS..."):
        jd0_cov = jd0

        cov_result = compute_coverage(
            result.t, result.r, jd0_cov,
            gs_lat_deg=gs_lat, gs_lon_deg=gs_lon,
            gs_alt_m=0.0, min_el_deg=min_el
        )

        life_result = compute_lifetime(
            kep_input["a"], kep_input["e"], mass, cd, area
        )

        env_result = compute_environment(
            t_s             = result.t,
            r_eci           = result.r,
            lat_deg         = cov_result.ground_track.lat,
            lon_deg         = cov_result.ground_track.lon,
            alt_km          = result.altitude / 1e3,
            inclination_deg = kep_input["i"],
            mission_days    = mission_days
        )

        adcs_result = compute_adcs_timeline(
            t_s                = result.t,
            eclipse_mask       = cov_result.eclipse_mask,
            mass               = mass,
            spacecraft_size_m  = sc_size_m,
            altitude_km        = float(np.mean(result.altitude / 1e3)),
            initial_rate_deg_s = init_rate,
            mission_profile    = mission_profile,
            slew_angle_deg     = slew_angle
        )

    # ══════════════════════════════════════════════════════════════
    # TOP METRICS ROW
    # ══════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">Mission Summary</div>', unsafe_allow_html=True)

    alt0    = result.altitude[0] / 1e3
    alt_f   = result.altitude[-1] / 1e3
    T_min   = orbital_period(kep_input["a"]) / 60
    v_circ  = circular_velocity(kep_input["a"]) / 1e3

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    for col, label, value in zip(
        [m1, m2, m3, m4, m5, m6],
        ["Initial Alt", "Final Alt", "Δ Altitude", "Period", "Circ. Velocity", "Steps"],
        [f"{alt0:.2f} km", f"{alt_f:.2f} km",
         f"{alt_f - alt0:.4f} km", f"{T_min:.2f} min",
         f"{v_circ:.4f} km/s", f"{len(result.t)}"]
    ):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════

    tab_orbit, tab_alt, tab_kep, tab_deltav, tab_cov, tab_life, tab_env, tab_adcs, tab_data = st.tabs([
        "🌍  3D Orbit",
        "📈  Altitude",
        "📐  Keplerian Elements",
        "🚀  Delta-V",
        "📡  Coverage",
        "⏳  Lifetime",
        "☢️  Environment",
        "🎯  ADCS",
        "📋  Raw Data"
    ])

    # ── TAB 1: 3D ORBIT ───────────────────────────────────────────
    with tab_orbit:
        st.markdown('<div class="section-header">3D Orbital Trajectory — ECI Frame</div>', unsafe_allow_html=True)

        r_km = result.r / 1e3
        t_hr = result.t / 3600

        fig3d = go.Figure()
        fig3d.add_trace(earth_sphere())
        fig3d.add_trace(go.Scatter3d(
            x=r_km[:, 0], y=r_km[:, 1], z=r_km[:, 2],
            mode="lines",
            line=dict(color=t_hr, colorscale="Plasma", width=2),
            name="Trajectory",
            hovertemplate="x: %{x:.1f} km<br>y: %{y:.1f} km<br>z: %{z:.1f} km"
        ))
        # Mark start and end
        fig3d.add_trace(go.Scatter3d(
            x=[r_km[0, 0]], y=[r_km[0, 1]], z=[r_km[0, 2]],
            mode="markers", marker=dict(size=6, color="#00E676"),
            name="Start"
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[r_km[-1, 0]], y=[r_km[-1, 1]], z=[r_km[-1, 2]],
            mode="markers", marker=dict(size=6, color="#FF5252"),
            name="End"
        ))

        fig3d.update_layout(
            scene=dict(
                xaxis=dict(title="X [km]", backgroundcolor="#FAF7F2", gridcolor="#C8BFA8"),
                yaxis=dict(title="Y [km]", backgroundcolor="#FAF7F2", gridcolor="#C8BFA8"),
                zaxis=dict(title="Z [km]", backgroundcolor="#FAF7F2", gridcolor="#C8BFA8"),
                bgcolor="#FAF7F2"
            ),
            paper_bgcolor="#F5F0E8",
            plot_bgcolor="#FAF7F2",
            font=dict(color="#1A2B4A"),
            margin=dict(l=0, r=0, t=20, b=0),
            height=520,
            legend=dict(bgcolor="#FAF7F2", bordercolor="#C8BFA8")
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # ── TAB 2: ALTITUDE ───────────────────────────────────────────
    with tab_alt:
        st.markdown('<div class="section-header">Altitude & Speed vs Time</div>', unsafe_allow_html=True)

        alt_km = result.altitude / 1e3
        spd_km = result.speed / 1e3

        fig_alt = go.Figure()
        fig_alt.add_trace(go.Scatter(
            x=t_hr, y=alt_km,
            mode="lines", name="Altitude",
            line=dict(color="#1A5C8A", width=2)
        ))
        fig_alt.update_layout(
            xaxis_title="Time [hr]", yaxis_title="Altitude [km]",
            paper_bgcolor="#F5F0E8", plot_bgcolor="#FAF7F2",
            font=dict(color="#1A2B4A"),
            xaxis=dict(gridcolor="#C8BFA8"),
            yaxis=dict(gridcolor="#C8BFA8"),
            height=300, margin=dict(l=40, r=20, t=20, b=40)
        )
        st.plotly_chart(fig_alt, use_container_width=True)

        fig_spd = go.Figure()
        fig_spd.add_trace(go.Scatter(
            x=t_hr, y=spd_km,
            mode="lines", name="Speed",
            line=dict(color="#FF8A65", width=2)
        ))
        fig_spd.update_layout(
            xaxis_title="Time [hr]", yaxis_title="Speed [km/s]",
            paper_bgcolor="#F5F0E8", plot_bgcolor="#FAF7F2",
            font=dict(color="#1A2B4A"),
            xaxis=dict(gridcolor="#C8BFA8"),
            yaxis=dict(gridcolor="#C8BFA8"),
            height=260, margin=dict(l=40, r=20, t=20, b=40)
        )
        st.plotly_chart(fig_spd, use_container_width=True)

    # ── TAB 3: KEPLERIAN ELEMENTS ─────────────────────────────────
    with tab_kep:
        st.markdown('<div class="section-header">Osculating Keplerian Elements vs Time</div>', unsafe_allow_html=True)

        kep_a    = np.array([k["a"] / 1e3 for k in result.kep])
        kep_e    = np.array([k["e"]       for k in result.kep])
        kep_i    = np.array([k["i"]       for k in result.kep])
        kep_raan = np.array([k["raan"]    for k in result.kep])

        for label, data, color, unit in [
            ("Semi-major Axis",   kep_a,    "#1A5C8A", "km"),
            ("Eccentricity",      kep_e,    "#CE93D8", ""),
            ("Inclination",       kep_i,    "#80CBC4", "deg"),
            ("RAAN",              kep_raan, "#FFB74D", "deg"),
        ]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t_hr, y=data,
                mode="lines", name=label,
                line=dict(color=color, width=1.5)
            ))
            fig.update_layout(
                xaxis_title="Time [hr]",
                yaxis_title=f"{label} [{unit}]" if unit else label,
                paper_bgcolor="#F5F0E8", plot_bgcolor="#FAF7F2",
                font=dict(color="#1A2B4A"),
                xaxis=dict(gridcolor="#C8BFA8"),
                yaxis=dict(gridcolor="#C8BFA8"),
                height=220, margin=dict(l=50, r=20, t=30, b=40),
                title=dict(text=label, font=dict(size=12, color=color))
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 4: DELTA-V ────────────────────────────────────────────
    with tab_deltav:
        st.markdown('<div class="section-header">Delta-V Budget</div>', unsafe_allow_html=True)

        r1 = kep_input["a"] * (1 - kep_input["e"])    # perigee radius
        r1_circ = kep_input["a"]                       # use SMA as current circular ref
        r2 = CONST.R_EARTH + target_alt_km * 1e3

        rec = recommend_transfer(r1_circ, r2, delta_i_dv, mass=mass, isp=isp)

        best = rec["recommended"]
        st.markdown(f"**Recommended:** `{best.name}`")
        st.markdown(f"**Total ΔV:** `{best.delta_v:.2f} m/s`  |  `{best.delta_v/1e3:.4f} km/s`")

        if best.prop_mass > 0:
            st.markdown(f"**Propellant mass (Isp={isp:.0f}s):** `{best.prop_mass:.2f} kg`")

        st.divider()
        for burn in best.burns:
            b1, b2 = st.columns([2, 5])
            b1.metric("ΔV", f"{burn['dv']:.2f} m/s")
            b2.markdown(f"📍 {burn['location']}")

        if "hohmann" in rec and "bielliptic" in rec:
            st.divider()
            st.markdown("**Transfer comparison:**")
            c1, c2 = st.columns(2)
            c1.metric("Hohmann", f"{rec['hohmann'].delta_v:.2f} m/s",
                      delta=f"ToF {rec['hohmann'].tof_s/3600:.2f} hr")
            c2.metric("Bi-elliptic", f"{rec['bielliptic'].delta_v:.2f} m/s",
                      delta=f"ToF {rec['bielliptic'].tof_s/3600:.2f} hr")

        if best.notes:
            st.markdown(f'<div class="success-box">ℹ️ {best.notes}</div>', unsafe_allow_html=True)

    # ── TAB 5: COVERAGE ───────────────────────────────────────────
    with tab_cov:
        st.markdown('<div class="section-header">Ground Track & Coverage Analysis</div>', unsafe_allow_html=True)

        gt = cov_result.ground_track

        # Ground track map
        fig_gt = go.Figure()
        fig_gt.add_trace(go.Scattergeo(
            lat=gt.lat, lon=gt.lon,
            mode="lines",
            line=dict(color="#1A5C8A", width=1.5),
            name="Ground Track"
        ))
        fig_gt.add_trace(go.Scattergeo(
            lat=[gs_lat], lon=[gs_lon],
            mode="markers+text",
            marker=dict(size=10, color="#FF5252", symbol="triangle-up"),
            text=["GS"], textposition="top right",
            name="Ground Station"
        ))
        fig_gt.update_layout(
            geo=dict(
                showland=True, landcolor="#E8E0D0",
                showocean=True, oceancolor="#C8DCF0",
                showcoastlines=True, coastlinecolor="#8A9BB0",
                showcountries=True, countrycolor="#B0B8C8",
                bgcolor="#FAF7F2",
                projection_type="natural earth"
            ),
            paper_bgcolor="#F5F0E8",
            font=dict(color="#1A2B4A"),
            height=380,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(bgcolor="#F5F0E8", font=dict(color="#1A2B4A"))
        )
        st.plotly_chart(fig_gt, use_container_width=True)

        # Coverage metrics
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val in [
            (c1, "Total Passes",      f"{len(cov_result.access_windows)}"),
            (c2, "Coverage %",        f"{cov_result.coverage_pct:.2f} %"),
            (c3, "Mean Revisit",      f"{cov_result.revisit_mean_s/60:.1f} min"),
            (c4, "Eclipse Fraction",  f"{cov_result.eclipse_frac*100:.1f} %"),
        ]:
            col.markdown(
                f'<div class="metric-card"><div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True
            )

        # Access window table
        if cov_result.access_windows:
            st.divider()
            st.markdown("**Access Windows (first 20)**")
            import pandas as pd
            df_aw = pd.DataFrame([{
                "Pass #":       i + 1,
                "Start [hr]":   round(w.start_s / 3600, 3),
                "End [hr]":     round(w.end_s / 3600, 3),
                "Duration [min]": round(w.duration_s / 60, 2),
                "Max El [deg]": round(w.max_el_deg, 1)
            } for i, w in enumerate(cov_result.access_windows[:20])])
            st.dataframe(df_aw, use_container_width=True, height=300)

    # ── TAB 6: LIFETIME ───────────────────────────────────────────
    with tab_life:
        st.markdown('<div class="section-header">Orbital Lifetime Estimation</div>', unsafe_allow_html=True)

        # Compliance badge
        compliant = life_result.compliant_25yr
        badge_color = "#66BB6A" if compliant else "#EF5350"
        badge_text  = "✓ 25-YEAR RULE COMPLIANT" if compliant else "✗ NON-COMPLIANT — DEORBIT RISK"
        st.markdown(
            f'<div style="background:{badge_color}18;border:2px solid {badge_color};'
            f'border-radius:6px;padding:12px 16px;font-family:Georgia,serif;'
            f'font-size:1rem;color:{badge_color};text-align:center;font-weight:bold;">{badge_text}</div>',
            unsafe_allow_html=True
        )
        st.markdown("")

        # Decay curves plot
        fig_life = go.Figure()
        colors = {"low": "#1A5C8A", "mean": "#FFB74D", "high": "#EF5350"}
        for dc in life_result.decay_curves:
            fig_life.add_trace(go.Scatter(
                x=dc.t_days, y=dc.alt_km,
                mode="lines",
                name=f"{dc.label} ({dc.lifetime_days:.0f} days)",
                line=dict(color=colors[dc.scenario], width=2)
            ))
        fig_life.add_hline(y=80, line_dash="dash", line_color="#888",
                           annotation_text="Re-entry 80 km")
        fig_life.update_layout(
            xaxis_title="Time [days]", yaxis_title="Altitude [km]",
            paper_bgcolor="#F5F0E8", plot_bgcolor="#FAF7F2",
            font=dict(color="#1A2B4A"),
            xaxis=dict(gridcolor="#C8BFA8"),
            yaxis=dict(gridcolor="#C8BFA8"),
            legend=dict(bgcolor="#F5F0E8", font=dict(color="#1A2B4A")),
            height=380, margin=dict(l=50, r=20, t=30, b=50)
        )
        st.plotly_chart(fig_life, use_container_width=True)

        # Lifetime metrics
        l1, l2, l3 = st.columns(3)
        for col, dc in zip([l1, l2, l3], life_result.decay_curves):
            col.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{dc.lifetime_days:.0f} days</div>'
                f'<div class="metric-label">{dc.label}<br>({dc.lifetime_days/365.25:.1f} yrs)</div>'
                f'</div>', unsafe_allow_html=True
            )

    # ── TAB 7: ENVIRONMENT ────────────────────────────────────────
    with tab_env:
        st.markdown('<div class="section-header">Space Environment Report</div>', unsafe_allow_html=True)

        env = env_result
        st.markdown(f"**Orbit type:** `{env.orbit_type}`  |  **Mean altitude:** `{env.mean_alt_km:.1f} km`")
        st.markdown(f"**Mean L-shell:** `{env.L_shell_mean:.3f} R_E`")

        if env.warnings:
            for w in env.warnings:
                st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)
            st.markdown("")

        e1, e2, e3 = st.columns(3)
        e1.markdown('<div class="section-header">Van Allen Belts</div>', unsafe_allow_html=True)
        e1.metric("Inner Belt Exposure", f"{env.belt_exposure.inner_belt_frac*100:.1f} %")
        e1.metric("Outer Belt Exposure", f"{env.belt_exposure.outer_belt_frac*100:.1f} %")
        e1.metric("Safe Zone",           f"{env.belt_exposure.safe_zone_frac*100:.1f} %")

        e2.markdown('<div class="section-header">Total Ionizing Dose</div>', unsafe_allow_html=True)
        e2.metric("1 mm Al shielding",  f"{env.tid.dose_1mm_krad:.2f} krad")
        e2.metric("3 mm Al shielding",  f"{env.tid.dose_3mm_krad:.2f} krad")
        e2.metric("10 mm Al shielding", f"{env.tid.dose_10mm_krad:.4f} krad")

        e3.markdown('<div class="section-header">SAA & Atomic Oxygen</div>', unsafe_allow_html=True)
        e3.metric("SAA Crossings",      f"{env.saa.n_crossings}")
        e3.metric("Time in SAA",        f"{env.saa.total_time_s/3600:.2f} hr")
        e3.metric("AO Flux",            f"{env.atomic_oxygen.flux_atoms_m2_s:.2e} /m²/s")
        ao_concern = "Yes — mitigate" if env.atomic_oxygen.significant else "Negligible"
        e3.metric("AO Concern",         ao_concern)

    # ── TAB 8: ADCS ───────────────────────────────────────────────
    with tab_adcs:
        st.markdown('<div class="section-header">ADCS Mode Timeline</div>', unsafe_allow_html=True)

        adcs = adcs_result

        # Mode timeline Gantt-style bar chart
        import pandas as pd
        fig_adcs = go.Figure()
        mode_colors = {k: v["color"] for k, v in ADCS_MODES.items()}

        for m in adcs.modes:
            fig_adcs.add_trace(go.Bar(
                x=[m.duration_s / 3600],
                y=[m.label],
                orientation="h",
                base=[m.start_s / 3600],
                marker_color=mode_colors.get(m.mode_key, "#888"),
                name=m.label,
                showlegend=False,
                hovertemplate=f"{m.label}<br>Start: {m.start_s/3600:.2f} hr<br>"
                              f"Duration: {m.duration_s/60:.1f} min<br>Power: {m.power_w} W"
            ))

        fig_adcs.update_layout(
            barmode="overlay",
            xaxis_title="Time [hr]", yaxis_title="",
            paper_bgcolor="#F5F0E8", plot_bgcolor="#FAF7F2",
            font=dict(color="#1A2B4A"),
            xaxis=dict(gridcolor="#C8BFA8"),
            yaxis=dict(gridcolor="#C8BFA8"),
            height=300, margin=dict(l=200, r=20, t=20, b=50)
        )
        st.plotly_chart(fig_adcs, use_container_width=True)

        # Detumble and power summary
        d1, d2 = st.columns(2)
        with d1:
            st.markdown('<div class="section-header">Detumbling</div>', unsafe_allow_html=True)
            st.metric("Initial rate",       f"{adcs.detumble.initial_rate_deg_s:.1f} deg/s")
            st.metric("Est. detumble time", f"{adcs.detumble.estimated_time_hr:.2f} hr")
            st.metric("Magnetorquer needed",f"{adcs.detumble.magnetorquer_Am2:.3f} Am²")

        with d2:
            st.markdown('<div class="section-header">Power Budget</div>', unsafe_allow_html=True)
            total_e = sum(m.energy_wh for m in adcs.modes)
            mean_p  = total_e / (adcs.total_duration_s / 3600)
            st.metric("Total ADCS energy", f"{total_e:.1f} Wh")
            st.metric("Mean ADCS power",   f"{mean_p:.2f} W")
            for key, frac in adcs.mode_fractions.items():
                st.markdown(f"`{ADCS_MODES[key]['label']}` — **{frac*100:.1f}%**")

    # ── TAB 9: RAW DATA ───────────────────────────────────────────
    with tab_data:
        st.markdown('<div class="section-header">Propagated State Vectors</div>', unsafe_allow_html=True)

        import pandas as pd

        df = pd.DataFrame({
            "t [hr]":    (result.t / 3600).round(4),
            "x [km]":    (result.r[:, 0] / 1e3).round(3),
            "y [km]":    (result.r[:, 1] / 1e3).round(3),
            "z [km]":    (result.r[:, 2] / 1e3).round(3),
            "vx [km/s]": (result.v[:, 0] / 1e3).round(5),
            "vy [km/s]": (result.v[:, 1] / 1e3).round(5),
            "vz [km/s]": (result.v[:, 2] / 1e3).round(5),
            "alt [km]":  (result.altitude / 1e3).round(3),
            "a [km]":    [round(k["a"] / 1e3, 3) for k in result.kep],
            "e":         [round(k["e"], 6) for k in result.kep],
            "i [deg]":   [round(k["i"], 4) for k in result.kep],
        })

        st.dataframe(df, use_container_width=True, height=400)

        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇️  Download CSV",
            data=csv,
            file_name="gnc_propagation_results.csv",
            mime="text/csv"
        )

else:
    # ── Landing state — no run yet ─────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 80px 0 40px 0;">
        <div style="font-family: Georgia, serif; font-size: 3.5rem; margin-bottom: 16px;">🛰️</div>
        <div style="font-family: Georgia, serif; font-size: 1.2rem; letter-spacing: 2px; color: #1A2B4A; font-weight: bold;">
            Configure inputs in the sidebar<br>then press Run Analysis
        </div>
        <div style="font-family: Georgia, serif; font-size: 0.95rem; color: #4A6B8A; margin-top: 12px; font-style: italic;">
            One orbit. Nine outputs. No switching tools.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c, col_d = st.columns(4)
    for col, icon, title, desc in [
        (col_a, "📡", "Coverage & Ground Track", "Access windows, revisit time, eclipse fraction"),
        (col_b, "⏳", "Orbital Lifetime",         "Drag decay curves across solar min/mean/max scenarios"),
        (col_c, "☢️", "Space Environment",        "Radiation belts, TID, SAA crossings, atomic oxygen"),
        (col_d, "🎯", "ADCS Timeline",            "Detumbling → sun acquisition → nadir-pointing → slew"),
    ]:
        col.markdown(
            f'<div class="metric-card" style="height:150px;">'
            f'<div style="font-size:1.8rem">{icon}</div>'
            f'<div style="font-family:Georgia,serif;font-size:0.9rem;color:#1A2B4A;font-weight:bold;margin:8px 0 6px">{title}</div>'
            f'<div style="font-family:Georgia,serif;font-size:0.82rem;color:#4A6B8A;font-style:italic">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True
        )