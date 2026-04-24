import streamlit as st


def render_header():
    st.set_page_config(
        page_title="NEON-BTC | Forecast Portal", 
        page_icon="⚡", 
        layout="wide"
    )
    st.markdown("<h1 style='text-align: center; color: #ff007f;'>⚡ NEON-BTC FORECASTING PORTAL ⚡</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #00e5ff;'>Advanced Time-Series Analysis for Crypto Markets</p>", unsafe_allow_html=True)

def render_arima_inputs():
    st.info("ARIMA/SARIMA Tuning Parameters")
    c1, c2, c3 = st.columns(3)
    p = c1.number_input("p (AR)", 0, 10, 1)
    d = c2.number_input("d (I)", 0, 10, 1)
    q = c3.number_input("q (MA)", 0, 10, 1)
    with st.expander("Seasonal Order"):
        sc1, sc2, sc3, sc4 = st.columns(4)
        P = sc1.number_input("P", 0, 10, 0)
        D = sc2.number_input("D", 0, 10, 0)
        Q = sc3.number_input("Q", 0, 10, 0)
        S = sc4.number_input("S", 0, 365, 0)
    return (p, d, q), (P, D, Q, S)


def render_prophet_inputs():
    st.info("Prophet Seasonality Tuning")
    
    col1, col2 = st.columns(2)
    growth = col1.selectbox("Growth", ["linear", "logistic"])
    mode = col2.selectbox("Mode", ["additive", "multiplicative"])

    st.divider()
    st.markdown("#### Seasonality Configuration")
    st.caption("Check to enable, and set Fourier Order (Standard: Y=10, W=3, D=4)")

    def seasonality_row(label, default_f):
        c1, c2 = st.columns([1, 2])
        is_on = c1.checkbox(label, value=(label != "Daily")) # Yearly/Weekly True by default
        f_order = c2.number_input(f"{label} Fourier", 1, 30, default_f) if is_on else None
        return f_order

    y_f = seasonality_row("Yearly", 10)
    m_f = seasonality_row("Monthly", 5)
    w_f = seasonality_row("Weekly", 3)
    d_f = seasonality_row("Daily", 4)

    st.divider()
    cp_prior = st.number_input("Changepoint Prior", 0.001, 1.0, 0.05)
    cp_range = st.slider("Changepoint Range", 0.0, 1.0, 0.8)
    spc = st.number_input("Seasonality Prior Scale", 1.0, 30.0, 10.0)

    return {
        "growth": growth, "seasonality_mode": mode,
        "changepoint_prior_scale": cp_prior, "changepoint_range": cp_range,
        "seasonality_prior_scale": spc,
        "seasonality_settings": {"y": y_f, "m": m_f, "w": w_f, "d": d_f}
    }

def render_auto_arima_inputs():
    st.info("StatsForecast: High-Performance Auto ARIMA")
    
    season_length = st.number_input(
        "Season Length", 
        min_value=1, 
        max_value=365, 
        value=7, 
        help="Number of rows per season (e.g., 7 for daily data with weekly patterns)"
    )
    
    return {
        "season_length": season_length
    }