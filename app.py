import streamlit as st
from modules.state import initialize_session_state, reset_state
from modules.ui_components import(
    render_header, render_arima_inputs, render_prophet_inputs,
    render_auto_arima_inputs
)
from modules.processing import ( 
    cached_load, get_target_columns, test_adf,transform_series,
    train_test_split, prepare_prophet_data
)
from modules.theme import (
    create_plot, plot_seasonal_decompose, plot_corr_func,
    plot_backtest_results, plot_future_forecast
)
from modules.evaluation import calculate_metrics
from modules.models import (
    run_arima, run_prophet_forecast, run_stats_auto_arima
)
import numpy as np
import pandas as pd


render_header()
initialize_session_state()

uploaded_file = st.file_uploader("Upload Bitcoin Historical CSV", type="csv")



if uploaded_file:
    df = cached_load(uploaded_file)
    
    if df is not None:

        with st.sidebar:
            st.markdown("## 🛠️ Control Panel")

        with st.sidebar.expander("1. Data Selection", expanded=(st.session_state.step == 1)):
            
            available_cols = get_target_columns(df)
            selected_price = st.selectbox("Price Target", options=available_cols)

                    
        with st.sidebar.expander("2. EDA", expanded=(st.session_state.step == 2)):
            model_type = st.radio("Decomposition Type", ["additive", "multiplicative"], horizontal=True)
            period = st.number_input("Periodicity", min_value=1, value=7)                
                
            st.divider()
                
            lags = st.number_input("Max Lags (ACF/PACF)", min_value=10, max_value=100, value=40)
                            
        
        with st.sidebar.expander("3. Preprocessing", expanded=(st.session_state.step == 3)):
            st.info("Transform the data.")

            st.session_state.apply_log = st.checkbox(
                "Log Transform", 
                value=st.session_state.apply_log
            )
            d = st.number_input(
                "Difference Lag (d)", 
                min_value=0, max_value=30, value=0, 
                help="Removes general trends. 1 is usually enough to make the mean constant."
            )

            D = st.number_input(
                "Seasonal Difference (D)", 
                min_value=0, max_value=30, value=0, 
                help="Removes seasonal patterns. Use 1 if your data has a repeating cycle."
            )

            s = st.number_input(
                "Seasonal Period (s)", 
                min_value=0, max_value=365, value=0, 
                help="Number of days in a cycle. Use 7 for weekly patterns or 365 for yearly patterns."
            )
            

            if st.button("Apply Transformation", width='stretch'):
                
                series = transform_series(df[selected_price].copy(),
                                           st.session_state.apply_log, d, D, s)
                
                st.session_state.update({
                    "processed_data": series,
                    "current_diff": d,
                    "is_processed": True,
                    "step": 3
                })
                st.rerun()
        
        with st.sidebar.expander("4. Model Selection", expanded=(st.session_state.step == 4)):
            model_choice = st.selectbox("Algorithm", ["ARIMA/SARIMA", "Prophet", "Auto ARIMA (StatsForecast)"])

            horizon = st.slider(
                f"Forecast Horizon (Days)", 
                min_value=1, 
                max_value=365, 
                value=30
            )


            ci_level = st.select_slider("Confidence Interval", options=[0.80, 0.90, 0.95, 0.99], value=0.95)
            
            st.divider()
            st.subheader("Technical Indicators (Optional)")
            st.info("Visual context for trend analysis")

            show_sma = st.toggle("Show SMA (20)", value=False, help="Simple Moving Average - Monthly Trend")
            show_ema = st.toggle("Show EMA (50)", value=False, help="Exponential Moving Average - Long-term Trend")

            
            if st.button("Configure Model", width='stretch'):
                st.session_state.step = 4
                st.rerun()


        with st.sidebar:
            st.divider() 
            st.header("Roll Back")
            
            if st.button("🔄 Reset to Original", width='stretch'):
                reset_state()
                st.rerun()

        # Main UI
        is_proc = st.session_state.is_processed and st.session_state.processed_data is not None
        display_series = st.session_state.processed_data if is_proc else df[selected_price]
        view_label = "PROCESSED SIGNAL" if is_proc else "ORIGINAL HISTORICAL TREND"

        display_df = pd.DataFrame({selected_price: display_series}, index=display_series.index)

        st.subheader(f"📈 {view_label}: {selected_price}")

        main_fig = create_plot(display_df, selected_price)

        st.plotly_chart(main_fig, width='stretch')
        st.divider()

        tab_adf, tab_corr, tab_decomp = st.tabs(["🎯 Stationarity", "📉 Correlations", "📊 Decomposition"])

        with tab_adf:
            res = test_adf(display_series)
            st.subheader("Stationarity Results (ADF)")
            c1, c2, c3 = st.columns(3)
            c1.metric("ADF Statistic", f"{res['statistic']:.4f}")
            c2.metric("P-Value", f"{res['p_value']:.4f}")
            c3.metric("Result", "Stationary" if res['is_stationary'] else "Non-Stationary")
            
            if not res['is_stationary']:
                st.error("Series is non-stationary. Transformation recommended.")
            else:
                st.success("Series is stationary.")

            if st.session_state.is_processed:
                if st.session_state.current_diff > 0:
                    st.caption(f"ADF applied after {st.session_state.current_diff} differencing steps")
                else:
                    st.caption("ADF applied without differencing")

        with tab_corr:
            st.subheader("Autocorrelation Analysis")
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(plot_corr_func(display_series, 'acf', nlags=lags), width='stretch')
            with col_b:
                st.plotly_chart(plot_corr_func(display_series, 'pacf', nlags=lags), width='stretch')
            
            st.info("""
            **How to interpret:**
            - **Spikes outside the blue zone** are statistically significant.
            - **ACF spikes** suggest Moving Average (MA) terms or strong seasonal cycles.
            - **PACF spikes** suggest Auto-Regressive (AR) terms or specific 'lag' features for ML.
            """)


        with tab_decomp:
            try:
                fig_d = plot_seasonal_decompose(display_df, selected_price, model_type, period)
                st.plotly_chart(fig_d, width='stretch')
            except:
                st.warning("Decomposition not available for current state.")



    
        if st.session_state.step == 4:
            st.markdown("---")
            st.header("Model Laboratory")

            tab_val, tab_future = st.tabs(["Validation & Backtesting", "Future Forecast"]) 

            with tab_val:
                st.subheader(f"Configure {model_choice}")  
                if model_choice == "ARIMA/SARIMA":
                    arima_order, seasonal_order = render_arima_inputs()

                elif model_choice == "Prophet":
                    prophet_params = render_prophet_inputs()

         
                elif model_choice == "Auto ARIMA (StatsForecast)":
                    auto_arima_params = render_auto_arima_inputs()


                if st.button("Run Backtest", width='stretch'):

                    base_series = df[selected_price].copy()
                    if st.session_state.apply_log:
                        base_series = np.log1p(base_series)

                    train, test = train_test_split(base_series, train_ratio=0.8)

                    if model_choice == "ARIMA/SARIMA":
                        val_results = run_arima(
                        train, 
                        *arima_order,    
                        *seasonal_order, 
                        steps=len(test), 
                        alpha=(1-ci_level)
                    )
                        
                    elif model_choice == "Prophet":
                        train_df = pd.DataFrame(train) 
                        prophet_input = prepare_prophet_data(train_df, selected_price)
                        val_results = run_prophet_forecast(
                            df=prophet_input, 
                            periods=len(test), 
                            ci_width=ci_level,
                            changepoint_prior_scale=prophet_params['changepoint_prior_scale'],
                            changepoint_range=prophet_params['changepoint_range'],
                            growth=prophet_params['growth'],
                            seasonality_mode=prophet_params['seasonality_mode'],
                            seasonality_settings=prophet_params['seasonality_settings'],
                            seasonality_prior_scale=prophet_params['seasonality_prior_scale'],
                        )

                    elif model_choice == "Auto ARIMA (StatsForecast)":
                        confidence_level = int(ci_level * 100)
                        
                        val_results = run_stats_auto_arima(
                            series=train, 
                            horizon=len(test),
                            level=confidence_level,
                            **auto_arima_params
                        )
                    

                    
                    train_plot = pd.DataFrame({selected_price: train}, index=train.index)
                    test_plot = pd.DataFrame({selected_price: test}, index=test.index)
                    forecast_plot = val_results.copy()
                    forecast_plot.index = test.index

                    target_cols = ['mean', 'mean_ci_lower', 'mean_ci_upper']

                    if st.session_state.apply_log:
                        train_plot = np.expm1(train_plot)
                        test_plot = np.expm1(test_plot)
                        forecast_plot[target_cols] = np.expm1(forecast_plot[target_cols])

                    limit = train.max() * 2
                    forecast_plot['mean_ci_upper'] = forecast_plot['mean_ci_upper'].clip(upper=limit)
                    forecast_plot['mean_ci_lower'] = forecast_plot['mean_ci_lower'].clip(lower=0) 
                    forecast_plot['mean'] = forecast_plot['mean'].clip(upper=limit)

                    st.divider()

                    val_fig = plot_backtest_results(
                        train_plot, 
                        test_plot, 
                        forecast_plot, 
                        selected_price,
                        show_sma=show_sma, 
                        show_ema=show_ema
                    )
                    val_fig.update_yaxes(range=[min(train), max(train)], autorange=False)
                    val_fig.update_layout(yaxis_type="linear", uirevision='constant')
                    st.plotly_chart(val_fig, width='stretch')

                    mae, rmse = calculate_metrics(test_plot[selected_price], forecast_plot['mean'])
                    m1, m2 = st.columns(2)
                    m1.metric("MAE (USD)", f"${mae:,.2f}")
                    m2.metric("RMSE (USD)", f"${rmse:,.2f}")

            

            with tab_future:
                st.subheader(f"Predicting {horizon} Days Ahead")
                
                if st.button("Generate Forecast", width='stretch'):
                    base_series = df[selected_price].copy()
                    if st.session_state.apply_log:
                        base_series = np.log1p(base_series)

                    if model_choice == "ARIMA/SARIMA":
                        future_results = run_arima(
                            base_series, 
                            *arima_order, 
                            *seasonal_order, 
                            steps=horizon, 
                            alpha=(1-ci_level)
                        )

                    elif model_choice == "Prophet":
                        full_df = pd.DataFrame(base_series)
                        prophet_input = prepare_prophet_data(full_df, selected_price)
                        future_results = run_prophet_forecast(
                            df=prophet_input, 
                            periods=horizon, 
                            ci_width=ci_level,
                            changepoint_prior_scale=prophet_params['changepoint_prior_scale'],
                            changepoint_range=prophet_params['changepoint_range'],
                            growth=prophet_params['growth'],
                            seasonality_mode=prophet_params['seasonality_mode'],
                            seasonality_settings=prophet_params['seasonality_settings'],
                            seasonality_prior_scale=prophet_params['seasonality_prior_scale'],
                        )

                    elif model_choice == "Auto ARIMA (StatsForecast)":
                        confidence_level = int(ci_level * 100)
                        
                        future_results = run_stats_auto_arima(
                            series=base_series,
                            horizon=horizon,
                            level=confidence_level,
                            **auto_arima_params
                        )
                        
                    last_date = base_series.index[-1]
                    future_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
                    future_results.index = future_index

                    historical_plot_df = pd.DataFrame({selected_price: base_series}, index=base_series.index)
                    
                    target_cols = ['mean', 'mean_ci_lower', 'mean_ci_upper']

                    if st.session_state.apply_log:
                        historical_plot_df = np.expm1(historical_plot_df)
                        future_results[target_cols] = np.expm1(future_results[target_cols])
                    

                    fig_future = plot_future_forecast(
                        historical_plot_df, 
                        future_results, 
                        selected_price,show_sma=show_sma, 
                        show_ema=show_ema
                    )
                    
                    zoom_start = historical_plot_df.index[-90]
                    fig_future.update_xaxes(range=[zoom_start, future_results.index[-1]])

                    st.plotly_chart(fig_future, width='stretch')
                    
                    st.markdown("### 📊 Forecast Data")
                    st.dataframe(
                        future_results[target_cols].style.format("${:,.2f}"), 
                        width='stretch'
                    )
            
                  
    
else:
    st.info("Please upload a CSV file to begin the analysis.")