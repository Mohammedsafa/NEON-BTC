import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
import pandas as pd
import numpy as np



def apply_chart_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#080a0f",
        paper_bgcolor="#080a0f",
        font_color="#94a3b8",
        margin=dict(t=100, b=50, l=50, r=50)
    )
    fig.update_xaxes(gridcolor="#1e293b", zeroline=False)
    fig.update_yaxes(gridcolor="#1e293b", zeroline=False)
    return fig

def create_plot(df, target_col):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df[target_col],
        mode='lines',
        line=dict(color='#38bdf8', width=5),
        opacity=0.1,
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df[target_col],
        mode='lines',
        name=f'Actual {target_col}',
        line=dict(color='#38bdf8', width=1.5),
    ))

    return apply_chart_theme(fig)


def plot_seasonal_decompose(df: pd.DataFrame, col: str, model_type: str, period: int):
    decompose = seasonal_decompose(df[col], model=model_type, period=period)
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residuals")
    )
    components = [
        (decompose.observed, '#38bdf8'), 
        (decompose.trend,    '#fbbf24'), 
        (decompose.seasonal, '#c084fc'), 
        (decompose.resid,    '#22c55e')  
    ]

    for i, (data, color) in enumerate(components, start=1):
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data, 
                mode='lines',
                line=dict(color=color, width=1.5),
                name=["Observed", "Trend", "Seasonal", "Residual"][i-1]
            ),
            row=i, col=1
        )
    
    fig.update_layout(
        height=900, 
        title_text=f"Decomposition: {col.upper()}",
        showlegend=False
    )

    return apply_chart_theme(fig)



def plot_corr_func(series, func_type='acf', nlags=40, alpha=0.05,
                   bartlett_confint=True, adjusted=False, fft=False,
                   method='ywm', zero=True, auto_ylims=False):
    data = series.dropna().astype(float)
    func_type = func_type.lower()

    if func_type == 'acf':
        acf_result = acf(
            data,
            nlags=nlags,
            fft=fft,
            alpha=alpha,
            bartlett_confint=bartlett_confint,
            adjusted=adjusted,
            missing='none',
        )
        if alpha is not None:
            y_values, confint = acf_result[:2]
        else:
            y_values = acf_result
            confint = None
        title = "AUTOCORRELATION (ACF)"

    elif func_type == 'pacf':
        pacf_result = pacf(
            data,
            nlags=nlags,
            alpha=alpha,
            method=method,
        )
        if alpha is not None:
            y_values, confint = pacf_result[:2]
        else:
            y_values = pacf_result
            confint = None
        title = "PARTIAL AUTOCORRELATION (PACF)"

    else:
        raise ValueError("func_type must be either 'acf' or 'pacf'")

    x_values = np.arange(len(y_values))

    # Handle zero flag — exclude lag 0 if zero=False
    if not zero:
        x_values = x_values[1:]
        y_values = y_values[1:]
        if confint is not None:
            confint = confint[1:]

    fig = go.Figure()

    # Confidence interval band (always exclude lag 0 from the band,
    # matching statsmodels' _plot_corr behavior)
    if confint is not None:
        if zero:
            # Skip index 0 for the CI band
            ci_x = x_values[1:]
            ci_upper = confint[1:, 1] - y_values[1:]
            ci_lower = confint[1:, 0] - y_values[1:]
        else:
            # lag 0 already removed
            ci_x = x_values
            ci_upper = confint[:, 1] - y_values
            ci_lower = confint[:, 0] - y_values

        fig.add_trace(go.Scatter(
            x=list(ci_x) + list(ci_x[::-1]),
            y=list(ci_upper) + list(ci_lower[::-1]),
            fill='toself',
            fillcolor='rgba(120, 170, 255, 0.35)',
            line=dict(color='rgba(0,0,0,0)'),
            hoverinfo='skip',
            showlegend=False
        ))

    # Zero line
    fig.add_trace(go.Scatter(
        x=[x_values.min(), x_values.max()],
        y=[0, 0],
        mode='lines',
        line=dict(color='#94a3b8', width=1),
        hoverinfo='skip',
        showlegend=False
    ))

    # Stem lines
    for x, y in zip(x_values, y_values):
        fig.add_trace(go.Scatter(
            x=[x, x],
            y=[0, y],
            mode='lines',
            line=dict(color='#38bdf8', width=1.5),
            hoverinfo='skip',
            showlegend=False
        ))

    # Markers
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers',
        marker=dict(color='#ff007f', size=7),
        hoverinfo='x+y',
        showlegend=False
    ))

    fig = apply_chart_theme(fig)

    # Y-axis limits
    if auto_ylims:
        y_min = min(y_values.min(), ci_lower.min() if confint is not None else 0)
        y_max = max(y_values.max(), ci_upper.max() if confint is not None else 0)
        margin = 0.05
        ylims = [y_min - margin, y_max + margin]
    else:
        ylims = [-1.1, 1.1]

    fig.update_layout(
        title=title,
        height=450,
        xaxis=dict(title="Lags", gridcolor="#1e293b"),
        yaxis=dict(range=ylims, gridcolor="#1e293b"),
        showlegend=False
    )

    return fig



def plot_backtest_results(train_df, test_df, forecast_df, target_col, show_sma, show_ema):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_df.index, y=train_df[target_col],
        mode='lines', name='Training Data',
        line=dict(color='#94a3b8', width=1.5, dash='dot')
    ))

    full_series = pd.concat([train_df[target_col], test_df[target_col]])

    if show_sma:
        sma_20 = full_series.rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=sma_20.index, y=sma_20,
            mode='lines', name='SMA (20)',
            line=dict(color='#f59e0b', width=1.5)
        ))

    if show_ema:
        ema_50 = full_series.ewm(span=50, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=ema_50.index, y=ema_50,
            mode='lines', name='EMA (50)',
            line=dict(color='#10b981', width=1.5)
        ))

    

    fig.add_trace(go.Scatter(
        x=test_df.index, y=test_df[target_col],
        mode='lines', name='Actual (Test)',
        line=dict(color='#38bdf8', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['mean_ci_upper'],
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['mean_ci_lower'],
        mode='lines', line=dict(width=0),
        fill='tonexty', 
        fillcolor='rgba(255, 0, 127, 0.15)', 
        name='Confidence Interval',
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['mean'],
        mode='lines', name='Projected Trend',
        line=dict(color='#ff007f', width=3) 
    ))

    split_date = test_df.index[0]

    fig.add_vline(
        x=split_date, 
        line_width=2, 
        line_dash="dash", 
        line_color="#00e5ff"
    )

    fig.add_annotation(
        x=split_date,
        y=1.02,               
        yref="paper",       
        text="Forecast Start",
        showarrow=False,
        font=dict(color="#00e5ff", size=12),
        xanchor="left",
        bgcolor="rgba(8, 10, 15, 0.8)" 
    )

    fig.update_layout(
        title="Model Validation: Actual vs. Predicted (USD)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return apply_chart_theme(fig)




def plot_future_forecast(historical_df, forecast_df, target_col, show_sma, show_ema):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=historical_df.index, y=historical_df[target_col],
        mode='lines', name='Historical Data',
        line=dict(color='#94a3b8', width=1.5)
    ))

    if show_sma:
        sma_20 = historical_df[target_col].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=sma_20.index, y=sma_20,
            mode='lines', name='SMA (20)',
            line=dict(color='#f59e0b', width=1.5) 
        ))

    if show_ema:
        ema_50 = historical_df[target_col].ewm(span=50, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=ema_50.index, y=ema_50,
            mode='lines', name='EMA (50)',
            line=dict(color='#10b981', width=1.5) 
        ))

    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['mean_ci_upper'],
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['mean_ci_lower'],
        mode='lines', line=dict(width=0),
        fill='tonexty', 
        fillcolor='rgba(255, 0, 127, 0.15)', 
        name='Confidence Interval',
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['mean'],
        mode='lines', name='Future Prediction',
        line=dict(color='#ff007f', width=3) 
    ))

    forecast_start_date = forecast_df.index[0]
    fig.add_vline(
        x=forecast_start_date, 
        line_width=2, 
        line_dash="dash", 
        line_color="#00e5ff"
    )

    fig.add_annotation(
        x=forecast_start_date,
        y=1.02,               
        yref="paper",       
        text="Prediction Start",
        showarrow=False,
        font=dict(color="#00e5ff", size=12),
        xanchor="left",
        bgcolor="rgba(8, 10, 15, 0.8)" 
    )

    fig.update_layout(
        title="Future Price Projection (USD)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return apply_chart_theme(fig)