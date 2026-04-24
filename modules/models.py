from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_forecast_component
from prophet.diagnostics import cross_validation, performance_metrics
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

def run_arima(series,p, d, q, P, D, Q, s, steps, alpha):



    model = ARIMA(series, order=(p, d, q), seasonal_order=(P, D, Q, s))
    model_fit = model.fit()

    forecast = model_fit.get_forecast(steps=steps)
    forecast_df = forecast.summary_frame(alpha=alpha)

    return forecast_df



def run_prophet_forecast(df, periods, ci_width,
                        changepoint_prior_scale,
                        seasonality_prior_scale,
                        changepoint_range,
                        seasonality_mode,
                        seasonality_settings, 
                        growth='linear'
                        ):
    if growth == 'logistic':
        df['cap'] = df['y'].max() * 2 
        df['floor'] = 0


    model = Prophet(
        growth=growth,
        yearly_seasonality=False, 
        weekly_seasonality=False, 
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode,
        changepoint_range=changepoint_range,
        interval_width=ci_width
    )

    s_set = seasonality_settings
    if s_set['y'] is not None:
        model.add_seasonality(name='yearly', period=365.25, fourier_order=s_set['y'])

    if s_set['m'] is not None:
        model.add_seasonality(name='monthly', period=30.5, fourier_order=s_set['m'])
    
    if s_set['w'] is not None:
        model.add_seasonality(name='weekly', period=7, fourier_order=s_set['w'])

    if s_set['d'] is not None:
        model.add_seasonality(name='daily', period=1, fourier_order=s_set['d'])

    model_fit = model.fit(df)
    future = model_fit.make_future_dataframe(periods=periods, freq='D')

    if growth == 'logistic':
        future['cap'] = df['y'].max() * 2
        future['floor'] = 0

    forecast = model_fit.predict(future)

    forecast = forecast.tail(periods).copy()
    forecast = forecast.rename(columns={
        'ds': 'Date',
        'yhat': 'mean',
        'yhat_lower': 'mean_ci_lower',
        'yhat_upper': 'mean_ci_upper'
    })

    return forecast




def run_stats_auto_arima(series, season_length, horizon, level):
    df_prep = series.reset_index()
    df_prep.columns = ['ds', 'y']
    df_prep['unique_id'] = 'ticker_1' 

    model_obj = AutoARIMA(
        season_length=season_length
    )
    
    sf = StatsForecast(
        models=[model_obj],
        freq='D', 
        n_jobs=-1
    )

    sf.fit(df_prep)
    forecast_df = sf.predict(h=horizon, level=[level])

    
    forecast_df = forecast_df.reset_index()
    forecast_df = forecast_df.rename(columns={
        'AutoARIMA': 'mean',
        f'AutoARIMA-lo-{level}': 'mean_ci_lower',
        f'AutoARIMA-hi-{level}': 'mean_ci_upper'
    })
    
    return forecast_df

