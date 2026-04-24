import pandas as pd
import streamlit as st
from statsmodels.tsa.stattools import adfuller
import numpy as np


def load_validate(file: str):
    try:
        df = pd.read_csv(file)
        date_col = None
        keywords = ['date', 'timestamp']
        for col in df.columns:
            if any(key in col.lower() for key in keywords):
               date_col = col
               break
        if not date_col:
            st.error("Couldn't find a Date or a Timestamp column")
            return None
        
        dtypes = ["int64", 'float64']
        if any(df[date_col].dtype == d for d in dtypes):
            unit = 'ms' if df[date_col].max() > 1e11 else 's'
            df[date_col] = pd.to_datetime(df[date_col], unit=unit)
        else:
            df[date_col] = pd.to_datetime(df[date_col])

        df = df.sort_values(by=date_col)
        df.set_index(date_col, inplace=True)

        df = df.resample('D').last()

        if df.isnull().values.any():
            df = df.interpolate(method='linear')
        
        return df
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None
    

@st.cache_data
def cached_load(file):
    return load_validate(file)

    
def get_target_columns(df: pd.DataFrame):
   
    keywords = ['close', 'open', 'high', 'low', 'price']
    
    target_cols = [col for col in df.columns if any(key in col.lower() for key in keywords)]
    
    return target_cols if target_cols else df.columns.tolist()


def transform_series(series, apply_log, d, D, s):
    if apply_log:
        series = np.log1p(series)
    
    series = apply_sarima_diff(series, d=d, D=D, s=s)
    return series


def test_adf(series:pd.Series):
    series = series.dropna()

    result = adfuller(series, regression='c', autolag='BIC')

    return {
        'statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05 and result[0] < result[4]['5%']
    }


def apply_sarima_diff(series, d=1, D=1, s=12):
    for _ in range(d):
        series = series.diff()
    
    if s > 0 and D > 0:
        for _ in range(D):
            series = series.diff(periods=s)
    
    return series.dropna()


def train_test_split(df, train_ratio):
    last_train_index = int(len(df) * train_ratio)
    train, test = df[:last_train_index], df[last_train_index:]
     
    return train, test


def prepare_prophet_data(data, selected_col):
    if isinstance(data, pd.Series):
        df = data.to_frame(name='y')
    else:
        df = data[[selected_col]].rename(columns={selected_col: 'y'})
    
    prophet_df = df.reset_index()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    return prophet_df








