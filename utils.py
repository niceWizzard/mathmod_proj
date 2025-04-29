import numpy as np
from pandas import DataFrame, Series
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_percentage_error as MAPE, mean_absolute_error as MAE






def plot_autocorrelations(series : Series) -> None:

    if not isinstance(series, Series):
            raise TypeError("Input must be a Pandas Series.")

    if series.name:
        text = f"{series.name} ACF"
    _ = plot_acf(series, lags=48, title=text)
    if series.name:
        text = f"{series.name} PACF"
    _ = plot_pacf(series, lags=48, title=text)



def adf_test(series : Series, log: bool=True) -> bool:
    res = adfuller(series)
    p = res[1]
    is_stationary = p < 0.05
    if log:
        print(f"The {series.name} is {"Stationary"if is_stationary else "Not Stationary"} (p-value: {p:.7f}) | (Statistic: {res[0]:.4f})")
    return is_stationary

def split_fixed(df: DataFrame, f: int=12) -> tuple[DataFrame, DataFrame]:
    n = int(len(df) -  f)  
    train = df.iloc[:n]      
    test = df.iloc[n:]       
    print(f"Split -- Total: {len(df)}, Train: {len(train)}, Test: {len(test)}")
    return train, test

def split_ratio(df: DataFrame, test_ratio : float=0.2) -> tuple[DataFrame, DataFrame]:
    n = int(len(df) * (1 - test_ratio))  
    train = df.iloc[:n]    
    test = df.iloc[n:]       
    print(f"Split -- Total: {len(df)}, Train: {len(train)}, Test: {len(test)}")
    return train, test


def get_metrics(true_values, forecasted) -> DataFrame:
    metrics_result = []
    for key, metric_func in {
        "MAE": MAE,
        "MAPE": MAPE,
        "MSE": MSE,
    }.items():
        metrics_result.append((key, metric_func(true_values, forecasted)))

    return DataFrame(metrics_result, columns=["Metric", "Value"])