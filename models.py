from pandas import DataFrame, Series
from statsmodels.tsa.arima.model import ARIMAResults, ARIMA
from utils import adf_test, get_metrics
from typing import List, Tuple, Literal
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as MAE, mean_absolute_percentage_error as MAPE, mean_squared_error as MSE
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def arima_grid_search(
        train: Series, 
        test: Series,
        d: int, 
        possible_ar: list[int], 
        possible_ma: list[int], 
        log: bool = True,
        criteria: Literal['aic', 'bic', "mae", "mape", "mse"] = "bic",
    ) -> tuple[ARIMAResults, DataFrame, DataFrame]:
    """
    Assumes the time series is stationary.
    Returns the best ARIMA model (fitted) and a DataFrame of model metrics.
    """
    if not adf_test((train if d == 0 else train.diff(d)).dropna()):
        raise ValueError(f"Time series {train.index.name} is not stationary at diff({d})!")

    if log:
        print(possible_ar, possible_ma)
    model_list = []

    if log:
        total_combos = len(possible_ar) * len(possible_ma)
        print(f"Trying {total_combos} combinations...")

    for ar in possible_ar:
        for ma in possible_ma:
            try:
                model = ARIMA(train, order=(ar, d, ma)).fit()
                forecasted = model.forecast(len(test))
                model_list.append((
                    f"ARIMA({ar}, {d}, {ma})", 
                    (ar, d, ma), 
                    model.aic, 
                    model.bic, 
                    MAPE(test, forecasted) * 100, 
                    MAE(test, forecasted), 
                    MSE(test, forecasted),
                ))
            except Exception as e:
                if log:
                    print(f"Failed ARIMA({ar}, {d}, {ma}): {e}")

    if not model_list:
        raise RuntimeError("No ARIMA models converged successfully.")

    model_list_df = DataFrame(
        model_list, columns=["text", "order", "aic", "bic", "mape", "mae", "mse"]
    ).sort_values(criteria)

    if log:
        print(model_list_df)
    best_model_row = model_list_df.iloc[0]
    best_model = ARIMA(train, order=best_model_row.order).fit()
    return best_model, best_model_row, model_list_df

def holt_winters_forecast(
        train : Series, 
        length: int, 
        seasonal_periods: int = 12, 
        trend: Literal['add', 'multiplicative']='add', 
        seasonal: Literal['add', 'multiplicative']='add'
    ):
    model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    model_fit = model.fit()
    forecasted = model_fit.forecast(length)
    return forecasted

def ARIMA_forecast(df : DataFrame, length = 12, order=tuple[int,int,int]) -> None:
    model = ARIMA(df, order=order).fit()
    f = model.forecast(length)
    f.plot(legend=True, label=f"ARIMA{order} Forecast", figsize=(16, 6))
    df.plot(legend=True, label=df.name)
    plt.show()

