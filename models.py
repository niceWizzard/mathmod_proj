from pandas import DataFrame, Series
from statsmodels.tsa.arima.model import ARIMAResults, ARIMA
from utils import adf_test, get_metrics
from typing import List, Tuple, Literal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error as MAE, mean_absolute_percentage_error as MAPE, mean_squared_error as MSE
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

def arima_grid_search(
        train: Series, 
        test: Series,
        d: int, 
        possible_ar: list[int], 
        possible_ma: list[int], 
        log: bool = True,
        criteria: Literal['aic', 'bic', "mae", "mape", "mse"] = "bic",
        remove_insignificant: bool = False,
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
                all_significant_lags = all(model.pvalues < 0.05)
                if remove_insignificant and not all_significant_lags:
                    if log:
                        print(f"ARIMA({ar}, {d}, {ma}) has insignificant lag.")
                    continue
                
                forecasted = model.forecast(len(test))
                model_list.append((
                    f"ARIMA({ar}, {d}, {ma})", 
                    (ar, d, ma), 
                    model.aic, 
                    model.bic, 
                    MAPE(test, forecasted) * 100, 
                    MAE(test, forecasted), 
                    MSE(test, forecasted),
                    all_significant_lags
                ))
            except Exception as e:
                if log:
                    print(f"Failed ARIMA({ar}, {d}, {ma}): {e}")

    if not model_list:
        raise RuntimeError("No ARIMA models converged successfully.")

    model_list_df = DataFrame(
        model_list, columns=["text", "order", "aic", "bic", "mape", "mae", "mse", "significant"]
    ).sort_values(criteria)

    if log:
        print(model_list_df)
    best_model_row = model_list_df.iloc[0]
    best_model = ARIMA(train, order=best_model_row.order).fit()
    return best_model, best_model_row, model_list_df


def sarima_grid_search(
        train: Series, 
        test: Series,
        d: int, 
        possible_ar: list[int], 
        possible_ma: list[int], 
        possible_P : list[int],
        possible_Q : list[int],
        possible_D : list[int],
        possible_seasonal_periods: list[int],
        log: bool = True,
        criteria: Literal['aic', 'bic', "mae", "mape", "mse"] = "bic",
        remove_insignificant: bool = False,
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
            for P in possible_P:
                for D in possible_D:
                    for Q in possible_Q:
                        for s in possible_seasonal_periods:
                            try:
                                model = SARIMAX(train, order=(ar, d, ma), seasonal_order=(P,D,Q, s)).fit()
                                all_significant_lags = all(model.pvalues < 0.05)
                                if remove_insignificant and not all_significant_lags:
                                    if log:
                                        print(f"SARIMA({ar}, {d}, {ma}) has insignificant lag.")
                                    continue
                                
                                forecasted = model.forecast(len(test))
                                model_list.append((
                                    f"SARIMA({ar}, {d}, {ma})x({P}, {D}, {Q}, {s})", 
                                    (ar, d, ma), 
                                    (P, D, Q, s), 
                                    model.aic, 
                                    model.bic, 
                                    MAPE(test, forecasted) * 100, 
                                    MAE(test, forecasted), 
                                    MSE(test, forecasted),
                                    all_significant_lags
                                ))
                            except Exception as e:
                                if log:
                                    print(f"Failed ARIMA({ar}, {d}, {ma}): {e}")

    if not model_list:
        raise RuntimeError("No ARIMA models converged successfully.")

    model_list_df = DataFrame(
        model_list, columns=["text", "order", "s-order","aic", "bic", "mape", "mae", "mse", "significant"]
    ).sort_values(criteria)

    if log:
        print(model_list_df)
    best_model_row = model_list_df.iloc[0]
    best_model = ARIMA(train, order=best_model_row.order).fit()
    return best_model, best_model_row, model_list_df




def holt_win_search(
        train: Series,
        test: Series,
        seasonal_periods : int = 12,
        criteria: Literal["mae", "mape", "mse"] = "mae",
):
    models = []
    combi = ["multiplicative", "additive"]
    boxcox = [True, False]
    alpha = train.min().iloc[0] + 1
    for t in combi:
        for s in combi:
            for b in boxcox:
                modelMul = ExponentialSmoothing(
                    train + alpha,
                    seasonal_periods=seasonal_periods,
                    trend=t,
                    seasonal=s,
                    use_boxcox=b,
                ).fit()
                f = modelMul.forecast(len(test)) - alpha
                models.append((
                    t, s, b,
                    f"{(MAPE(test, f) * 100):.3f}",
                    MAE(test, f),
                    MSE(test, f),
                ))
    return DataFrame(models, columns=["Trend", "Seasonality", "BoxCox", "MAPE (%)", "MAE", "MSE"]).sort_values(criteria.upper())


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
    f = model.get_forecast(length)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    # Optional: format ticks
    ax.tick_params(axis='x', which='major', length=5)  # major ticks
    ax.tick_params(axis='x', which='minor', length=2)  # minor ticks, no labels
    ax.grid(axis="x", linestyle='--', linewidth=0.5)
    ax.set_xlim(df.index.min(), f.predicted_mean.index.max())
    ax.plot(f.predicted_mean, label=f"ARIMA{order} Forecast")
    ax.set_title("ARIMA Forecast")
    ax.set_ylabel("Inflation Rate (%)")
    ax.set_xlabel("Date")


    conf_int = f.conf_int()
    ax.fill_between(f.predicted_mean.index,
                    conf_int.iloc[:, 0],
                    conf_int.iloc[:, 1],
                    color='skyblue', alpha=0.3, label='95% C.I.',
                    )
    ax.plot(df, label=df.name, color='black', alpha=0.5)
    plt.legend()
    plt.show()

