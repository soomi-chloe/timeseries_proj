"""
시계열 예측 모듈
ARIMA/AutoARIMA, Holt-Winters, STL, NaiveForecaster
"""
import pandas as pd
import numpy as np
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.trend import STLForecaster
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import (
    mean_absolute_error as MAE,
    mean_squared_error as MSE,
    mean_absolute_percentage_error as MAPE,
)
import warnings
warnings.filterwarnings("ignore")


MODELS = {
    "NaiveForecaster (베이스라인)": "naive",
    "Holt-Winters (지수평활)": "holtwinters",
    "STL 분해법": "stl",
    "ARIMA": "arima",
    "AutoARIMA": "auto_arima",
}


def get_forecasting_horizon(y_test_index, is_relative=False):
    return ForecastingHorizon(y_test_index, is_relative=is_relative)


def build_model(model_key, params=None):
    """모델 이름과 파라미터로 sktime forecaster 생성"""
    if params is None:
        params = {}

    if model_key == "naive":
        strategy = params.get("strategy", "last")
        window = params.get("window_length", 1)
        return NaiveForecaster(strategy=strategy, window_length=window if strategy == "mean" else None)

    elif model_key == "holtwinters":
        trend = params.get("trend", "add")
        seasonal = params.get("seasonal", "mul")
        sp = params.get("sp", 12)
        sl = params.get("smoothing_level", None)
        st = params.get("smoothing_trend", None)
        ss = params.get("smoothing_seasonal", None)
        return ExponentialSmoothing(
            trend=trend,
            seasonal=seasonal,
            sp=sp,
            smoothing_level=sl,
            smoothing_trend=st,
            smoothing_seasonal=ss,
        )

    elif model_key == "stl":
        sp = params.get("sp", 12)
        return STLForecaster(sp=sp)

    elif model_key == "arima":
        p = params.get("p", 1)
        d = params.get("d", 1)
        q = params.get("q", 1)
        P = params.get("P", 0)
        D = params.get("D", 0)
        Q = params.get("Q", 0)
        sp = params.get("sp", 0)
        if sp > 0:
            return ARIMA(order=(p, d, q), seasonal_order=(P, D, Q, sp))
        return ARIMA(order=(p, d, q))

    elif model_key == "auto_arima":
        sp = params.get("sp", 12)
        max_p = params.get("max_p", 3)
        max_q = params.get("max_q", 3)
        return AutoARIMA(
            max_p=max_p, max_q=max_q,
            sp=sp,
            suppress_warnings=True,
            error_action="ignore",
        )

    raise ValueError(f"알 수 없는 모델: {model_key}")


def split_train_test(ts, test_size=0.2):
    """시간 순서 유지 train/test 분리"""
    n = len(ts)
    n_test = max(1, int(n * test_size))
    y_train = ts.iloc[:-n_test]
    y_test = ts.iloc[-n_test:]
    return y_train, y_test


def run_forecast(model_key, y_train, y_test, params=None, confidence_levels=(0.80, 0.95)):
    """
    모델 학습 + 예측 + 신뢰구간
    Returns dict: pred, intervals, metrics, model
    """
    forecaster = build_model(model_key, params)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)

    # 신뢰구간
    intervals = {}
    try:
        for level in confidence_levels:
            alpha = 1 - level
            ci = forecaster.predict_interval(fh, coverage=level)
            lower_col = ci.columns[0]
            upper_col = ci.columns[1]
            intervals[level] = {
                "lower": ci.iloc[:, 0],
                "upper": ci.iloc[:, 1],
            }
    except Exception:
        for level in confidence_levels:
            std_est = float(np.std(y_train.values))
            z = {0.80: 1.28, 0.95: 1.96}.get(level, 1.96)
            intervals[level] = {
                "lower": y_pred - z * std_est,
                "upper": y_pred + z * std_est,
            }

    # 성능 지표
    try:
        mae = float(MAE(y_test, y_pred))
        mse = float(MSE(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        mape = float(MAPE(y_test, y_pred))
    except Exception:
        mae = mse = rmse = mape = float("nan")

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}

    return {
        "pred": y_pred,
        "intervals": intervals,
        "metrics": metrics,
        "model": forecaster,
    }


def forecast_future(forecaster, y_full, horizon):
    """학습 완료된 모델로 미래 예측"""
    fh = ForecastingHorizon(list(range(1, horizon + 1)), is_relative=True)
    forecaster.fit(y_full)
    y_future = forecaster.predict(fh)

    intervals = {}
    try:
        for level in (0.80, 0.95):
            ci = forecaster.predict_interval(fh, coverage=level)
            intervals[level] = {
                "lower": ci.iloc[:, 0],
                "upper": ci.iloc[:, 1],
            }
    except Exception:
        std_est = float(np.std(y_full.values))
        for level in (0.80, 0.95):
            z = {0.80: 1.28, 0.95: 1.96}.get(level, 1.96)
            intervals[level] = {
                "lower": y_future - z * std_est,
                "upper": y_future + z * std_est,
            }

    return y_future, intervals
