"""
시계열 데이터 전처리 모듈
결측치 처리, 이상치 탐지/처리, 디노이징
"""
import pandas as pd
import numpy as np
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.outlier_detection import HampelFilter
from sktime.forecasting.trend import STLForecaster
import scikit_posthocs as sp
import warnings
warnings.filterwarnings("ignore")


def load_and_validate_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"파일 읽기 오류: {e}"
    if df.shape[1] < 2:
        return None, "최소 2개 컬럼(시간, 값)이 필요합니다."
    if df.shape[0] < 10:
        return None, "데이터가 너무 적습니다 (최소 10행 이상)."
    return df, None


def detect_time_column(df):
    candidates = []
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True)
            if parsed.notna().sum() > len(df) * 0.8:
                candidates.append(col)
        except Exception:
            pass
    return candidates


def guess_sp(ts):
    try:
        freq = ts.index.freqstr
    except Exception:
        freq = ""
    freq_u = freq.upper().split("-")[0]  # 'Q-DEC' → 'Q' 처리
    if freq_u in ("M", "MS"):
        return 12
    elif freq_u in ("Q", "QS", "A", "AS", "Y", "YS"):
        return 4
    elif freq_u in ("W",):
        return 52
    elif freq_u in ("D",):
        return 7
    elif freq_u in ("H",):
        return 24
    # 부분 매칭 fallback
    if "M" in freq_u:
        return 12
    elif "Q" in freq_u or "Y" in freq_u or "A" in freq_u:
        return 4
    elif "W" in freq_u:
        return 52
    elif "D" in freq_u:
        return 7
    elif "H" in freq_u:
        return 24
    return 12


# DatetimeIndex freq → PeriodIndex freq 변환 테이블
# asfreq()는 Datetime freq 사용, to_period()는 Period freq 사용
_PERIOD_FREQ_MAP = {
    "MS": "M",   "M": "M",
    "QS": "Q",   "Q": "Q",
    "YS": "Y",   "Y": "Y",   "AS": "Y",   "A": "Y",
    "W":  "W",
    "D":  "D",
    "H":  "H",
    "T":  "T",   "min": "T",
    "S":  "S",
}

def _to_period_freq(freq):
    """asfreq용 freq → to_period()용 freq 변환"""
    if freq is None:
        return "M"
    # 접두사/접미사 제거 후 매핑
    base = freq.split("-")[0].upper()
    return _PERIOD_FREQ_MAP.get(base, _PERIOD_FREQ_MAP.get(freq, "M"))


def prepare_series(df, time_col, value_col, freq=None):
    df = df[[time_col, value_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col], infer_datetime_format=True)
    df = df.sort_values(time_col).reset_index(drop=True)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.groupby(time_col, as_index=False)[value_col].mean()
    df = df.set_index(time_col)
    ts = df[value_col]

    # freq 자동 감지
    if freq is None:
        inferred = pd.infer_freq(ts.index)
        freq = inferred if inferred else "MS"

    # DatetimeIndex asfreq (MS, QS, YS 등 그대로 사용 가능)
    ts.index = pd.DatetimeIndex(ts.index)
    ts = ts.asfreq(freq)

    # PeriodIndex 변환 시 Period 호환 freq로 변환 (MS→M, QS→Q 등)
    period_freq = _to_period_freq(freq)
    ts.index = ts.index.to_period(period_freq)

    return ts, freq


def detect_missing(ts):
    n_missing = int(ts.isna().sum())
    missing_idx = ts[ts.isna()].index.tolist()
    return n_missing, missing_idx


def impute_missing(ts, method="linear"):
    method_map = {"locf": "ffill", "nocb": "bfill", "linear": "linear"}
    sktime_method = method_map.get(method, "linear")
    imputer = Imputer(method=sktime_method)
    return imputer.fit_transform(ts)


def detect_outliers_hampel(ts, window_length=5, n_sigma=3):
    try:
        transformer = HampelFilter(window_length=window_length, n_sigma=n_sigma)
        inliers = transformer.fit_transform(ts)
        diff_mask = (ts != inliers) | (inliers.isna() & ts.notna())
        outlier_idx = ts.index[diff_mask].tolist()
        return outlier_idx, inliers
    except Exception:
        return [], ts.copy()


def detect_outliers_gesd(ts, max_outliers=10, alpha=0.05):
    try:
        sp_val = guess_sp(ts)
        forecaster = STLForecaster(sp=sp_val)
        ts_clean = ts.dropna()
        forecaster.fit(ts_clean)
        resid = forecaster.resid_
        inliers = sp.outliers_gesd(resid, outliers=max_outliers, report=False)
        if len(inliers) > 0:
            outlier_vals = resid[(resid < inliers.min()) | (resid > inliers.max())]
            return outlier_vals.index.tolist()
        return []
    except Exception:
        return []


def replace_outliers(ts, outlier_idx, method="linear"):
    ts_copy = ts.copy()
    ts_copy[outlier_idx] = np.nan
    return impute_missing(ts_copy, method=method)


def denoise(ts, method="sma", window=3, alpha=0.3):
    ts_vals = ts.copy()
    if hasattr(ts_vals.index, "to_timestamp"):
        ts_vals.index = ts_vals.index.to_timestamp()
    if method == "sma":
        denoised_vals = ts_vals.rolling(window=window, min_periods=1).mean()
    else:
        denoised_vals = ts_vals.ewm(alpha=alpha).mean()
    denoised_vals.index = ts.index
    return denoised_vals


def full_preprocess(ts, impute_method="linear", outlier_method="hampel",
                    outlier_replace_method="linear", denoise_method=None,
                    denoise_window=3, denoise_alpha=0.3):
    report = {}

    # 1. 결측치
    n_missing, missing_idx = detect_missing(ts)
    report["n_missing"] = n_missing
    report["missing_idx"] = [str(i) for i in missing_idx]
    if n_missing > 0:
        ts = impute_missing(ts, method=impute_method)

    # 2. 이상치
    if outlier_method == "hampel":
        outlier_idx, ts_inlier = detect_outliers_hampel(ts)
        report["outlier_method"] = "Hampel Filter"
        ts = ts_inlier if len(outlier_idx) == 0 else replace_outliers(ts, outlier_idx, outlier_replace_method)
    else:
        outlier_idx = detect_outliers_gesd(ts)
        report["outlier_method"] = "G-ESD (STL 잔차)"
        if len(outlier_idx) > 0:
            ts = replace_outliers(ts, outlier_idx, outlier_replace_method)

    report["n_outliers"] = len(outlier_idx)
    report["outlier_idx"] = [str(i) for i in outlier_idx]

    # 3. 디노이징
    if denoise_method and denoise_method != "없음":
        ts = denoise(ts, method=denoise_method, window=denoise_window, alpha=denoise_alpha)
        report["denoised"] = True
    else:
        report["denoised"] = False

    return ts, report