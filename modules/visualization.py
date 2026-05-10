"""
시각화 모듈 - Plotly 기반
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

COLORS = {
    "actual": "#4F8EF7",
    "pred": "#F7A74F",
    "future": "#4FF7A0",
    "ci80_fill": "rgba(247,167,79,0.15)",
    "ci95_fill": "rgba(247,167,79,0.08)",
    "outlier": "#F74F4F",
    "missing": "#A04FF7",
    "bg": "#0F1117",
    "grid": "#2A2D3E",
    "text": "#E8EAF6",
}

LAYOUT_BASE = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(color=COLORS["text"], family="monospace"),
    xaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
    yaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor=COLORS["grid"], borderwidth=1),
    margin=dict(l=20, r=20, t=50, b=30),
    hovermode="x unified",
)


def _to_datetime_index(series):
    """PeriodIndex → DatetimeIndex 변환"""
    s = series.copy()
    if hasattr(s.index, "to_timestamp"):
        s.index = s.index.to_timestamp()
    return s


def plot_raw_series(ts, title="원본 시계열 데이터"):
    ts = _to_datetime_index(ts)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts.index, y=ts.values,
        mode="lines+markers",
        name="원본 데이터",
        line=dict(color=COLORS["actual"], width=2),
        marker=dict(size=3),
    ))
    fig.update_layout(**LAYOUT_BASE, title=dict(text=title, font=dict(size=16)))
    return fig


def plot_preprocessed(ts_raw, ts_clean, outlier_idx=None, missing_idx=None, title="전처리 결과"):
    ts_raw = _to_datetime_index(ts_raw)
    ts_clean = _to_datetime_index(ts_clean)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_raw.index, y=ts_raw.values,
        mode="lines", name="원본",
        line=dict(color=COLORS["actual"], width=1.5, dash="dot"),
        opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=ts_clean.index, y=ts_clean.values,
        mode="lines", name="전처리 후",
        line=dict(color=COLORS["future"], width=2),
    ))

    # 이상치 마커
    if outlier_idx and len(outlier_idx) > 0:
        try:
            out_ts = ts_raw.copy()
            out_ts_filtered = out_ts[out_ts.index.isin(outlier_idx) | 
                                      out_ts.index.astype(str).isin([str(i) for i in outlier_idx])]
            if len(out_ts_filtered) > 0:
                fig.add_trace(go.Scatter(
                    x=out_ts_filtered.index, y=out_ts_filtered.values,
                    mode="markers", name="이상치",
                    marker=dict(color=COLORS["outlier"], size=10, symbol="x"),
                ))
        except Exception:
            pass

    fig.update_layout(**LAYOUT_BASE, title=dict(text=title, font=dict(size=16)))
    return fig


def plot_forecast(y_train, y_test, y_pred, intervals, outlier_idx=None, title="예측 결과"):
    y_train = _to_datetime_index(y_train)
    y_test = _to_datetime_index(y_test)
    y_pred = _to_datetime_index(y_pred)

    fig = go.Figure()

    # 95% CI
    if 0.95 in intervals:
        lo = _to_datetime_index(intervals[0.95]["lower"])
        hi = _to_datetime_index(intervals[0.95]["upper"])
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series(hi.index), pd.Series(lo.index[::-1])]),
            y=pd.concat([hi, lo[::-1]]),
            fill="toself", fillcolor=COLORS["ci95_fill"],
            line=dict(color="rgba(0,0,0,0)"),
            name="95% 신뢰구간", showlegend=True,
        ))

    # 80% CI
    if 0.80 in intervals:
        lo = _to_datetime_index(intervals[0.80]["lower"])
        hi = _to_datetime_index(intervals[0.80]["upper"])
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series(hi.index), pd.Series(lo.index[::-1])]),
            y=pd.concat([hi, lo[::-1]]),
            fill="toself", fillcolor=COLORS["ci80_fill"],
            line=dict(color="rgba(0,0,0,0)"),
            name="80% 신뢰구간", showlegend=True,
        ))

    # 학습 데이터
    fig.add_trace(go.Scatter(
        x=y_train.index, y=y_train.values,
        mode="lines", name="학습 데이터",
        line=dict(color=COLORS["actual"], width=2),
    ))

    # 실제값 (테스트)
    fig.add_trace(go.Scatter(
        x=y_test.index, y=y_test.values,
        mode="lines+markers", name="실제값 (테스트)",
        line=dict(color="#A0D8EF", width=2),
        marker=dict(size=5),
    ))

    # 예측값
    fig.add_trace(go.Scatter(
        x=y_pred.index, y=y_pred.values,
        mode="lines+markers", name="예측값",
        line=dict(color=COLORS["pred"], width=2.5, dash="dash"),
        marker=dict(size=5, symbol="diamond"),
    ))

    # 이상치
    if outlier_idx and len(outlier_idx) > 0:
        try:
            all_ts = pd.concat([y_train, y_test])
            all_ts = _to_datetime_index(all_ts)
            out_filtered = all_ts[all_ts.index.isin(outlier_idx) |
                                   all_ts.index.astype(str).isin([str(i) for i in outlier_idx])]
            if len(out_filtered) > 0:
                fig.add_trace(go.Scatter(
                    x=out_filtered.index, y=out_filtered.values,
                    mode="markers", name="이상치",
                    marker=dict(color=COLORS["outlier"], size=12, symbol="x-open", line=dict(width=2)),
                ))
        except Exception:
            pass

    # 예측 시작선
    if len(y_test.index) > 0:
        fig.add_vline(x=str(y_test.index[0]), line_dash="dot",
                      line_color="rgba(255,255,255,0.3)", line_width=1)

    fig.update_layout(**LAYOUT_BASE, title=dict(text=title, font=dict(size=16)))
    return fig


def plot_future_forecast(y_full, y_future, intervals, title="미래 예측"):
    y_full = _to_datetime_index(y_full)
    y_future = _to_datetime_index(y_future)

    fig = go.Figure()

    if 0.95 in intervals:
        lo = _to_datetime_index(intervals[0.95]["lower"])
        hi = _to_datetime_index(intervals[0.95]["upper"])
        fig.add_trace(go.Scatter(
            x=list(hi.index) + list(lo.index[::-1]),
            y=list(hi.values) + list(lo.values[::-1]),
            fill="toself", fillcolor=COLORS["ci95_fill"],
            line=dict(color="rgba(0,0,0,0)"),
            name="95% 신뢰구간",
        ))

    if 0.80 in intervals:
        lo = _to_datetime_index(intervals[0.80]["lower"])
        hi = _to_datetime_index(intervals[0.80]["upper"])
        fig.add_trace(go.Scatter(
            x=list(hi.index) + list(lo.index[::-1]),
            y=list(hi.values) + list(lo.values[::-1]),
            fill="toself", fillcolor=COLORS["ci80_fill"],
            line=dict(color="rgba(0,0,0,0)"),
            name="80% 신뢰구간",
        ))

    fig.add_trace(go.Scatter(
        x=y_full.index, y=y_full.values,
        mode="lines", name="과거 데이터",
        line=dict(color=COLORS["actual"], width=2),
    ))

    fig.add_trace(go.Scatter(
        x=y_future.index, y=y_future.values,
        mode="lines+markers", name="미래 예측",
        line=dict(color=COLORS["future"], width=2.5),
        marker=dict(size=6, symbol="diamond"),
    ))

    if len(y_future.index) > 0:
        fig.add_vline(x=str(y_future.index[0]), line_dash="dot",
                      line_color="rgba(255,255,255,0.3)", line_width=1)

    fig.update_layout(**LAYOUT_BASE, title=dict(text=title, font=dict(size=16)))
    return fig


def plot_metrics_bar(metrics_dict):
    """여러 모델 성능 비교 바차트"""
    models = list(metrics_dict.keys())
    metric_names = ["MAE", "RMSE", "MAPE"]

    fig = go.Figure()
    palette = ["#4F8EF7", "#F7A74F", "#4FF7A0", "#F74F4F", "#A04FF7"]

    for i, metric in enumerate(metric_names):
        vals = [metrics_dict[m].get(metric, float("nan")) for m in models]
        fig.add_trace(go.Bar(
            name=metric,
            x=models,
            y=vals,
            marker_color=palette[i % len(palette)],
            text=[f"{v:.4f}" if not np.isnan(v) else "N/A" for v in vals],
            textposition="outside",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        title="모델 성능 비교",
        barmode="group",
        xaxis_tickangle=-30,
    )
    return fig


def plot_residuals(y_test, y_pred, title="잔차 분석"):
    y_test = _to_datetime_index(y_test)
    y_pred = _to_datetime_index(y_pred)
    residuals = y_test.values - y_pred.values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test.index, y=residuals,
        mode="lines+markers", name="잔차",
        line=dict(color=COLORS["pred"], width=1.5),
        marker=dict(size=4),
    ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.4)", line_dash="dash")
    fig.update_layout(**LAYOUT_BASE, title=dict(text=title, font=dict(size=16)))
    return fig
