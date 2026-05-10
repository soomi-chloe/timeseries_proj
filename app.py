"""
시계열 예측 자동화 웹 애플리케이션
기능: 업로드/전처리, 모델선택/파라미터, 대시보드, 내역관리, 다운로드
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# 모듈 경로 추가 (Windows/Mac/Linux 모두 호환)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from modules.preprocessing import (
    load_and_validate_csv, detect_time_column,
    prepare_series, full_preprocess, guess_sp,
)
from modules.forecasting import (
    MODELS, split_train_test, run_forecast, forecast_future, build_model,
)
from modules.visualization import (
    plot_raw_series, plot_preprocessed, plot_forecast,
    plot_future_forecast, plot_metrics_bar, plot_residuals,
)
from modules import history as hist

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TimeSight | 시계열 예측",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 글로벌 CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}
code, .stCode, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* 사이드바 배경 */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D0F1A 0%, #131729 100%);
    border-right: 1px solid #2A2D3E;
}
/* 사이드바 텍스트 전체 밝게 */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #E8EAF6 !important;
}
/* 사이드바 라디오 버튼 텍스트 */
section[data-testid="stSidebar"] .stRadio label p {
    color: #E8EAF6 !important;
    font-size: 0.92rem;
    font-weight: 400;
}
/* 선택된 항목 강조 */
section[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label p {
    color: #4F8EF7 !important;
    font-weight: 600;
}
/* 사이드바 구분선 */
section[data-testid="stSidebar"] hr {
    border-color: #2A2D3E !important;
}

/* 헤더 */
.main-header {
    background: linear-gradient(135deg, #0F1117 0%, #1A1D2E 100%);
    border: 1px solid #2A2D3E;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
}
.main-header h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #4F8EF7;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #8892B0;
    margin: 0;
    font-size: 0.9rem;
}

/* 메트릭 카드 */
.metric-card {
    background: #1A1D2E;
    border: 1px solid #2A2D3E;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.metric-card .label {
    color: #8892B0;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.metric-card .value {
    color: #4F8EF7;
    font-size: 1.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    margin-top: 4px;
}
.metric-card.orange .value { color: #F7A74F; }
.metric-card.green .value  { color: #4FF7A0; }
.metric-card.red .value    { color: #F74F4F; }

/* 섹션 타이틀 */
.section-title {
    font-family: 'JetBrains Mono', monospace;
    color: #4F8EF7;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-left: 3px solid #4F8EF7;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

/* 상태 뱃지 */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}
.badge-ok  { background: rgba(79,247,160,0.15); color: #4FF7A0; border: 1px solid #4FF7A0; }
.badge-warn{ background: rgba(247,167,79,0.15); color: #F7A74F; border: 1px solid #F7A74F; }
.badge-err { background: rgba(247,79,79,0.15);  color: #F74F4F; border: 1px solid #F74F4F; }

/* 히스토리 카드 */
.hist-card {
    background: #1A1D2E;
    border: 1px solid #2A2D3E;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.hist-card:hover { border-color: #4F8EF7; }
.hist-card .hid   { font-family: 'JetBrains Mono', monospace; color: #4F8EF7; font-size: 0.8rem; }
.hist-card .hname { font-weight: 600; color: #E8EAF6; margin: 4px 0 2px; }
.hist-card .hmeta { color: #8892B0; font-size: 0.78rem; }

/* 버튼 커스텀 */
.stButton > button {
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    transition: all 0.2s;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 사이드바 네비게이션
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:12px 0 20px;'>
        <div style='font-family:JetBrains Mono,monospace;font-size:1.3rem;
                    font-weight:600;color:#4F8EF7;'>📈 TimeSight</div>
        <div style='color:#8892B0;font-size:0.78rem;margin-top:4px;'>시계열 예측 자동화</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "메뉴",
        ["① 데이터 업로드 & 전처리",
         "② 모델 선택 & 예측",
         "③ 예측 대시보드",
         "④ 예측 내역 관리"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    # 현재 세션 상태 요약
    st.markdown("<div style='color:#8892B0;font-size:0.75rem;font-family:JetBrains Mono,monospace;'>SESSION STATUS</div>", unsafe_allow_html=True)

    def _badge(ok, label):
        cls = "badge-ok" if ok else "badge-warn"
        st.markdown(f"<span class='badge {cls}'>{label}</span>", unsafe_allow_html=True)

    _badge("ts_clean" in st.session_state, "데이터 로드됨")
    _badge("forecast_result" in st.session_state, "예측 완료")


# ─────────────────────────────────────────────
# ① 데이터 업로드 & 전처리
# ─────────────────────────────────────────────
if page == "① 데이터 업로드 & 전처리":
    st.markdown("""
    <div class='main-header'>
        <h1>📂 데이터 업로드 & 전처리</h1>
        <p>CSV 파일을 업로드하면 자동으로 유효성 검사 및 전처리를 수행합니다.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── 파일 업로드 ──────────────────────────
    uploaded = st.file_uploader(
        "CSV 파일을 업로드하세요 (단변량 시계열)",
        type=["csv"],
        help="시간 컬럼 + 값 컬럼이 포함된 CSV",
    )

    if uploaded:
        df, err = load_and_validate_csv(uploaded)
        if err:
            st.error(f"❌ {err}")
            st.stop()

        st.success(f"✅ 파일 로드 완료 | {df.shape[0]}행 × {df.shape[1]}열")

        # 컬럼 미리보기
        with st.expander("📋 데이터 미리보기", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)

        # ── 컬럼 설정 ──────────────────────────
        st.markdown("<div class='section-title'>컬럼 설정</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        time_candidates = detect_time_column(df)
        with col1:
            time_col = st.selectbox(
                "시간 컬럼",
                df.columns.tolist(),
                index=df.columns.tolist().index(time_candidates[0]) if time_candidates else 0,
            )
        with col2:
            value_candidates = [c for c in df.columns if c != time_col]
            value_col = st.selectbox("값 컬럼", value_candidates)
        with col3:
            freq_options = {"자동 감지": None, "월별(MS)": "MS", "일별(D)": "D",
                            "주별(W)": "W", "분기별(QS)": "QS", "연별(YS)": "YS", "시간별(H)": "H"}
            freq_label = st.selectbox("주기(Frequency)", list(freq_options.keys()))
            freq = freq_options[freq_label]

        # 원본 시계열 시각화
        try:
            ts_raw, freq_used = prepare_series(df, time_col, value_col, freq)
            st.session_state["ts_raw"] = ts_raw
            st.session_state["freq"] = freq_used
            st.plotly_chart(plot_raw_series(ts_raw, "원본 시계열 데이터"), use_container_width=True)
        except Exception as e:
            st.error(f"시계열 변환 오류: {e}")
            st.stop()

        # ── 전처리 설정 ──────────────────────────
        st.markdown("<div class='section-title'>전처리 설정</div>", unsafe_allow_html=True)
        pc1, pc2, pc3 = st.columns(3)

        with pc1:
            impute_method = st.selectbox(
                "결측치 처리 방법",
                ["linear (선형보간)", "locf (직전값)", "nocb (직후값)"],
            ).split(" ")[0]
        with pc2:
            outlier_method = st.selectbox(
                "이상치 탐지 방법",
                ["hampel (Hampel Filter)", "gesd (G-ESD, STL 기반)"],
            ).split(" ")[0]
        with pc3:
            denoise_opts = {"없음": None, "SMA (단순이동평균)": "sma", "EMA (지수이동평균)": "ema"}
            denoise_label = st.selectbox("디노이징", list(denoise_opts.keys()))
            denoise_method = denoise_opts[denoise_label]

        extra_cols = st.columns(3)
        with extra_cols[0]:
            denoise_window = st.slider("SMA 윈도우", 2, 20, 3, disabled=(denoise_method != "sma"))
        with extra_cols[1]:
            denoise_alpha = st.slider("EMA alpha", 0.05, 1.0, 0.3, step=0.05, disabled=(denoise_method != "ema"))
        with extra_cols[2]:
            outlier_replace = st.selectbox("이상치 대체 방법", ["linear", "locf", "nocb"])

        # 전처리 실행
        if st.button("🔧 전처리 실행", type="primary", use_container_width=True):
            with st.spinner("전처리 중..."):
                ts_clean, report = full_preprocess(
                    ts_raw,
                    impute_method=impute_method,
                    outlier_method=outlier_method,
                    outlier_replace_method=outlier_replace,
                    denoise_method=denoise_method,
                    denoise_window=denoise_window,
                    denoise_alpha=denoise_alpha,
                )
                st.session_state["ts_clean"] = ts_clean
                st.session_state["preprocess_report"] = report
                st.session_state["outlier_idx"] = report.get("outlier_idx", [])

            # 결과 요약
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"""<div class='metric-card orange'>
                    <div class='label'>결측치</div>
                    <div class='value'>{report['n_missing']}</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div class='metric-card red'>
                    <div class='label'>이상치 ({report['outlier_method']})</div>
                    <div class='value'>{report['n_outliers']}</div>
                </div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""<div class='metric-card green'>
                    <div class='label'>디노이징</div>
                    <div class='value'>{'ON' if report['denoised'] else 'OFF'}</div>
                </div>""", unsafe_allow_html=True)

            # 전처리 전후 비교
            st.plotly_chart(
                plot_preprocessed(ts_raw, ts_clean,
                                   outlier_idx=report.get("outlier_idx", []),
                                   title="전처리 전후 비교"),
                use_container_width=True,
            )
            st.success("✅ 전처리 완료! '② 모델 선택 & 예측' 탭으로 이동하세요.")

    else:
        # 샘플 데이터 안내
        st.info("👆 CSV 파일을 업로드하거나 아래 샘플 데이터를 사용해보세요.")
        if st.button("📥 샘플 데이터 생성 (airline)"):
            from sktime.datasets import load_airline
            y = load_airline()
            df_sample = pd.DataFrame({
                "date": [str(p) for p in y.index],
                "passengers": y.values,
            })
            csv_buf = io.StringIO()
            df_sample.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ airline.csv 다운로드",
                data=csv_buf.getvalue(),
                file_name="airline.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────────
# ② 모델 선택 & 예측
# ─────────────────────────────────────────────
elif page == "② 모델 선택 & 예측":
    st.markdown("""
    <div class='main-header'>
        <h1>🤖 모델 선택 & 예측</h1>
        <p>예측 모델과 파라미터를 설정하고 예측을 실행합니다.</p>
    </div>
    """, unsafe_allow_html=True)

    if "ts_clean" not in st.session_state:
        st.warning("⚠️ 먼저 '① 데이터 업로드 & 전처리' 탭에서 데이터를 준비하세요.")
        st.stop()

    ts = st.session_state["ts_clean"]
    freq = st.session_state.get("freq", "MS")
    sp_default = guess_sp(ts)

    # ── 모델 선택 ──────────────────────────
    st.markdown("<div class='section-title'>예측 모델 선택</div>", unsafe_allow_html=True)

    model_display = st.selectbox("모델", list(MODELS.keys()))
    model_key = MODELS[model_display]

    # ── 파라미터 설정 ──────────────────────────
    st.markdown("<div class='section-title'>파라미터 설정</div>", unsafe_allow_html=True)
    params = {}

    if model_key == "naive":
        c1, c2 = st.columns(2)
        with c1:
            strategy = st.selectbox("전략", ["last", "mean", "drift"])
            params["strategy"] = strategy
        with c2:
            if strategy == "mean":
                params["window_length"] = st.slider("윈도우 길이", 2, 24, 12)

    elif model_key == "holtwinters":
        c1, c2, c3 = st.columns(3)
        with c1:
            params["trend"] = st.selectbox("추세(trend)", ["add", "mul", None])
            params["sp"] = st.number_input("계절 주기(sp)", 2, 52, sp_default)
        with c2:
            params["seasonal"] = st.selectbox("계절성(seasonal)", ["mul", "add", None])
        with c3:
            use_custom = st.checkbox("평활계수 직접 설정")
            if use_custom:
                params["smoothing_level"] = st.slider("α (level)", 0.01, 1.0, 0.3)
                params["smoothing_trend"] = st.slider("β (trend)", 0.01, 1.0, 0.05)
                params["smoothing_seasonal"] = st.slider("γ (seasonal)", 0.01, 1.0, 0.05)

    elif model_key == "stl":
        params["sp"] = st.slider("계절 주기(sp)", 2, 52, sp_default)

    elif model_key == "arima":
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**비계절 (p, d, q)**")
            params["p"] = st.number_input("p (AR)", 0, 5, 1)
            params["d"] = st.number_input("d (차분)", 0, 2, 1)
            params["q"] = st.number_input("q (MA)", 0, 5, 1)
        with c2:
            use_seasonal = st.checkbox("계절 ARIMA (SARIMA)")
            if use_seasonal:
                st.markdown("**계절 (P, D, Q, sp)**")
                params["P"] = st.number_input("P", 0, 3, 1)
                params["D"] = st.number_input("D", 0, 2, 1)
                params["Q"] = st.number_input("Q", 0, 3, 1)
                params["sp"] = st.number_input("sp", 2, 52, sp_default)
        with c3:
            st.info("💡 ACF/PACF 해석:\n- ACF 느린 감소 → d 증가\n- PACF p 이후 절단 → AR(p)\n- ACF q 이후 절단 → MA(q)")

    elif model_key == "auto_arima":
        c1, c2 = st.columns(2)
        with c1:
            params["max_p"] = st.slider("max_p", 1, 5, 3)
            params["max_q"] = st.slider("max_q", 1, 5, 3)
        with c2:
            params["sp"] = st.number_input("계절 주기(sp)", 1, 52, sp_default)
        st.info("💡 AutoARIMA가 AIC/BIC 기준으로 최적 차수를 자동 탐색합니다.")

    # ── 예측 설정 ──────────────────────────
    st.markdown("<div class='section-title'>예측 설정</div>", unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        test_ratio = st.slider("테스트 데이터 비율", 0.1, 0.4, 0.2, step=0.05,
                                help="전체 데이터 중 모델 평가에 사용할 비율")
    with sc2:
        horizon = st.number_input("미래 예측 기간 (시평)", 1, 60, 12,
                                   help="전처리된 데이터 이후 몇 스텝을 예측할지")

    forecast_name = st.text_input("예측 이름 (선택)", placeholder="예: 2026년 수요 예측")

    # ── 예측 실행 ──────────────────────────
    if st.button("🚀 예측 실행", type="primary", use_container_width=True):
        with st.spinner(f"{model_display} 학습 및 예측 중..."):
            try:
                y_train, y_test = split_train_test(ts, test_size=test_ratio)
                result = run_forecast(model_key, y_train, y_test, params)

                # 미래 예측
                forecaster_for_future = build_model(model_key, params)
                y_future, future_intervals = forecast_future(forecaster_for_future, ts, horizon)

                st.session_state["forecast_result"] = result
                st.session_state["y_train"] = y_train
                st.session_state["y_test"] = y_test
                st.session_state["y_future"] = y_future
                st.session_state["future_intervals"] = future_intervals
                st.session_state["model_display"] = model_display
                st.session_state["model_key"] = model_key
                st.session_state["params"] = params
                st.session_state["horizon"] = horizon

                # 내역 저장
                rec_name = forecast_name if forecast_name else f"{model_display} 예측"
                record_id = hist.save_record(
                    name=rec_name,
                    model_name=model_display,
                    params=params,
                    metrics=result["metrics"],
                    horizon=horizon,
                    freq=freq,
                    n_data=len(ts),
                    preprocess_report=st.session_state.get("preprocess_report", {}),
                )
                st.session_state["last_record_id"] = record_id

                metrics = result["metrics"]
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""<div class='metric-card'>
                        <div class='label'>MAE</div><div class='value'>{metrics['MAE']:.2f}</div></div>""",
                        unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""<div class='metric-card orange'>
                        <div class='label'>RMSE</div><div class='value'>{metrics['RMSE']:.2f}</div></div>""",
                        unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""<div class='metric-card green'>
                        <div class='label'>MAPE</div><div class='value'>{metrics['MAPE']:.4f}</div></div>""",
                        unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""<div class='metric-card red'>
                        <div class='label'>MSE</div><div class='value'>{metrics['MSE']:.2f}</div></div>""",
                        unsafe_allow_html=True)

                st.success(f"✅ 예측 완료! (ID: {record_id}) '③ 예측 대시보드' 탭에서 결과를 확인하세요.")

            except Exception as e:
                st.error(f"❌ 예측 오류: {e}")
                import traceback
                st.code(traceback.format_exc())


# ─────────────────────────────────────────────
# ③ 예측 대시보드
# ─────────────────────────────────────────────
elif page == "③ 예측 대시보드":
    st.markdown("""
    <div class='main-header'>
        <h1>📊 예측 대시보드</h1>
        <p>실제값 vs 예측값, 신뢰구간, 오차 지표, 잔차 분석을 한눈에 확인합니다.</p>
    </div>
    """, unsafe_allow_html=True)

    if "forecast_result" not in st.session_state:
        st.warning("⚠️ 먼저 '② 모델 선택 & 예측' 탭에서 예측을 실행하세요.")
        st.stop()

    result = st.session_state["forecast_result"]
    y_train = st.session_state["y_train"]
    y_test = st.session_state["y_test"]
    y_future = st.session_state["y_future"]
    future_intervals = st.session_state["future_intervals"]
    outlier_idx = st.session_state.get("outlier_idx", [])
    model_display = st.session_state.get("model_display", "모델")
    metrics = result["metrics"]

    # ── 성능 지표 ──────────────────────────
    st.markdown("<div class='section-title'>성능 지표</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    metric_data = [
        ("MAE", metrics["MAE"], "metric-card"),
        ("RMSE", metrics["RMSE"], "metric-card orange"),
        ("MAPE", metrics["MAPE"], "metric-card green"),
        ("MSE", metrics["MSE"], "metric-card red"),
    ]
    for col, (label, val, cls) in zip([m1, m2, m3, m4], metric_data):
        with col:
            st.markdown(f"""<div class='{cls}'>
                <div class='label'>{label}</div>
                <div class='value'>{val:.4f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 예측 결과 차트 ──────────────────────────
    st.markdown("<div class='section-title'>실제값 vs 예측값 (테스트 기간)</div>", unsafe_allow_html=True)
    fig_forecast = plot_forecast(
        y_train, y_test, result["pred"], result["intervals"],
        outlier_idx=outlier_idx,
        title=f"{model_display} — 예측 결과"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # ── 미래 예측 ──────────────────────────
    st.markdown("<div class='section-title'>미래 예측</div>", unsafe_allow_html=True)
    ts_full = st.session_state["ts_clean"]
    fig_future = plot_future_forecast(
        ts_full, y_future, future_intervals,
        title=f"{model_display} — 미래 {st.session_state.get('horizon', '?')}스텝 예측"
    )
    st.plotly_chart(fig_future, use_container_width=True)

    # ── 잔차 분석 ──────────────────────────
    st.markdown("<div class='section-title'>잔차 분석</div>", unsafe_allow_html=True)
    fig_resid = plot_residuals(y_test, result["pred"])
    st.plotly_chart(fig_resid, use_container_width=True)

    # 잔차 통계
    resid_vals = y_test.values - result["pred"].values
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        st.metric("잔차 평균", f"{np.mean(resid_vals):.4f}")
    with rc2:
        st.metric("잔차 표준편차", f"{np.std(resid_vals):.4f}")
    with rc3:
        st.metric("잔차 최솟값", f"{np.min(resid_vals):.4f}")
    with rc4:
        st.metric("잔차 최댓값", f"{np.max(resid_vals):.4f}")

    # ── 다운로드 ──────────────────────────
    st.markdown("<div class='section-title'>결과 데이터 다운로드</div>", unsafe_allow_html=True)

    def make_download_df():
        pred_s = result["pred"].copy()
        if hasattr(pred_s.index, "to_timestamp"):
            pred_s.index = pred_s.index.to_timestamp()
        test_s = y_test.copy()
        if hasattr(test_s.index, "to_timestamp"):
            test_s.index = test_s.index.to_timestamp()

        df_out = pd.DataFrame({
            "date": pred_s.index,
            "actual": test_s.values,
            "predicted": pred_s.values,
            "residual": test_s.values - pred_s.values,
        })
        # 신뢰구간 추가
        for lv in (0.80, 0.95):
            if lv in result["intervals"]:
                lo = result["intervals"][lv]["lower"]
                hi = result["intervals"][lv]["upper"]
                if hasattr(lo.index, "to_timestamp"):
                    lo.index = lo.index.to_timestamp()
                    hi.index = hi.index.to_timestamp()
                df_out[f"ci{int(lv*100)}_lower"] = lo.values
                df_out[f"ci{int(lv*100)}_upper"] = hi.values

        # 미래 예측 추가
        fut_s = y_future.copy()
        if hasattr(fut_s.index, "to_timestamp"):
            fut_s.index = fut_s.index.to_timestamp()
        df_future = pd.DataFrame({"date": fut_s.index, "predicted_future": fut_s.values})

        return df_out, df_future

    df_result, df_future_dl = make_download_df()

    dc1, dc2 = st.columns(2)
    with dc1:
        csv_result = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ 예측 결과 CSV 다운로드",
            data=csv_result,
            file_name=f"forecast_result_{model_display}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dc2:
        csv_future = df_future_dl.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ 미래 예측 CSV 다운로드",
            data=csv_future,
            file_name=f"future_forecast_{model_display}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ─────────────────────────────────────────────
# ④ 예측 내역 관리
# ─────────────────────────────────────────────
elif page == "④ 예측 내역 관리":
    st.markdown("""
    <div class='main-header'>
        <h1>📋 예측 내역 관리</h1>
        <p>과거 예측 내역을 조회하고, 파라미터를 변경하여 재분석할 수 있습니다.</p>
    </div>
    """, unsafe_allow_html=True)

    records = hist.load_all_records()

    if not records:
        st.info("아직 저장된 예측 내역이 없습니다. 예측을 실행하면 자동으로 저장됩니다.")
        st.stop()

    # 전체 삭제
    if st.button("🗑️ 전체 내역 삭제", type="secondary"):
        hist.clear_all()
        st.rerun()

    st.markdown(f"**총 {len(records)}개 예측 내역**")
    st.markdown("---")

    for rec in records:
        with st.container():
            st.markdown(f"""
            <div class='hist-card'>
                <div class='hid'>ID: {rec['id']}</div>
                <div class='hname'>{rec['name']}</div>
                <div class='hmeta'>
                    🕐 {rec['datetime']} &nbsp;|&nbsp;
                    🤖 {rec['model']} &nbsp;|&nbsp;
                    📅 시평: {rec['horizon']} &nbsp;|&nbsp;
                    📊 데이터: {rec['n_data']}개
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 메트릭 표시
            mc = st.columns(4)
            for col, (k, v) in zip(mc, rec.get("metrics", {}).items()):
                with col:
                    st.metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))

            # 파라미터 상세 & 재실행
            with st.expander(f"📂 상세 정보 (ID: {rec['id']})"):
                st.json(rec.get("params", {}))
                ppc = st.columns(2)
                with ppc[0]:
                    st.markdown("**전처리 정보**")
                    pr = rec.get("preprocess_report", {})
                    st.write(f"결측치: {pr.get('n_missing', '-')}개")
                    st.write(f"이상치: {pr.get('n_outliers', '-')}개 ({pr.get('outlier_method', '-')})")
                    st.write(f"디노이징: {'ON' if pr.get('denoised') else 'OFF'}")

                with ppc[1]:
                    # 파라미터 수정 & 재예측 (현재 세션에 데이터가 있을 때만)
                    if "ts_clean" in st.session_state:
                        new_horizon = st.number_input(
                            "재예측 시평", 1, 60, rec["horizon"],
                            key=f"horizon_{rec['id']}"
                        )
                        if st.button(f"🔁 이 설정으로 재예측", key=f"rerun_{rec['id']}"):
                            ts = st.session_state["ts_clean"]
                            model_key = MODELS.get(rec["model"], "naive")
                            params = rec.get("params", {})
                            try:
                                y_train, y_test = split_train_test(ts)
                                result = run_forecast(model_key, y_train, y_test, params)
                                forecaster_f = build_model(model_key, params)
                                y_future, future_intervals = forecast_future(forecaster_f, ts, new_horizon)

                                st.session_state["forecast_result"] = result
                                st.session_state["y_train"] = y_train
                                st.session_state["y_test"] = y_test
                                st.session_state["y_future"] = y_future
                                st.session_state["future_intervals"] = future_intervals
                                st.session_state["model_display"] = rec["model"]
                                st.session_state["model_key"] = model_key
                                st.session_state["params"] = params
                                st.session_state["horizon"] = new_horizon
                                st.success("✅ 재예측 완료! '③ 예측 대시보드' 탭을 확인하세요.")
                            except Exception as e:
                                st.error(f"재예측 오류: {e}")
                    else:
                        st.info("재예측하려면 먼저 데이터를 업로드하세요.")

                if st.button(f"🗑️ 이 내역 삭제", key=f"del_{rec['id']}", type="secondary"):
                    hist.delete_record(rec["id"])
                    st.rerun()

            st.markdown("---")