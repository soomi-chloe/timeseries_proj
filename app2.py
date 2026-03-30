import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import numpy as np

# 1. 시스템 엔진 체크 (비판적 관점: 라이브러리 의존성 문제는 런타임 전에 체크하는 것이 안전함)
def check_prophet_engine():
    try:
        from prophet import Prophet
        return True
    except ImportError:
        st.error("⚠️ Prophet 라이브러리가 설치되지 않았습니다. 'pip install prophet'을 실행하세요.")
        return False

# 2. 열 분류 함수 (현업 수준의 휴리스틱 적용)
def analyze_columns(df):
    """
    개선된 열 분류 함수: 
    데이터 타입뿐만 아니라 실제 변환 가능성을 타진하여 후보를 추출합니다.
    """
    time_candidates = []
    value_candidates = []

    for col in df.columns:
        # 전처리를 위해 샘플 추출 (결측치 제외)
        sample = df[col].dropna().head(100).astype(str)
        
        if sample.empty:
            continue

        # 1. 시간형 후보 탐색
        try:
            pd.to_datetime(sample, errors='raise')
            time_candidates.append(col)
            continue # 시간으로 분류되면 수치형 검사 건너뜀
        except:
            pass

        # 2. 수치형 후보 탐색 (현업 데이터 클렌징 로직 포함)
        # 콤마 제거 후 숫자로 변환 가능한지 확인
        try:
            cleaned_sample = sample.str.replace(',', '').str.replace(' ', '')
            pd.to_numeric(cleaned_sample, errors='raise')
            value_candidates.append(col)
        except:
            # 기존 타입이 이미 숫자라면 추가
            if pd.api.types.is_numeric_dtype(df[col]):
                value_candidates.append(col)
    
    return time_candidates, value_candidates

# --- UI 설정 ---
st.set_page_config(page_title="시계열 분석 워크벤치 Pro", layout="wide")
st.title("📈 시계열 데이터 자동 분석 및 예측")

if not check_prophet_engine():
    st.stop()

# --- 사이드바: 데이터 로드 ---
st.sidebar.header("📁 데이터 업로드")
uploaded_file = st.sidebar.file_uploader("분석할 CSV 파일을 선택하세요", type=['csv'])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    
    # 열 분석 수행
    time_cols, value_cols = analyze_columns(raw_df)

    # --- 1단계: 열 선택 UI ---
    st.subheader("🛠️ 단계 1: 데이터 구조 정의")
    c1, c2 = st.columns(2)
    
    with c1:
        selected_time = st.selectbox("시간(Date/Time) 열을 선택하세요", 
                                    options=time_cols, 
                                    help="Prophet의 'ds' 열이 됩니다.")
    with c2:
        # 시간으로 선택된 열은 수치 후보에서 제외 (논리적 결함 방지)
        remaining_values = [v for v in value_cols if v != selected_time]
        selected_value = st.selectbox("예측 대상(Value) 열을 선택하세요", 
                                     options=remaining_values, 
                                     help="Prophet의 'y' 열이 됩니다.")

    # --- 메인 로직 내 데이터 처리 부분도 아래와 같이 수정 권장 ---
    if selected_time and selected_value:
        df_ready = raw_df[[selected_time, selected_value]].copy()
        
        # [추가] 수치 데이터 강제 클렌징 (콤마 제거 등)
        if not pd.api.types.is_numeric_dtype(df_ready[selected_value]):
            df_ready[selected_value] = df_ready[selected_value].astype(str).str.replace(',', '')
        
        df_ready.columns = ['ds', 'y']
        df_ready['ds'] = pd.to_datetime(df_ready['ds'])
        df_ready['y'] = pd.to_numeric(df_ready['y'], errors='coerce') # 숫자가 아닌 값은 NaN 처리
        
        # 결측치 처리
        df_ready = df_ready.dropna(subset=['ds', 'y']) # 필수값 없는 행 제거

        # --- 2단계: 데이터 편집 및 시각화 ---
        col_edit, col_viz = st.columns([1, 2])
        
        with col_edit:
            st.subheader("📝 데이터 편집")
            edited_df = st.data_editor(df_ready, num_rows="dynamic", use_container_width=True)

        with col_viz:
            st.subheader("🔮 미래 예측")
            forecast_periods = st.slider("예측 기간(단위: 데이터 주기)", 1, 100, 30)
            
            if st.button("🚀 예측 실행"):
                with st.spinner('모델 학습 중...'):
                    try:
                        m = Prophet(changepoint_prior_scale=0.05, daily_seasonality=True)
                        m.fit(edited_df)
                        
                        # 미래 데이터프레임 생성
                        freq = pd.infer_freq(edited_df['ds']) or 'D'
                        future = m.make_future_dataframe(periods=forecast_periods, freq=freq)
                        forecast = m.predict(future)

                        # Plotly 시각화
                        fig = go.Figure()
                        # 실제 데이터
                        fig.add_trace(go.Scatter(x=edited_df['ds'], y=edited_df['y'], name='Original', mode='lines'))
                        # 예측 데이터
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], 
                                                 name='Forecast', line=dict(color='red', dash='dot')))
                        # 범주(Uncertainty)
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], 
                                                 fill=None, mode='lines', line_color='rgba(255,0,0,0)', showlegend=False))
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], 
                                                 fill='tonexty', mode='lines', line_color='rgba(255,0,0,0)', 
                                                 fillcolor='rgba(255,0,0,0.1)', name='Confidence'))

                        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 다운로드
                        csv = forecast.to_csv(index=False).encode('utf-8')
                        st.download_button("결과 다운로드 (CSV)", csv, "forecast.csv", "text/csv")
                        
                    except Exception as e:
                        st.error(f"예측 도중 오류가 발생했습니다: {e}")
else:
    st.info("💡 왼쪽 사이드바에서 CSV 파일을 업로드하여 시작하세요.")