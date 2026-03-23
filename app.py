import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import os

def check_prophet_engine():
    try:
        from prophet import Prophet
        # 더미 모델 생성 시도
        m = Prophet()
        return True
    except Exception as e:
        st.error(f"⚠️ 시스템 엔진 오류: Prophet 라이브러리가 현재 환경과 호환되지 않습니다. \n에러 내용: {e}")
        st.info("해결 방법: 터미널에서 'pip install --upgrade prophet'를 실행하세요.")
        return False

# 메인 로직 시작 전 체크
if not check_prophet_engine():
    st.stop() # 엔진이 없으면 앱 실행 중단

# 페이지 설정
st.set_page_config(page_title="시계열 예측 워크벤치", layout="wide")

st.title("🚀 시계열 데이터 수정 및 자동 예측 엔진")
st.markdown("""
이 앱은 CSV 데이터를 업로드하고, 필요에 따라 데이터를 수정한 뒤 미래 값을 예측합니다.
데이터 형식: 첫 번째 컬럼은 **날짜(Date)**, 두 번째 컬럼은 **수치(Value)**여야 합니다.
""")

# --- 1. 데이터 로드 섹션 ---
st.sidebar.header("📂 데이터 설정")
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=['csv'])

def process_data(df):
    """Prophet 규격에 맞게 데이터 프레임 전처리"""
    try:
        # 컬럼이 2개 이상인지 확인
        if df.shape[1] < 2:
            st.error("CSV 파일은 최소 2개의 컬럼(날짜, 값)을 가져야 합니다.")
            return None
        
        # Prophet 필수 컬럼명으로 변경 (ds: datestamp, y: value)
        new_df = df.iloc[:, [0, 1]].copy()
        new_df.columns = ['ds', 'y']
        new_df['ds'] = pd.to_datetime(new_df['ds'])
        return new_df
    except Exception as e:
        st.error(f"데이터 처리 중 오류 발생: {e}")
        return None

# 데이터가 없을 때 기본 샘플 로드 (Air Passengers)
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    df = process_data(raw_df)
else:
    st.info("테스트를 위해 샘플 데이터를 로드합니다. (Air Passengers)")
    sample_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    try:
        raw_df = pd.read_csv(sample_url)
        df = process_data(raw_df)
    except:
        st.error("샘플 데이터를 불러올 수 없습니다. CSV 파일을 직접 업로드해주세요.")
        df = None

# --- 2. 메인 로직 ---
if df is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. 데이터 편집")
        st.write("표에서 값을 직접 수정할 수 있습니다.")
        # 사용자가 데이터를 수정할 수 있는 인터페이스
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="data_editor")

    with col2:
        st.subheader("2. 예측 결과 시각화")
        
        # 예측 설정
        forecast_periods = st.slider("예측 기간 (개월/일)", 1, 60, 24)
        
        if st.button("✨ 예측 엔진 가동"):
            with st.spinner('Prophet 모델 최적화 중...'):
                try:
                    # 모델 학습 (Prophet)
                    model = Prophet(yearly_seasonality=True, weekly_seasonality='auto', daily_seasonality='auto')
                    model.fit(edited_df)
                    
                    # 미래 날짜 생성 및 예측
                    # 데이터의 빈도(Frequency)를 자동 감지하여 미래 데이터 생성
                    freq = pd.infer_freq(edited_df['ds']) or 'MS' 
                    future = model.make_future_dataframe(periods=forecast_periods, freq=freq)
                    forecast = model.predict(future)

                    # 시각화 (Plotly)
                    fig = go.Figure()

                    # 실제/수정 데이터
                    fig.add_trace(go.Scatter(x=edited_df['ds'], y=edited_df['y'], name='Actual/Edited', mode='lines+markers'))
                    
                    # 예측 데이터
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(dash='dash', color='red')))
                    
                    # 신뢰 구간
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line_color='rgba(0,0,0,0)', fill='tonexty', fillcolor='rgba(255,0,0,0.1)', name='Confidence Interval'))

                    fig.update_layout(hovermode="x unified", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 데이터 다운로드
                    st.download_button("예측 결과 CSV 다운로드", forecast.to_csv(index=False).encode('utf-8'), "forecast_results.csv")
                
                except Exception as e:
                    st.error(f"예측 실패: {e}. 데이터 형식을 확인하세요.")