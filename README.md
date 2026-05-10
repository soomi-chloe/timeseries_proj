# 📈 TimeSight — 시계열 예측 자동화 웹 애플리케이션

## 구조

```
ts_app/
├── app.py                  # 메인 Streamlit 앱
├── requirements.txt        # 패키지 목록
├── .streamlit/
│   └── config.toml         # 테마 설정
└── modules/
    ├── preprocessing.py    # 결측치/이상치/디노이징 전처리
    ├── forecasting.py      # 예측 모델 (ARIMA, HoltWinters, STL, Naive, AutoARIMA)
    ├── visualization.py    # Plotly 시각화
    └── history.py          # 예측 내역 저장/조회 (JSON)
```

## 설치 및 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 주요 기능

| 탭 | 기능 |
|---|---|
| ① 데이터 업로드 & 전처리 | CSV 업로드, 결측치(LOCF/NOCB/선형보간), 이상치(Hampel/G-ESD), 디노이징(SMA/EMA) |
| ② 모델 선택 & 예측 | NaiveForecaster, Holt-Winters, STL, ARIMA, AutoARIMA, 파라미터 설정, 미래 예측 |
| ③ 예측 대시보드 | 실제 vs 예측 차트, 80%/95% 신뢰구간, MAE/RMSE/MAPE/MSE, 잔차 분석, CSV 다운로드 |
| ④ 예측 내역 관리 | 자동 저장, 상세 조회, 파라미터 수정 후 재예측, 삭제 |

## 테스트용 샘플 데이터

앱 내 "샘플 데이터 생성 (airline)" 버튼으로 airline 데이터셋 CSV를 다운로드 후 업로드하여 테스트할 수 있습니다.
