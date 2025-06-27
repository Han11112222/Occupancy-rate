import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 제목
st.title("📈 도시가스 공급량 예측 시뮬레이터")

# 날짜 선택
start_date = st.date_input("예측 시작일", pd.to_datetime("2022-11-01"))
end_date = st.date_input("예측 종료일", pd.to_datetime("2023-03-31"))

# 엑셀 업로드
uploaded_file = st.file_uploader("엑셀 파일 업로드", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    
    # 필수 컬럼 존재 확인
    required_cols = ['기온추정1', '공급량(MJ)']
    if all(col in df.columns for col in required_cols):
        df = df.dropna(subset=required_cols)
        
        st.subheader("📋 업로드된 데이터 미리보기")
        st.dataframe(df)

        # 모델 학습
        X = df[['기온추정1']]
        y = df['공급량(MJ)']
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)

        # 기온 입력 받기
        st.subheader("🌡️ 기온 입력 후 공급량 예측")
        input_temp = st.slider("기온 (°C)", min_value=-15.0, max_value=25.0, value=5.0, step=0.5)
        input_poly = poly.transform([[input_temp]])
        predicted_supply = model.predict(input_poly)[0]
        st.success(f"예측된 공급량: {predicted_supply:,.0f} MJ")

        # 예측 곡선 시각화
        st.subheader("📊 예측 모델 시각화")
        temp_range = np.linspace(-15, 25, 200).reshape(-1, 1)
        temp_poly = poly.transform(temp_range)
        predicted_curve = model.predict(temp_poly)

        fig, ax = plt.subplots()
        ax.scatter(X, y, label="실제 데이터", alpha=0.6)
        ax.plot(temp_range, predicted_curve, color='red', label="3차 다항 회귀")
        ax.set_xlabel("기온 (°C)")
        ax.set_ylabel("공급량 (MJ)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error(f"필수 컬럼 {required_cols} 이 누락되어 있습니다.")
else:
    st.info("좌측에 엑셀 파일을 업로드해주세요.")

