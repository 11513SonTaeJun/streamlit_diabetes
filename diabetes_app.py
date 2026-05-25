import streamlit as st
import pandas as pd
import joblib

# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="당뇨 예측 시스템 2026",
    page_icon="🩺",
    layout="centered"
)

# =========================
# CSS 스타일
# =========================
st.markdown("""
<style>

.main {
    background-color: #0f172a;
    color: white;
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #111827);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 900px;
}

.title-text {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: white;
    margin-bottom: 10px;
}

.sub-text {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 40px;
}

.section-card {
    background-color: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 20px;
    margin-bottom: 25px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(8px);
}

.result-box {
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.good {
    background-color: rgba(34,197,94,0.15);
    border: 1px solid #22c55e;
    color: #4ade80;
}

.bad {
    background-color: rgba(239,68,68,0.15);
    border: 1px solid #ef4444;
    color: #f87171;
}

.prob-text {
    text-align: center;
    font-size: 22px;
    color: white;
    margin-top: 20px;
}

.stSlider label {
    color: white !important;
    font-size: 16px !important;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# =========================
# 모델 불러오기
# =========================
scaler = joblib.load("diabetes_scaler.pkl")
log_model_eng = joblib.load("diabetes.pkl")

# =========================
# 제목
# =========================
st.markdown(
    '<div class="title-text">당뇨 예측 시스템 2026_Made By 11513</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-text">귀하의 건강 정보를 입력하면 프로그램 모델이 당뇨 위험도를 예측합니다.</div>',
    unsafe_allow_html=True
)

# =========================
# 입력 UI
# =========================
with st.container():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        preg = st.slider("임신횟수", 0, 20, 1)

        glucose = st.slider(
            "혈당수치",
            0,
            250,
            120
        )

        bp = st.slider(
            "혈압",
            0,
            150,
            70
        )

        skin = st.slider(
            "피부두께",
            0,
            100,
            20
        )

    with col2:
        insulin = st.slider(
            "인슐린수치",
            0,
            900,
            80
        )

        bmi = st.slider(
            "체질량지수(BMI)",
            0.0,
            60.0,
            25.0
        )

        dpf = st.slider(
            "당뇨병유전력지수",
            0.0,
            3.0,
            0.5
        )

        age = st.slider(
            "나이",
            1,
            100,
            30
        )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 예측 버튼
# =========================
predict_btn = st.button(
    "당뇨 위험도 예측하기",
    use_container_width=True
)

# =========================
# 예측 실행
# =========================
if predict_btn:

    # DataFrame 생성
    input_data = pd.DataFrame(
        [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
        columns=[
            '임신횟수',
            '혈당수치',
            '혈압',
            '피부두께',
            '인슐린수치',
            '체질량지수',
            '당뇨병유전력지수',
            '나이'
        ]
    )

    # =========================
    # 파생 변수 생성
    # =========================
    input_data['대사위험요소'] = (
        input_data['혈당수치']
        + input_data['혈압']
        + input_data['나이']
        + input_data['당뇨병유전력지수']
    )

    input_data['비만혈압지수'] = (
        input_data['체질량지수']
        + input_data['혈압']
    )

    input_data['고위험군점수'] = (
        input_data['나이']
        + input_data['임신횟수']
        + input_data['당뇨병유전력지수']
    )

    # =========================
    # 스케일링
    # =========================
    input_scaled = scaler.transform(input_data)

    # =========================
    # 예측
    # =========================
    predicted = log_model_eng.predict(input_scaled)
    prob = log_model_eng.predict_proba(input_scaled)

    diabetes_prob = prob[0][1] * 100

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # 결과 출력
    # =========================
    if predicted[0] == 1:
        st.markdown(
            f"""
            <div class="result-box bad">
                당뇨 위험군으로 예측됩니다
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="result-box good">
                정상 범위로 예측됩니다
            </div>
            """,
            unsafe_allow_html=True
        )

    # =========================
    # 확률 게이지
    # =========================
    st.markdown(
        f"""
        <div class="prob-text">
            당뇨 확률: <b>{diabetes_prob:.1f}%</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(int(diabetes_prob))

    # =========================
    # 위험도 메시지
    # =========================
    if diabetes_prob < 30:
        st.success("현재 위험도는 낮은 편입니다.")
    elif diabetes_prob < 70:
        st.warning("생활습관 관리가 필요할 수 있습니다.")
    else:
        st.error("의료기관 검진을 권장합니다.")