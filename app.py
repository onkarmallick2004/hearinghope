# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Hearing Hope – Wellness Recommender", page_icon="🦻")

# --- Load trained pipeline ---
@st.cache_resource
def load_pipeline(path="hearinghope_pipeline.pkl"):
    return joblib.load(path)

pipe = load_pipeline()

# --- Expected columns (same as training X, excluding user_id/visit_date/consent) ---
EXPECTED_COLS_CAT = [
    "clinic_id", "city", "gender", "occupation_noise_risk",
    "hearing_aid_brand", "smoking_status"
]
EXPECTED_COLS_NUM = [
    "age", "pure_tone_avg_db", "speech_recognition_score",
    "hours_noisy_per_day", "hearing_aid_use", "hearing_aid_hours",
    "hearing_difficulty_score", "followup_last_months", "engages_exercises",
    "tinnitus", "prior_ear_infections", "diabetes"
]
EXPECTED_COLS = EXPECTED_COLS_CAT + EXPECTED_COLS_NUM

# --- UI ---
st.title("Hearing Hope – Hearing Wellness Recommendation")
st.markdown("Fill details and get a recommended action (Maintain / UseProtection / ScheduleCheckup / ConsiderHearingAid / FollowExercises).")

col1, col2 = st.columns(2)

with col1:
    clinic_id = st.selectbox("Clinic", ["HH-GK","HH-DW","HH-ND"])
    city = st.selectbox("City", ["New Delhi","Gurugram","Noida","Ghaziabad"])
    gender = st.selectbox("Gender", ["M","F","O"])
    occupation_noise_risk = st.selectbox("Occupation Noise Risk", ["Low","Medium","High"])
    hearing_aid_brand = st.selectbox("Hearing Aid Brand", ["None","Signia","Widex","Phonak","Oticon","Resound"])
    smoking_status = st.selectbox("Smoking Status", ["Never","Former","Current"])

with col2:
    age = st.number_input("Age", min_value=10, max_value=100, value=46)
    pure_tone_avg_db = st.slider("Pure Tone Avg (dB HL)", 0, 110, 28)
    speech_recognition_score = st.slider("Speech Recognition (%)", 0, 100, 70)
    hours_noisy_per_day = st.slider("Hours in noisy env/day", 0.0, 16.0, 2.5)
    hearing_aid_use = st.selectbox("Using Hearing Aid?", [0,1])
    hearing_aid_hours = st.number_input("Hearing Aid Hours/Day", 0, 16, 0)
    hearing_difficulty_score = st.slider("Self-reported Difficulty (0–10)", 0, 10, 4)
    followup_last_months = st.number_input("Months since follow-up", 0, 60, 12)
    engages_exercises = st.selectbox("Doing hearing exercises?", [0,1])
    tinnitus = st.selectbox("Tinnitus?", [0,1])
    prior_ear_infections = st.selectbox("Prior ear infections?", [0,1])
    diabetes = st.selectbox("Diabetes?", [0,1])

# Align input to expected columns and dtypes
def align_row_to_expected(row_dict):
    row = dict(row_dict)
    # Convert "None" brand to actual None
    if row.get("hearing_aid_brand") == "None":
        row["hearing_aid_brand"] = None
    # Build DataFrame with exactly the expected columns
    out = {}
    for c in EXPECTED_COLS:
        if c in row:
            out[c] = row[c]
        else:
            # sensible defaults
            out[c] = None if c in EXPECTED_COLS_CAT else 0
    df = pd.DataFrame([out])
    # Cast numerics
    for c in EXPECTED_COLS_NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[EXPECTED_COLS]

if st.button("Get Recommendation"):
    payload = dict(
        clinic_id=clinic_id, city=city, gender=gender,
        occupation_noise_risk=occupation_noise_risk,
        hearing_aid_brand=hearing_aid_brand,
        smoking_status=smoking_status,
        age=age, pure_tone_avg_db=pure_tone_avg_db,
        speech_recognition_score=speech_recognition_score,
        hours_noisy_per_day=hours_noisy_per_day, hearing_aid_use=hearing_aid_use,
        hearing_aid_hours=hearing_aid_hours, hearing_difficulty_score=hearing_difficulty_score,
        followup_last_months=followup_last_months, engages_exercises=engages_exercises,
        tinnitus=tinnitus, prior_ear_infections=prior_ear_infections, diabetes=diabetes
    )
    X_in = align_row_to_expected(payload)
    pred = pipe.predict(X_in)[0]
    st.success(f"**Recommended Action:** {pred}")
