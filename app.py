import streamlit as st
import numpy as np
import joblib
import random

# ------------------ LOAD ML MODEL ------------------
model = joblib.load("stress_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ------------------ UI CONFIG ------------------
st.set_page_config(page_title="Stress Monitoring System", page_icon="🧠", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #7f7fd5, #86a8e7, #91eae4);
    font-family: 'Arial';
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    width: 420px;
    margin: auto;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
h1 { text-align:center; font-size:40px; color:white; font-weight:bold; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

quotes = [
    "🌿 Relax… one breath at a time.",
    "💪 You are stronger than your stress.",
    "✨ Just breathe, everything will be okay.",
]
st.markdown("<h1>🧠 ML Based Stress Detection</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;font-size:20px;color:white;'>{random.choice(quotes)}</p>", unsafe_allow_html=True)


# ------------------ SpO₂ Evaluation ------------------
def evaluate_oxygen(spo2):
    if spo2 >= 85:
        return "🟢 Normal Oxygen Saturation (85–100%)"
    elif 85 > spo2 >= 75:
        return "🟡 Mild Hypoxia — Deep breathing recommended"
    elif 75 > spo2 >= 65:
        return "🟠 Moderate Hypoxia — Sit and breathe slowly"
    else:
        return "🔴 Severe Hypoxia — Seek medical help immediately!"


# ------------------ ACTIVITY SUGGESTIONS ------------------
def suggestions(stress_type):
    stress_type = stress_type.lower()

    if "high" in stress_type:
        return [
            "🧘 Deep Breathing (4s inhale → 4s hold → 6s exhale)",
            "🎧 Listen to calm music",
            "🚶 Take a short walk"
        ]
    elif "moderate" in stress_type:
        return [
            "☀ Go outside for fresh air",
            "🎯 Do a small relaxing activity",
            "📞 Talk to a friend"
        ]
    else:
        return [
            "✅ You are relaxed",
            "💧 Stay hydrated",
            "🙂 Maintain positive routine"
        ]


# ------------------ UI INPUT ------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📊 Enter Your Readings")

    spo2 = st.number_input("SpO₂ (%)", min_value=40, max_value=100, value=97)
    hr = st.number_input("Heart Rate (BPM)", min_value=30, max_value=200, value=90)

    if st.button("🔍 Predict Stress", use_container_width=True):

        # ML Prediction
        scaled = scaler.transform([[spo2, hr]])
        prediction = model.predict(scaled)[0]
        stress_label = label_encoder.inverse_transform([prediction])[0]

        st.success(f"❤️ Predicted Stress Level (ML): **{stress_label}**")

        # Oxygen evaluation separately
        oxygen_output = evaluate_oxygen(spo2)
        st.info(f"🫁 Oxygen Status: **{oxygen_output}**")

        # Suggestions
        st.subheader("💡 Suggested Activities:")
        for tip in suggestions(stress_label):
            st.write(f"- {tip}")

    st.markdown("</div>", unsafe_allow_html=True)
