"""Spam Message Detection - web app.  Run with:  streamlit run app.py"""
import joblib
import streamlit as st

st.set_page_config(page_title="Spam Message Detection", page_icon="📩")
st.title("📩 Spam Message Detection")
model = joblib.load("models/spam_model.joblib")

msg = st.text_area("Paste a message:", height=150, placeholder="Congratulations! You won a free prize...")
if st.button("Check") and msg.strip():
    label = model.predict([msg])[0]
    clf = model.named_steps["clf"]
    if label:
        st.error("🚨 This looks like SPAM")
    else:
        st.success("✅ This looks safe (HAM)")
    if hasattr(clf, "predict_proba"):
        p = model.predict_proba([msg])[0][1]
        st.progress(float(p), text=f"Spam probability: {p:.1%}")
st.caption("Model: TF-IDF + linear classifier trained on SMS messages.")
