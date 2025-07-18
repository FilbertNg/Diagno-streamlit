import streamlit as st
from backend_preprocessing import preprocess_input, predict
import pandas as pd

# Page config
st.set_page_config(
    page_title="Diagno – Medical Symptom Diagnostic Chatbot",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Title
st.title("🩺 Diagno: Medical Symptom Diagnostic Chatbot")
st.markdown(
    "Enter your symptoms in free text below — our RNN‑based model will predict the top **4** possible diseases, "
    "complete with descriptions, symptom lists, and treatment suggestions."
)

# User input
user_text = st.text_area(
    "Describe your symptoms:",
    height=150,
    placeholder="e.g. I've had a persistent cough and mild fever for 3 days...",
)

if st.button("Analyze Symptoms"):
    if not user_text.strip():
        st.warning("Please enter some symptoms to analyze.")
    else:
        with st.spinner("Processing…"):
            # Preprocess and predict
            seq = preprocess_input(user_text)
            results = predict(seq)  # returns dict: disease_name → detail dict

        # Display results
        st.success("Top 4 predictions:")
        for i, (disease, detail) in enumerate(results.items(), start=1):
            st.subheader(f"{i}. {disease}")
            sim = float(detail["similarity"])
            st.write(f"**Similarity:** {sim:.1f}%")
            with st.expander("General Description"):
                st.write(detail["description"])
            with st.expander("Common Symptoms"):
                # assuming symptoms are a comma‑separated string
                for sym in detail["symptoms"].split(","):
                    st.write(f"- {sym.strip()}")
            with st.expander("Treatment Suggestions"):
                for tip in detail["solution"].split("."):
                    tip = tip.strip()
                    if tip:
                        st.write(f"- {tip}")

# Footer
st.markdown("---")
st.caption("Built with TensorFlow, NLTK & Tkinter logic, now in Streamlit 🚀")
