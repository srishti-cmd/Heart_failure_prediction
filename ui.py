import streamlit as st
import pandas as pd
import pickle
import base64

# Load the model and data
with open("../outputs/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../outputs/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

final_dataset = pd.read_csv("../outputs/final_dataset.csv")

# Load and inject CSS
with open("1.css") as f:
    css = f.read()


# Inject CSS into Streamlit
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)



# UI Title
st.title("🫀 Patient Readmission Predictor")
st.markdown("Enter Patient ID (SUBJECT_ID) to check the 30-day readmission risk.")

# Patient ID input
subject_id_input = st.text_input("Enter Patient ID (numeric):", "")

# On Button Click
if st.button("Check Readmission "):
    if subject_id_input.strip().isdigit():
        subject_id = int(subject_id_input.strip())
        patient_rows = final_dataset[final_dataset["SUBJECT_ID"] == subject_id]

        if not patient_rows.empty:
            st.success(f"✅ Patient found with ID: {subject_id}")
            st.subheader("📄 Latest Admission Details:")
            patient_record = patient_rows.sort_values("HADM_ID", ascending=False).iloc[0]
            st.dataframe(patient_record.to_frame().T)

            input_features = patient_record[feature_names].to_frame().T
            prediction = model.predict_proba(input_features)[0][1]

            st.subheader("📊 30-Day Readmission Risk:")
            st.metric(label="Risk Score", value=f"{prediction:.2%}")

            if prediction < 0.30:
                heart_class = "healthy"
                status_text = "Healthy Heart"
                status_class = "green"
            elif 0.30 <= prediction <= 0.60:
                heart_class = "at-risk"
                status_text = "Heart At Risk"
                status_class = "amber"
            else:
                heart_class = "critical"
                status_text = "Critical Heart"
                status_class = "red"

            # Use CSS to represent the heart status with colored circles or icons
            st.markdown(f'<div class="heart {heart_class}"></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-box"><div class="status {status_class}">{status_text}</div></div>', unsafe_allow_html=True)
           

        else:
            st.error("❌ Patient ID not found in the dataset.")
    else:
        st.warning("⚠️ Please enter a valid numeric Patient ID.")
