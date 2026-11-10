import streamlit as st
import numpy as np
import pandas as pd
import joblib

from PIL import Image
from datetime import datetime
from pathlib import Path

# ---------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Plant Disease Detection AI Agent",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Plant Disease Detection AI Agent")

st.write(
    "An intelligent crop health monitoring system "
    "using Machine Learning."
)

st.divider()

# ---------------------------------
# 2. LOAD TRAINED SVM MODEL
# ---------------------------------

@st.cache_resource
def load_model():
    return joblib.load("plant_disease_model.pkl")


try:
    model = load_model()
    st.success("SVM Model Loaded Successfully!")

except Exception as e:
    st.error("Unable to load the trained model.")
    st.write("Check whether plant_disease_model.pkl is uploaded.")
    st.stop()


# ---------------------------------
# 3. TREATMENT RECOMMENDATIONS
# ---------------------------------

def get_treatment(disease):

    disease = disease.lower()

    if "healthy" in disease:

        return [
            "Your plant is classified as healthy.",
            "Provide adequate sunlight and water.",
            "Maintain proper soil nutrition.",
            "Monitor the plant regularly."
        ]

    elif "bacterial" in disease:

        return [
            "Remove severely infected leaves.",
            "Avoid watering directly on leaves.",
            "Clean gardening tools regularly.",
            "Use disease-free seeds.",
            "Consult an agricultural expert for suitable treatment."
        ]

    elif any(word in disease for word in [
        "virus", "mosaic", "yellow_leaf_curl"
    ]):

        return [
            "Separate affected plants where possible.",
            "Check for insects such as aphids and whiteflies.",
            "Remove severely infected plants when infection is confirmed.",
            "Control weeds around the crop.",
            "Use disease-resistant varieties in future cultivation."
        ]

    elif any(word in disease for word in [
        "mite", "spider"
    ]):

        return [
            "Inspect the underside of leaves.",
            "Remove severely affected leaves.",
            "Maintain proper plant moisture.",
            "Protect beneficial insects.",
            "Consult an agricultural expert for suitable treatment."
        ]

    elif any(word in disease for word in [
        "blight", "rust", "mildew", "scab",
        "rot", "spot", "mold", "mould",
        "scorch", "esca"
    ]):

        return [
            "Remove infected leaves and plant debris.",
            "Avoid excessive watering.",
            "Maintain proper spacing between plants.",
            "Improve air circulation.",
            "Use locally approved treatments after confirming the disease."
        ]

    else:

        return [
            "Inspect the plant for visible symptoms.",
            "Remove severely damaged leaves if appropriate.",
            "Maintain proper watering.",
            "Keep the growing area clean.",
            "Consult an agricultural expert for confirmation."
        ]


# ---------------------------------
# 4. IMAGE UPLOAD
# ---------------------------------

st.header("1. Upload Plant Image")

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------
# 5. IMAGE ANALYSIS
# ---------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Plant Image",
        width=350
    )

    if st.button("Analyze Plant Health", type="primary"):

        # Resize image exactly as in ML training
        resized_image = image.resize((32, 32))

        # Convert image into numerical values
        image_array = np.array(resized_image)

        # Flatten image
        features = image_array.flatten()

        # Normalize pixel values
        features = features.astype(np.float32) / 255.0

        # Reshape for SVM prediction
        features = features.reshape(1, -1)

        # Predict disease
        prediction = model.predict(features)

        disease = prediction[0]

        # Separate crop name and disease name
        if "___" in disease:

            crop, disease_name = disease.split("___", 1)

        else:

            crop = "Unknown"
            disease_name = disease

        crop = crop.replace("_", " ")
        disease_name = disease_name.replace("_", " ")

        # ---------------------------------
        # 6. DISPLAY PREDICTION
        # ---------------------------------

        st.divider()

        st.header("2. Plant Health Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Detected Crop", crop)

        with col2:
            st.metric("Predicted Condition", disease_name)

        if "healthy" in disease.lower():

            st.success("The model classified this plant as healthy.")

        else:

            st.warning("Possible plant disease detected.")

        # ---------------------------------
        # 7. TREATMENT RECOMMENDATIONS
        # ---------------------------------

        st.header("3. Treatment Recommendations")

        recommendations = get_treatment(disease)

        for recommendation in recommendations:

            st.write("✅", recommendation)

        # ---------------------------------
        # 8. SAVE CROP HEALTH REPORT
        # ---------------------------------

        report = {

            "Date and Time": datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            ),

            "Image Name": uploaded_file.name,

            "Crop": crop,

            "Predicted Disease": disease_name,

            "Recommendations": " | ".join(recommendations)

        }

        report_file = Path("crop_health_reports.csv")

        new_report = pd.DataFrame([report])

        if report_file.exists():

            old_reports = pd.read_csv(report_file)

            all_reports = pd.concat(
                [old_reports, new_report],
                ignore_index=True
            )

        else:

            all_reports = new_report

        all_reports.to_csv(report_file, index=False)

        st.success("Crop Health Report Saved Successfully!")


# ---------------------------------
# 9. CROP HEALTH MONITORING REPORTS
# ---------------------------------

st.divider()

st.header("4. Crop Health Monitoring Reports")

report_file = Path("crop_health_reports.csv")

if report_file.exists():

    reports = pd.read_csv(report_file)

    st.metric("Total Crop Reports", len(reports))

    st.dataframe(
        reports,
        use_container_width=True
    )

    csv_data = reports.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Crop Health Reports",
        data=csv_data,
        file_name="crop_health_reports.csv",
        mime="text/csv"
    )

else:

    st.info("No crop health reports available yet.")


# ---------------------------------
# 10. FOOTER
# ---------------------------------

st.divider()

st.caption(
    "Plant Disease Detection AI Agent | "
    "Powered by Support Vector Machine (SVM)"
)

st.caption(
    "Note: Predictions are model-based estimates. "
    "Confirm suspected diseases with an agricultural expert "
    "before applying treatments."
)
