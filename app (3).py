import streamlit as st
import numpy as np
import pandas as pd
import joblib
import io

from PIL import Image
from pathlib import Path
from datetime import datetime
from skimage.color import rgb2gray
from skimage.feature import hog


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Plant Disease Detection AI Agent",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Plant Disease Detection AI Agent")

st.write(
    "An intelligent plant health monitoring system "
    "using Machine Learning and Linear Support Vector Machine."
)

st.divider()


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================


MODEL_PATH = "plant_disease_model(2).pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model_bundle = load_model()

    # Load the pipeline from the saved bundle
    model = model_bundle["model"]

    IMAGE_SIZE = model_bundle["image_size"]
    HIST_BINS = model_bundle["hist_bins"]

    st.success("LinearSVC Model Loaded Successfully!")

except Exception as e:
    st.error("Unable to load the trained model.")
    st.write("Make sure plant_disease_model(2).pkl is in your project folder.")
    st.code(str(e))
    st.stop()



# ==========================================
# 3. IMAGE FEATURE EXTRACTION
# ==========================================

def extract_features(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize using the training image dimensions
    resized = image.resize(
        tuple(IMAGE_SIZE),
        Image.Resampling.LANCZOS
    )

    image_array = np.array(resized)

    # Convert RGB image to grayscale
    gray_image = rgb2gray(image_array)

    # Extract HOG features
    hog_features = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    # Extract RGB histogram features
    rgb_features = []

    for channel in range(3):

        histogram, _ = np.histogram(
            image_array[:, :, channel],
            bins=HIST_BINS,
            range=(0, 256)
        )

        # Normalize histogram
        histogram = histogram.astype(np.float32)

        histogram = histogram / (
            histogram.sum() + 1e-7
        )

        rgb_features.extend(histogram)

    # Combine HOG and RGB features
    features = np.concatenate([
        hog_features,
        np.array(rgb_features)
    ])

    return features.reshape(1, -1)


# ==========================================
# 4. CROP AND DISEASE NAME
# ==========================================

def separate_crop_disease(disease):

    if "___" in disease:

        crop, disease_name = disease.split("___", 1)

    else:

        crop_names = [
            "Apple",
            "Blueberry",
            "Cherry",
            "Corn_(maize)",
            "Grape",
            "Orange",
            "Peach",
            "Pepper,_bell",
            "Potato",
            "Raspberry",
            "Soybean",
            "Squash",
            "Strawberry",
            "Tomato"
        ]

        crop = "Unknown"
        disease_name = disease

        for name in sorted(crop_names, key=len, reverse=True):

            if disease.lower().startswith(name.lower() + "_"):

                crop = name
                disease_name = disease[len(name) + 1:]
                break

    crop = crop.replace("_", " ").replace(",", "")
    disease_name = disease_name.replace("_", " ")

    return crop, disease_name


# ==========================================
# 5. TREATMENT RECOMMENDATIONS
# ==========================================

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


# ==========================================
# 6. IMAGE UPLOAD
# ==========================================

st.header("1. Upload Plant Image")

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# 7. IMAGE PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])

    with col1:

        st.image(
            image,
            caption="Uploaded Plant Image",
            use_container_width=True
        )

    with col2:

        st.write("### Image Information")
        st.write("File name:", uploaded_file.name)
        st.write("Image dimensions:", image.size)

    if st.button("Analyze Plant Health", type="primary"):

        try:

            with st.spinner("Analyzing plant image..."):

                # Extract features using training configuration
                features = extract_features(image)

                # Predict disease
                prediction = model.predict(features)

                disease = str(prediction[0])

            # Separate crop and disease
            crop, disease_name = separate_crop_disease(disease)

            st.divider()

            # ==================================
            # 8. DISPLAY PREDICTION
            # ==================================

            st.header("2. Plant Health Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Detected Crop", crop)

            with col2:
                st.metric("Predicted Condition", disease_name)

            if "healthy" in disease.lower():

                st.success(
                    "The model classified this plant as healthy."
                )

            else:

                st.warning(
                    "Possible plant disease detected. "
                    "Please confirm with an agricultural expert."
                )

            # ==================================
            # 9. TOP 5 PREDICTION SCORES
            # ==================================

            st.subheader("3. Top 5 Prediction Scores")

            scores = model.decision_function(features)

            if scores.ndim == 1:

                scores = np.column_stack([
                    -scores,
                    scores
                ])

            class_names = model.classes_

            top_indices = np.argsort(scores[0])[::-1][:5]

            debug_results = pd.DataFrame({

                "Disease Class": class_names[top_indices],

                "SVM Decision Score": scores[0][top_indices]

            })

            st.dataframe(
                debug_results,
                use_container_width=True
            )

            st.caption(
                "SVM decision scores are not probabilities or confidence percentages."
            )

            # ==================================
            # 10. TREATMENT RECOMMENDATIONS
            # ==================================

            st.header("4. Treatment Recommendations")

            recommendations = get_treatment(disease)

            for recommendation in recommendations:

                st.write("✅", recommendation)

            # ==================================
            # 11. SAVE CROP HEALTH REPORT
            # ==================================

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

            all_reports.to_csv(
                report_file,
                index=False
            )

            st.success("Crop Health Report Saved Successfully!")

        except Exception as e:

            st.error("Prediction failed.")

            st.write("Error details:")
            st.exception(e)


# ==========================================
# 12. CROP HEALTH MONITORING REPORTS
# ==========================================

st.divider()

st.header("5. Crop Health Monitoring Reports")

report_file = Path("crop_health_reports.csv")

if report_file.exists():

    reports = pd.read_csv(report_file)

    st.metric(
        "Total Crop Reports",
        len(reports)
    )

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


# ==========================================
# 13. FOOTER
# ==========================================

st.divider()

st.caption(
    "Plant Disease Detection AI Agent | "
    "Powered by Linear Support Vector Machine (LinearSVC)"
)

st.caption(
    "Note: Predictions are model-based estimates. "
    "Confirm suspected diseases with an agricultural expert "
    "before applying treatments."
)
