
import streamlit as st
import numpy as np
import pandas as pd
import joblib

from PIL import Image, ImageOps
from pathlib import Path
from datetime import datetime
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
# 2. MODEL CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "plant_disease_model (2).pkl"

REPORT_PATH = BASE_DIR / "crop_health_reports.csv"


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model(model_path):

    bundle = joblib.load(model_path)

    return bundle


try:

    if not MODEL_PATH.exists():

        st.error("Trained model file not found.")

        st.write("Expected model location:")
        st.code(str(MODEL_PATH))

        st.write("Available files:")
        st.write([p.name for p in BASE_DIR.iterdir()])

        st.stop()

    model_bundle = load_model(str(MODEL_PATH))

    model = model_bundle["model"]

    IMAGE_SIZE = tuple(model_bundle["image_size"])

    HIST_BINS = int(model_bundle["hist_bins"])

    CLASS_NAMES = np.asarray(model.classes_)

    st.success("LinearSVC Model Loaded Successfully!")

except Exception as e:

    st.error("Unable to load the trained model.")

    st.exception(e)

    st.stop()


# ==========================================
# 4. IMAGE FEATURE EXTRACTION
# IMPORTANT: MATCHES TRAINING PREPROCESSING
# ==========================================

def extract_features(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Same resizing method used during training
    resized = ImageOps.fit(
        image,
        IMAGE_SIZE,
        method=Image.Resampling.LANCZOS
    )

    # RGB image
    rgb = np.asarray(
        resized,
        dtype=np.uint8
    )

    # Same grayscale conversion used during training
    gray = np.asarray(
        resized.convert("L"),
        dtype=np.uint8
    )

    # HOG feature extraction
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    # RGB histogram feature extraction
    colour_features = []

    for channel in range(3):

        histogram, _ = np.histogram(
            rgb[:, :, channel],
            bins=HIST_BINS,
            range=(0, 256),
            density=True
        )

        colour_features.extend(
            histogram.astype(np.float32)
        )

    # Combine HOG and RGB features
    features = np.concatenate([
        hog_features.astype(np.float32),
        np.asarray(
            colour_features,
            dtype=np.float32
        )
    ])

    return features.reshape(1, -1)


# ==========================================
# 5. CROP AND DISEASE NAME
# ==========================================

def format_crop_name(crop):

    crop = crop.replace("_", " ").strip()

    crop_lower = crop.lower()

    crop_mapping = {
        "pepper bell": "Bell Pepper",
        "pepper": "Bell Pepper",
        "corn (maize)": "Corn",
        "corn maize": "Corn"
    }

    if crop_lower in crop_mapping:

        return crop_mapping[crop_lower]

    return crop.title()


def separate_crop_disease(disease):

    disease = str(disease).strip()

    if "___" in disease:

        crop, disease_name = disease.split(
            "___",
            1
        )

    else:

        crop = "Unknown"

        disease_name = disease

    crop = format_crop_name(crop)

    disease_name = (
        disease_name
        .replace("_", " ")
        .strip()
        .title()
    )

    return crop, disease_name


# ==========================================
# 6. TREATMENT RECOMMENDATIONS
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
            "Avoid overhead watering.",
            "Avoid working with plants when foliage is wet.",
            "Clean gardening tools regularly.",
            "Use disease-free seeds and healthy seedlings.",
            "Maintain proper spacing between plants.",
            "Consult an agricultural expert about locally approved treatments."
        ]

    elif any(word in disease for word in [
        "virus",
        "mosaic",
        "yellow_leaf_curl"
    ]):

        return [
            "Separate affected plants where possible.",
            "Inspect for insects such as aphids and whiteflies.",
            "Remove severely infected plants when infection is confirmed.",
            "Control weeds around the crop.",
            "Use disease-resistant varieties in future cultivation."
        ]

    elif any(word in disease for word in [
        "mite",
        "spider"
    ]):

        return [
            "Inspect the underside of leaves.",
            "Check for webbing and small moving mites.",
            "Remove severely affected leaves.",
            "Protect beneficial insects.",
            "Consult an agricultural expert for suitable treatment."
        ]

    elif any(word in disease for word in [
        "blight",
        "rust",
        "mildew",
        "scab",
        "rot",
        "spot",
        "mold",
        "mould",
        "scorch",
        "esca"
    ]):

        return [
            "Remove infected leaves and plant debris.",
            "Avoid excessive watering.",
            "Maintain proper spacing between plants.",
            "Improve air circulation.",
            "Avoid wetting foliage unnecessarily.",
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
# 7. DISPLAY MODEL CLASSES
# ==========================================

with st.expander("View supported disease classes"):

    st.write("Classes learned by the trained model:")

    st.write([
        str(class_name)
        for class_name in CLASS_NAMES
    ])


# ==========================================
# 8. IMAGE UPLOAD
# ==========================================

st.header("1. Upload Plant Image")

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# 9. IMAGE PREDICTION
# ==========================================

if uploaded_file is not None:

    try:

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

            st.write(
                "File name:",
                uploaded_file.name
            )

            st.write(
                "Image dimensions:",
                image.size
            )

        if st.button(
            "Analyze Plant Health",
            type="primary"
        ):

            with st.spinner("Analyzing plant image..."):

                # Extract image features
                features = extract_features(image)

                # Verify feature dimensions
                expected_features = (
                    model_bundle["model"]
                    .n_features_in_
                )

                if features.shape[1] != expected_features:

                    st.error(
                        "Image feature dimensions do not match "
                        "the trained model."
                    )

                    st.write(
                        "Extracted features:",
                        features.shape[1]
                    )

                    st.write(
                        "Expected features:",
                        expected_features
                    )

                    st.stop()

                # Predict disease
                prediction = model.predict(features)

                disease = str(prediction[0])

                # Get decision scores
                scores = model.decision_function(features)

            # Separate crop and disease
            crop, disease_name = separate_crop_disease(
                disease
            )

            st.divider()

            # ==================================
            # 10. PLANT HEALTH ANALYSIS
            # ==================================

            st.header("2. Plant Health Analysis")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Detected Crop",
                    crop
                )

            with col2:

                st.metric(
                    "Predicted Condition",
                    disease_name
                )

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
            # 11. TOP 5 PREDICTION SCORES
            # ==================================

            st.header("3. Top 5 Prediction Scores")

            if scores.ndim == 1:

                # Binary LinearSVC
                score_values = np.array([
                    -scores[0],
                    scores[0]
                ])

            else:

                score_values = scores[0]

            # Sort decision scores
            top_indices = np.argsort(
                score_values
            )[::-1][:5]

            debug_results = pd.DataFrame({

                "Disease Class": CLASS_NAMES[top_indices],

                "SVM Decision Score": score_values[top_indices]

            })

            debug_results["SVM Decision Score"] = (
                debug_results["SVM Decision Score"].round(4)
            )

            st.dataframe(
                debug_results,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Decision scores indicate the model's relative "
                "class scores. They are NOT probabilities "
                "or calibrated confidence percentages."
            )

            # ==================================
            # 12. CLASS SUPPORT CHECK
            # ==================================

            st.header("4. Prediction Validation")

            bacterial_spot_classes = [
                str(c)
                for c in CLASS_NAMES
                if "bacterial_spot" in str(c).lower()
            ]

            if bacterial_spot_classes:

                st.success(
                    "The trained model includes these "
                    "Bacterial Spot classes:"
                )

                for class_name in bacterial_spot_classes:

                    st.write("•", class_name)

            else:

                st.error(
                    "Bacterial Spot is not present in "
                    "the trained model's classes."
                )

            # ==================================
            # 13. TREATMENT RECOMMENDATIONS
            # ==================================

            st.header("5. Treatment Recommendations")

            recommendations = get_treatment(disease)

            for recommendation in recommendations:

                st.write("✅", recommendation)

            # ==================================
            # 14. SAVE CROP HEALTH REPORT
            # ==================================

            report = {

                "Date and Time": datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),

                "Image Name": uploaded_file.name,

                "Crop": crop,

                "Predicted Disease": disease_name,

                "Original Model Class": disease,

                "Recommendations": " | ".join(
                    recommendations
                )

            }

            new_report = pd.DataFrame([report])

            if REPORT_PATH.exists():

                old_reports = pd.read_csv(REPORT_PATH)

                all_reports = pd.concat(
                    [old_reports, new_report],
                    ignore_index=True
                )

            else:

                all_reports = new_report

            all_reports.to_csv(
                REPORT_PATH,
                index=False
            )

            st.success(
                "Crop Health Report Saved Successfully!"
            )

    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)


# ==========================================
# 15. CROP HEALTH MONITORING REPORTS
# ==========================================

st.divider()

st.header("6. Crop Health Monitoring Reports")

if REPORT_PATH.exists():

    reports = pd.read_csv(REPORT_PATH)

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

    st.info(
        "No crop health reports available yet."
    )


# ==========================================
# 16. FOOTER
# ==========================================

st.divider()

st.caption(
    "Plant Disease Detection AI Agent | "
    "Powered by Linear Support Vector Machine (LinearSVC)"
)

st.caption(
    "Note: Predictions are model-based estimates. "
    "Confirm suspected diseases with an agricultural "
    "expert before applying treatments."
)
