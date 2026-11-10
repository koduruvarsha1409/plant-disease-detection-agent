# 🌿 Plant Disease Detection AI Agent

## 📌 Project Overview

The Plant Disease Detection AI Agent is an intelligent crop health monitoring system that uses Machine Learning to identify plant diseases from leaf images.

The system uses a trained Support Vector Machine (SVM) model to analyze uploaded plant images, predict possible diseases, recommend general treatment measures, and maintain crop health reports.

The application is developed using Python and Streamlit and deployed through Streamlit Community Cloud.

## 🎯 Objectives

- Detect plant diseases using leaf images.
- Classify plant health using a Machine Learning model.
- Provide suitable general treatment recommendations.
- Maintain crop health reports for future reference.
- Support farmers and users in monitoring crop health.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Machine Learning | Disease classification |
| SVM | Prediction algorithm |
| NumPy | Numerical operations |
| Pandas | Report management |
| Pillow | Image processing |
| Scikit-learn | ML model implementation |
| Joblib | Model loading |
| Streamlit | Web application development |
| GitHub | Source code management |
| Streamlit Community Cloud | Application deployment |

## 🧠 Machine Learning Model

**Algorithm:** Support Vector Machine (SVM)

The model is trained using the PlantVillage dataset, containing more than 41,000 plant leaf images across multiple plant and disease categories.

### Model Workflow

1. Collect plant leaf images.
2. Resize images to 32 × 32 pixels.
3. Convert images into numerical pixel values.
4. Normalize the pixel values.
5. Split the dataset into training and testing sets.
6. Train the SVM classifier.
7. Evaluate model performance.
8. Save the trained model using Joblib.

## ⚙️ Features

### 1. Plant Image Upload
Users can upload plant leaf images in JPG, JPEG, and PNG formats.

### 2. Disease Detection
The trained SVM model analyzes the uploaded image and predicts the corresponding plant disease category.

### 3. Treatment Recommendations
The application provides general treatment and prevention suggestions based on the predicted disease category.

### 4. Crop Health Monitoring
The application records:
- Date and time
- Uploaded image name
- Crop name
- Predicted disease
- Treatment recommendations

### 5. Report Download
Users can view previous crop health reports and download them in CSV format.

## 📂 Project Structure

```text
plant-disease-detection-agent/
│
├── app.py
│
├── plant_disease_model.pkl
│
├── requirements.txt
│
└── README.md
```

## 📦 Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/plant-disease-detection-agent.git
```

### Step 2: Open the project folder

```bash
cd plant-disease-detection-agent
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the application locally

```bash
streamlit run app.py
```

## 📋 Requirements

The following libraries are required:

```text
streamlit
numpy
pandas
Pillow
scikit-learn
joblib
```

For reproducible deployment, use the scikit-learn version used during model training.

## ☁️ Deployment

The application is deployed using Streamlit Community Cloud.

Deployment steps:

1. Upload the project files to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select the project repository.
5. Choose `app.py` as the main application file.
6. Click Deploy.

## 🔄 System Workflow

```text
Plant Leaf Image
       |
       v
Image Upload
       |
       v
Image Preprocessing
       |
       v
Pixel Feature Extraction
       |
       v
Trained SVM Model
       |
       v
Disease Prediction
       |
       v
Treatment Recommendations
       |
       v
Crop Health Report
       |
       v
Report Download
```

## 🚀 Future Enhancements

- Add more plant disease categories.
- Improve prediction accuracy.
- Integrate a permanent cloud database.
- Add multilingual support.
- Provide weather-based crop health insights.
- Develop a mobile-friendly interface.
- Improve recommendations using verified agricultural information.

## ⚠️ Limitations

- Prediction quality depends on the training dataset.
- Images with poor lighting or unclear symptoms may produce incorrect predictions.
- PlantVillage images may differ from real-world field photographs.
- Treatment recommendations are general guidance and should be verified by agricultural experts.
- CSV reports stored on Streamlit Community Cloud's local filesystem are not guaranteed to persist permanently.

## 👩‍💻 Project Information

**Project Title:** Plant Disease Detection AI Agent

**Domain:** Machine Learning and Agriculture

**Algorithm:** Support Vector Machine (SVM)

**Application Framework:** Streamlit

**Dataset:** PlantVillage Dataset

## 📜 Disclaimer

This application is developed for educational and crop-monitoring purposes. Its predictions are not a substitute for professional agricultural diagnosis. Consult an agricultural expert before applying pesticides or other treatments.

---

⭐ If you find this project useful, consider giving the repository a star!
