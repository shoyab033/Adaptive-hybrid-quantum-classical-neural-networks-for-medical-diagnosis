import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from schemas.api_schemas import (
    DatasetMetadata, FeatureInfo, CategoricalOption, PresetSample,
    ModelPerformanceItem, QuantumWorkloadItem, ModelInfoResponse
)

class DataService:
    def __init__(self):
        self.datasets_metadata: Dict[str, DatasetMetadata] = self._init_dataset_metadata()

    def _init_dataset_metadata(self) -> Dict[str, DatasetMetadata]:
        # 1. WDBC Breast Cancer
        wdbc_features = [
            # Group 1: Mean Cell Nuclei Measurements
            FeatureInfo(name="radius_mean", display_name="Radius (Mean)", description="Mean distance from center to points on perimeter", min_value=6.0, max_value=30.0, default_value=14.12, unit="mm", step=0.1, category_group="Mean Cell Measurements"),
            FeatureInfo(name="texture_mean", display_name="Texture (Mean)", description="Standard deviation of gray-scale values", min_value=9.0, max_value=40.0, default_value=19.28, unit="", step=0.1, category_group="Mean Cell Measurements"),
            FeatureInfo(name="perimeter_mean", display_name="Perimeter (Mean)", description="Mean core tumor perimeter", min_value=43.0, max_value=190.0, default_value=91.96, unit="mm", step=0.1, category_group="Mean Cell Measurements"),
            FeatureInfo(name="area_mean", display_name="Area (Mean)", description="Mean core tumor area", min_value=140.0, max_value=2500.0, default_value=654.88, unit="mm²", step=1.0, category_group="Mean Cell Measurements"),
            FeatureInfo(name="smoothness_mean", display_name="Smoothness (Mean)", description="Local variation in radius lengths", min_value=0.05, max_value=0.20, default_value=0.096, unit="", step=0.001, category_group="Mean Cell Measurements"),
            FeatureInfo(name="compactness_mean", display_name="Compactness (Mean)", description="Perimeter² / area - 1.0", min_value=0.01, max_value=0.35, default_value=0.104, unit="", step=0.001, category_group="Mean Cell Measurements"),
            FeatureInfo(name="concavity_mean", display_name="Concavity (Mean)", description="Severity of concave portions of the contour", min_value=0.0, max_value=0.45, default_value=0.088, unit="", step=0.001, category_group="Mean Cell Measurements"),
            FeatureInfo(name="concave_points_mean", display_name="Concave Points (Mean)", description="Number of concave portions of contour", min_value=0.0, max_value=0.25, default_value=0.048, unit="", step=0.001, category_group="Mean Cell Measurements"),
            FeatureInfo(name="symmetry_mean", display_name="Symmetry (Mean)", description="Nuclear symmetry score", min_value=0.10, max_value=0.35, default_value=0.181, unit="", step=0.001, category_group="Mean Cell Measurements"),
            FeatureInfo(name="fractal_dimension_mean", display_name="Fractal Dimension (Mean)", description="Coastline approximation - 1", min_value=0.04, max_value=0.10, default_value=0.062, unit="", step=0.001, category_group="Mean Cell Measurements"),
            
            # Group 2: Standard Error Measurements
            FeatureInfo(name="radius_se", display_name="Radius (SE)", description="Standard error for mean of distances", min_value=0.1, max_value=3.0, default_value=0.405, unit="mm", step=0.01, category_group="Standard Error (SE)"),
            FeatureInfo(name="texture_se", display_name="Texture (SE)", description="Standard error for gray-scale variance", min_value=0.3, max_value=5.0, default_value=1.216, unit="", step=0.01, category_group="Standard Error (SE)"),
            FeatureInfo(name="perimeter_se", display_name="Perimeter (SE)", description="Standard error for perimeter", min_value=0.7, max_value=22.0, default_value=2.866, unit="mm", step=0.01, category_group="Standard Error (SE)"),
            FeatureInfo(name="area_se", display_name="Area (SE)", description="Standard error for area", min_value=6.0, max_value=550.0, default_value=40.33, unit="mm²", step=0.1, category_group="Standard Error (SE)"),
            FeatureInfo(name="smoothness_se", display_name="Smoothness (SE)", description="Standard error for local radius variation", min_value=0.001, max_value=0.03, default_value=0.007, unit="", step=0.0001, category_group="Standard Error (SE)"),
            FeatureInfo(name="compactness_se", display_name="Compactness (SE)", description="Standard error for compactness", min_value=0.002, max_value=0.15, default_value=0.025, unit="", step=0.001, category_group="Standard Error (SE)"),
            FeatureInfo(name="concavity_se", display_name="Concavity (SE)", description="Standard error for concavity", min_value=0.0, max_value=0.40, default_value=0.031, unit="", step=0.001, category_group="Standard Error (SE)"),
            FeatureInfo(name="concave_points_se", display_name="Concave Points (SE)", description="Standard error for concave points", min_value=0.0, max_value=0.06, default_value=0.011, unit="", step=0.0001, category_group="Standard Error (SE)"),
            FeatureInfo(name="symmetry_se", display_name="Symmetry (SE)", description="Standard error for symmetry", min_value=0.007, max_value=0.08, default_value=0.020, unit="", step=0.001, category_group="Standard Error (SE)"),
            FeatureInfo(name="fractal_dimension_se", display_name="Fractal Dimension (SE)", description="Standard error for fractal dimension", min_value=0.0008, max_value=0.03, default_value=0.0037, unit="", step=0.0001, category_group="Standard Error (SE)"),
            
            # Group 3: Extreme / Worst Measurements
            FeatureInfo(name="radius_worst", display_name="Radius (Worst)", description="Largest nuclear radius observed", min_value=7.0, max_value=40.0, default_value=16.26, unit="mm", step=0.1, category_group="Worst / Extreme Measurements"),
            FeatureInfo(name="texture_worst", display_name="Texture (Worst)", description="Largest texture variation observed", min_value=12.0, max_value=50.0, default_value=25.67, unit="", step=0.1, category_group="Worst / Extreme Measurements"),
            FeatureInfo(name="perimeter_worst", display_name="Perimeter (Worst)", description="Largest core perimeter observed", min_value=50.0, max_value=260.0, default_value=107.26, unit="mm", step=0.1, category_group="Worst / Extreme Measurements"),
            FeatureInfo(name="area_worst", display_name="Area (Worst)", description="Largest core area observed", min_value=180.0, max_value=4300.0, default_value=880.58, unit="mm²", step=1.0, category_group="Worst / Extreme Measurements"),
            FeatureInfo(name="smoothness_worst", display_name="Smoothness (Worst)", description="Largest smoothness variation", min_value=0.07, max_value=0.25, default_value=0.132, unit="", step=0.001, category_group="Worst / Extreme Measurements"),
            FeatureInfo(name="compactness_worst", display_name="Compactness (Worst)", description="Largest compactness value", min_value=0.02, max_value=1.1, default_value=0.254, unit="", step=0.001, category_group="Worst / Extreme Measurements"),
            FeatureInfo(name="concavity_worst", display_name="Concavity (Worst)", description="Largest concavity value", min_value=0.0, max_value=1.3, default_value=0.272, unit="", step=0.001, category_group="Worst / Extreme Measurements"),
            FeatureInfo(name="concave_points_worst", display_name="Concave Points (Worst)", description="Largest concave points count", min_value=0.0, max_value=0.30, default_value=0.114, unit="", step=0.001, category_group="Worst / Extreme Measurements"),
            FeatureInfo(name="symmetry_worst", display_name="Symmetry (Worst)", description="Largest nuclear symmetry value", min_value=0.15, max_value=0.70, default_value=0.290, unit="", step=0.001, category_group="Worst / Extreme Measurements"),
            FeatureInfo(name="fractal_dimension_worst", display_name="Fractal Dimension (Worst)", description="Largest fractal dimension value", min_value=0.05, max_value=0.21, default_value=0.083, unit="", step=0.001, category_group="Worst / Extreme Measurements"),
        ]

        wdbc_presets = [
            PresetSample(
                id="wdbc_benign_path0",
                name="Clear Benign Case (High Confidence)",
                description="Typical non-malignant presentation with low radius and concave points. Confidently resolved via Path 0 (0 Q-Layers).",
                expected_path="Path 0 (Classical Only)",
                values={
                    "radius_mean": 11.42, "texture_mean": 20.38, "perimeter_mean": 77.58, "area_mean": 386.1, "smoothness_mean": 0.1425,
                    "compactness_mean": 0.2839, "concavity_mean": 0.0241, "concave_points_mean": 0.0259, "symmetry_mean": 0.1812, "fractal_dimension_mean": 0.0566,
                    "radius_se": 0.184, "texture_se": 0.892, "perimeter_se": 1.25, "area_se": 14.5, "smoothness_se": 0.0058,
                    "compactness_se": 0.0142, "concavity_se": 0.0125, "concave_points_se": 0.0062, "symmetry_se": 0.0154, "fractal_dimension_se": 0.0021,
                    "radius_worst": 12.56, "texture_worst": 26.50, "perimeter_worst": 84.46, "area_worst": 467.2, "smoothness_worst": 0.1199,
                    "compactness_worst": 0.1486, "concavity_worst": 0.0712, "concave_points_worst": 0.0558, "symmetry_worst": 0.2471, "fractal_dimension_worst": 0.0712
                }
            ),
            PresetSample(
                id="wdbc_malignant_path0",
                name="Clear Malignant Case (High Confidence)",
                description="Severe nuclear atypia with high perimeter and large concave points. Confidently resolved via Path 0 (0 Q-Layers).",
                expected_path="Path 0 (Classical Only)",
                values={
                    "radius_mean": 19.69, "texture_mean": 21.25, "perimeter_mean": 130.0, "area_mean": 1203.0, "smoothness_mean": 0.1096,
                    "compactness_mean": 0.1599, "concavity_mean": 0.1974, "concave_points_mean": 0.1279, "symmetry_mean": 0.2069, "fractal_dimension_mean": 0.0599,
                    "radius_se": 0.745, "texture_se": 0.786, "perimeter_se": 4.58, "area_se": 94.0, "smoothness_se": 0.0061,
                    "compactness_se": 0.0400, "concavity_se": 0.0547, "concave_points_se": 0.0158, "symmetry_se": 0.0267, "fractal_dimension_se": 0.0045,
                    "radius_worst": 23.57, "texture_worst": 25.53, "perimeter_worst": 152.5, "area_worst": 1709.0, "smoothness_worst": 0.1444,
                    "compactness_worst": 0.4245, "concavity_worst": 0.4504, "concave_points_worst": 0.2430, "symmetry_worst": 0.3613, "fractal_dimension_worst": 0.0875
                }
            ),
            PresetSample(
                id="wdbc_borderline_path1",
                name="Borderline Atypical Case (Intermediate Uncertainty)",
                description="Intermediate concave points and mixed morphology triggering Quantum Escalation Path 1 (1 Q-Layer).",
                expected_path="Path 1 or Path 2",
                values={
                    "radius_mean": 13.54, "texture_mean": 14.36, "perimeter_mean": 87.46, "area_mean": 566.3, "smoothness_mean": 0.0977,
                    "compactness_mean": 0.0812, "concavity_mean": 0.0666, "concave_points_mean": 0.0478, "symmetry_mean": 0.1885, "fractal_dimension_mean": 0.0576,
                    "radius_se": 0.269, "texture_se": 0.788, "perimeter_se": 2.05, "area_se": 23.5, "smoothness_se": 0.0044,
                    "compactness_se": 0.0138, "concavity_se": 0.0198, "concave_points_se": 0.0101, "symmetry_se": 0.0131, "fractal_dimension_se": 0.0023,
                    "radius_worst": 15.11, "texture_worst": 19.26, "perimeter_worst": 99.70, "area_worst": 711.2, "smoothness_worst": 0.1440,
                    "compactness_worst": 0.1773, "concavity_worst": 0.2390, "concave_points_worst": 0.1288, "symmetry_worst": 0.2977, "fractal_dimension_worst": 0.0725
                }
            )
        ]

        # 2. Cleveland Heart Disease
        heart_features = [
            FeatureInfo(
                name="age", display_name="Patient Age", description="Age in years",
                min_value=25.0, max_value=85.0, default_value=54.0, unit="years", step=1.0,
                category_group="Demographics & History"
            ),
            FeatureInfo(
                name="sex", display_name="Biological Sex", description="Patient biological sex",
                data_type="categorical", min_value=0.0, max_value=1.0, default_value=1.0, unit="", step=1.0,
                category_group="Demographics & History",
                options=[
                    CategoricalOption(value=1.0, label="Male", description="Male patient"),
                    CategoricalOption(value=0.0, label="Female", description="Female patient"),
                ]
            ),
            FeatureInfo(
                name="cp", display_name="Chest Pain Type", description="Type of chest pain experienced",
                data_type="categorical", min_value=1.0, max_value=4.0, default_value=4.0, unit="", step=1.0,
                category_group="Symptoms & Presentation",
                options=[
                    CategoricalOption(value=1.0, label="Typical Angina", description="Substernal chest pressure or squeezing"),
                    CategoricalOption(value=2.0, label="Atypical Angina", description="Non-typical chest discomfort"),
                    CategoricalOption(value=3.0, label="Non-Anginal Pain", description="Pain unrelated to myocardial ischemia"),
                    CategoricalOption(value=4.0, label="Asymptomatic", description="No primary angina reported"),
                ]
            ),
            FeatureInfo(
                name="trestbps", display_name="Resting Blood Pressure", description="Resting blood pressure upon clinical intake",
                min_value=90.0, max_value=210.0, default_value=131.0, unit="mm Hg", step=1.0,
                category_group="Vital Signs & Labs"
            ),
            FeatureInfo(
                name="chol", display_name="Serum Cholesterol", description="Total fasting serum cholesterol",
                min_value=120.0, max_value=580.0, default_value=246.0, unit="mg/dl", step=1.0,
                category_group="Vital Signs & Labs"
            ),
            FeatureInfo(
                name="fbs", display_name="Fasting Blood Sugar > 120 mg/dl", description="Blood sugar > 120 mg/dl indicator",
                data_type="categorical", min_value=0.0, max_value=1.0, default_value=0.0, unit="", step=1.0,
                category_group="Vital Signs & Labs",
                options=[
                    CategoricalOption(value=0.0, label="Normal (≤ 120 mg/dl)", description="Fasting glucose is normal"),
                    CategoricalOption(value=1.0, label="Elevated (> 120 mg/dl)", description="Fasting glucose is elevated"),
                ]
            ),
            FeatureInfo(
                name="restecg", display_name="Resting ECG Results", description="Baseline electrocardiogram results",
                data_type="categorical", min_value=0.0, max_value=2.0, default_value=1.0, unit="", step=1.0,
                category_group="Diagnostic Testing",
                options=[
                    CategoricalOption(value=0.0, label="Normal", description="Normal baseline tracing"),
                    CategoricalOption(value=1.0, label="ST-T Wave Abnormality", description="T-wave inversions or ST depression/elevation"),
                    CategoricalOption(value=2.0, label="Left Ventricular Hypertrophy", description="Estes criteria positive"),
                ]
            ),
            FeatureInfo(
                name="thalach", display_name="Maximum Heart Rate (Stress Test)", description="Max heart rate achieved during exercise stress testing",
                min_value=70.0, max_value=210.0, default_value=149.0, unit="bpm", step=1.0,
                category_group="Stress Exercise Test"
            ),
            FeatureInfo(
                name="exang", display_name="Exercise-Induced Angina", description="Chest pain triggered by exertion",
                data_type="categorical", min_value=0.0, max_value=1.0, default_value=0.0, unit="", step=1.0,
                category_group="Stress Exercise Test",
                options=[
                    CategoricalOption(value=0.0, label="No", description="No angina during stress test"),
                    CategoricalOption(value=1.0, label="Yes", description="Angina provoked during exertion"),
                ]
            ),
            FeatureInfo(
                name="oldpeak", display_name="ST Depression (Oldpeak)", description="ST depression induced by exercise relative to rest",
                min_value=0.0, max_value=6.5, default_value=1.0, unit="mm", step=0.1,
                category_group="Stress Exercise Test"
            ),
            FeatureInfo(
                name="slope", display_name="Slope of Peak Exercise ST", description="Slope of the peak exercise ST segment",
                data_type="categorical", min_value=1.0, max_value=3.0, default_value=2.0, unit="", step=1.0,
                category_group="Stress Exercise Test",
                options=[
                    CategoricalOption(value=1.0, label="Upsloping", description="Normal upsloping ST response"),
                    CategoricalOption(value=2.0, label="Flat", description="Horizontal / flat ST depression"),
                    CategoricalOption(value=3.0, label="Downsloping", description="Downsloping ST segment"),
                ]
            ),
            FeatureInfo(
                name="ca", display_name="Major Vessels Colored (Fluoroscopy)", description="Number of major coronary vessels (0-3) colored",
                data_type="categorical", min_value=0.0, max_value=3.0, default_value=0.0, unit="", step=1.0,
                category_group="Fluoroscopy & Imaging",
                options=[
                    CategoricalOption(value=0.0, label="0 Vessels", description="No major vessel stenosis"),
                    CategoricalOption(value=1.0, label="1 Vessel", description="1 major vessel colored"),
                    CategoricalOption(value=2.0, label="2 Vessels", description="2 major vessels colored"),
                    CategoricalOption(value=3.0, label="3 Vessels", description="3 major vessels colored"),
                ]
            ),
            FeatureInfo(
                name="thal", display_name="Thallium Scintigraphy Result", description="Myocardial perfusion status",
                data_type="categorical", min_value=3.0, max_value=7.0, default_value=3.0, unit="", step=1.0,
                category_group="Fluoroscopy & Imaging",
                options=[
                    CategoricalOption(value=3.0, label="Normal Perfusion (3)", description="Uniform myocardial tracer uptake"),
                    CategoricalOption(value=6.0, label="Fixed Defect (6)", description="Non-reversible cold spot / scar tissue"),
                    CategoricalOption(value=7.0, label="Reversible Defect (7)", description="Stress-induced reversible ischemia"),
                ]
            )
        ]

        heart_presets = [
            PresetSample(
                id="heart_healthy_path0",
                name="Clear Normal Cardiac Profile",
                description="Younger patient, asymptomatic/typical chest pain, high max heart rate (175 bpm), 0 ST depression. Resolved via Path 0 (0 QL).",
                expected_path="Path 0 (Classical Only)",
                values={"age": 42.0, "sex": 0.0, "cp": 2.0, "trestbps": 120.0, "chol": 198.0, "fbs": 0.0, "restecg": 0.0, "thalach": 175.0, "exang": 0.0, "oldpeak": 0.0, "slope": 1.0, "ca": 0.0, "thal": 3.0}
            ),
            PresetSample(
                id="heart_severe_path0",
                name="High-Risk CAD Profile",
                description="Elderly male, asymptomatic chest pain, high ST depression (3.6 mm), 3 colored vessels, reversible defect. Resolved via Path 0.",
                expected_path="Path 0 (Classical Only)",
                values={"age": 67.0, "sex": 1.0, "cp": 4.0, "trestbps": 160.0, "chol": 286.0, "fbs": 1.0, "restecg": 2.0, "thalach": 108.0, "exang": 1.0, "oldpeak": 3.6, "slope": 2.0, "ca": 3.0, "thal": 7.0}
            ),
            PresetSample(
                id="heart_uncertain_path1",
                name="Ambiguous Cardiac Presentation",
                description="Middle-aged patient with intermediate ST depression (1.4 mm) and flat slope triggering Quantum Path 1.",
                expected_path="Path 1 or Path 2",
                values={"age": 56.0, "sex": 1.0, "cp": 3.0, "trestbps": 130.0, "chol": 256.0, "fbs": 0.0, "restecg": 1.0, "thalach": 142.0, "exang": 1.0, "oldpeak": 1.4, "slope": 2.0, "ca": 1.0, "thal": 6.0}
            )
        ]

        # 3. Pima Indians Diabetes
        diabetes_features = [
            FeatureInfo(
                name="Age", display_name="Patient Age", description="Patient age in years",
                min_value=21.0, max_value=81.0, default_value=33.0, unit="years", step=1.0,
                category_group="Demographics"
            ),
            FeatureInfo(
                name="Pregnancies", display_name="Number of Pregnancies", description="Number of times pregnant",
                min_value=0.0, max_value=17.0, default_value=3.0, unit="", step=1.0,
                category_group="Obstetric & Family History"
            ),
            FeatureInfo(
                name="DiabetesPedigreeFunction", display_name="Diabetes Pedigree Function", description="Genetic history score based on diabetic family pedigree",
                min_value=0.07, max_value=2.45, default_value=0.47, unit="", step=0.01,
                category_group="Obstetric & Family History"
            ),
            FeatureInfo(
                name="Glucose", display_name="Fasting Plasma Glucose (2-hr OGTT)", description="Plasma glucose concentration at 2 hours in oral glucose tolerance test",
                min_value=50.0, max_value=200.0, default_value=120.0, unit="mg/dL", step=1.0,
                category_group="Metabolic & Lab Measurements"
            ),
            FeatureInfo(
                name="Insulin", display_name="2-Hour Serum Insulin", description="2-Hour post-load serum insulin concentration",
                min_value=14.0, max_value=846.0, default_value=125.0, unit="mu U/ml", step=1.0,
                category_group="Metabolic & Lab Measurements"
            ),
            FeatureInfo(
                name="BloodPressure", display_name="Diastolic Blood Pressure", description="Diastolic blood pressure reading",
                min_value=40.0, max_value=125.0, default_value=72.0, unit="mm Hg", step=1.0,
                category_group="Vital Signs & Body Metrics"
            ),
            FeatureInfo(
                name="BMI", display_name="Body Mass Index (BMI)", description="Weight in kg / (height in meters)²",
                min_value=18.0, max_value=67.0, default_value=32.0, unit="kg/m²", step=0.1,
                category_group="Vital Signs & Body Metrics"
            ),
            FeatureInfo(
                name="SkinThickness", display_name="Triceps Skinfold Thickness", description="Triceps skinfold measurement for subcutaneous body fat",
                min_value=7.0, max_value=99.0, default_value=29.0, unit="mm", step=1.0,
                category_group="Vital Signs & Body Metrics"
            ),
        ]

        diabetes_presets = [
            PresetSample(
                id="diabetes_lowrisk_path0",
                name="Clear Non-Diabetic Presentation",
                description="Low plasma glucose (85 mg/dL), normal BMI (22.5), low pedigree score. Confidently resolved via Path 0 (0 QL).",
                expected_path="Path 0 (Classical Only)",
                values={"Pregnancies": 1.0, "Glucose": 85.0, "BloodPressure": 66.0, "SkinThickness": 29.0, "Insulin": 125.0, "BMI": 22.5, "DiabetesPedigreeFunction": 0.23, "Age": 24.0}
            ),
            PresetSample(
                id="diabetes_highrisk_path0",
                name="Confirmed Diabetic Presentation",
                description="High glucose (180 mg/dL), elevated BMI (39.2), multiple pregnancies. Confidently resolved via Path 0.",
                expected_path="Path 0 (Classical Only)",
                values={"Pregnancies": 6.0, "Glucose": 180.0, "BloodPressure": 84.0, "SkinThickness": 38.0, "Insulin": 240.0, "BMI": 39.2, "DiabetesPedigreeFunction": 0.95, "Age": 51.0}
            ),
            PresetSample(
                id="diabetes_impaired_path2",
                name="Borderline Impaired Fasting Glucose (High Uncertainty)",
                description="Impaired fasting glucose (132 mg/dL) near diagnostic threshold (126 mg/dL) with borderline BMI. Escalates to Quantum Path 2 (2 QL).",
                expected_path="Path 2 (2 Quantum Layers)",
                values={"Pregnancies": 3.0, "Glucose": 132.0, "BloodPressure": 76.0, "SkinThickness": 28.0, "Insulin": 110.0, "BMI": 30.5, "DiabetesPedigreeFunction": 0.52, "Age": 36.0}
            )
        ]

        return {
            "diabetes": DatasetMetadata(
                key="diabetes",
                display_name="Diabetes Risk Prediction",
                short_name="Diabetes",
                domain="Endocrinology (Type 2 Diabetes Screening)",
                description="Screening for type 2 diabetes and metabolic impairment based on fasting glucose, insulin, body mass index, and pedigree genetics.",
                num_samples=768,
                num_features=8,
                target_positive="Diabetic",
                target_negative="Non-Diabetic",
                class_0_count=500,
                class_1_count=268,
                missing_pct=10.61,
                tau_high=0.85,
                tau_low=0.65,
                features=diabetes_features,
                presets=diabetes_presets
            ),
            "heart": DatasetMetadata(
                key="heart",
                display_name="Heart Disease Prediction",
                short_name="Heart Disease",
                domain="Cardiology (Coronary Artery Disease Screening)",
                description="Predicts the presence of significant coronary artery disease (CAD) based on resting vitals, stress ECG, and cardiac fluoroscopy.",
                num_samples=303,
                num_features=13,
                target_positive="Heart Disease Present",
                target_negative="Absence of Disease",
                class_0_count=164,
                class_1_count=139,
                missing_pct=0.15,
                tau_high=0.85,
                tau_low=0.65,
                features=heart_features,
                presets=heart_presets
            ),
            "breast_cancer": DatasetMetadata(
                key="breast_cancer",
                display_name="Breast Cancer Prediction",
                short_name="Breast Cancer",
                domain="Oncology (Fine Needle Aspirate Biopsy Analysis)",
                description="Classifies breast mass aspirates as malignant or benign from 30 continuous cell nuclei morphometric measurements.",
                num_samples=569,
                num_features=30,
                target_positive="Malignant",
                target_negative="Benign",
                class_0_count=357,
                class_1_count=212,
                missing_pct=0.0,
                tau_high=0.85,
                tau_low=0.65,
                features=wdbc_features,
                presets=wdbc_presets
            ),
        }

    def get_all_datasets(self) -> List[DatasetMetadata]:
        return list(self.datasets_metadata.values())

    def get_dataset(self, key: str) -> DatasetMetadata:
        # Resolve aliases
        alias_map = {
            "breast_cancer": "breast_cancer",
            "wdbc": "breast_cancer",
            "breast-cancer": "breast_cancer",
            "cancer": "breast_cancer",
            "heart": "heart",
            "heart_disease": "heart",
            "heart-disease": "heart",
            "cardio": "heart",
            "diabetes": "diabetes",
            "pima": "diabetes",
        }
        resolved = alias_map.get(key.lower(), key.lower())
        if resolved in self.datasets_metadata:
            return self.datasets_metadata[resolved]
        raise KeyError(f"Dataset '{key}' not found. Valid: {list(self.datasets_metadata.keys())}")

    def get_performance_records(self) -> List[ModelPerformanceItem]:
        return [
            # WDBC
            ModelPerformanceItem(dataset="WDBC Breast Cancer", model="Classical MLP (M1)", model_id="M1", cv_accuracy="96.22% ± 1.55%", cv_f1="94.95% ± 2.06%", cv_roc_auc="0.9904 ± 0.0076", test_f1="94.74%", test_roc_auc="0.9960"),
            ModelPerformanceItem(dataset="WDBC Breast Cancer", model="Fixed HQNN (M2)", model_id="M2", cv_accuracy="96.00% ± 1.58%", cv_f1="94.59% ± 2.21%", cv_roc_auc="0.9908 ± 0.0076", test_f1="94.74%", test_roc_auc="0.9950"),
            ModelPerformanceItem(dataset="WDBC Breast Cancer", model="Adaptive HQNN (M3)", model_id="M3", cv_accuracy="96.00% ± 1.58%", cv_f1="94.59% ± 2.21%", cv_roc_auc="0.9908 ± 0.0076", test_f1="94.74%", test_roc_auc="0.9950"),
            ModelPerformanceItem(dataset="WDBC Breast Cancer", model="Adaptive Classical (M4)", model_id="M4", cv_accuracy="96.22% ± 1.55%", cv_f1="94.95% ± 2.06%", cv_roc_auc="0.9904 ± 0.0076", test_f1="94.74%", test_roc_auc="0.9960"),
            
            # Heart
            ModelPerformanceItem(dataset="UCI Cleveland Heart", model="Classical MLP (M1)", model_id="M1", cv_accuracy="83.89% ± 2.76%", cv_f1="81.28% ± 3.42%", cv_roc_auc="0.9122 ± 0.0354", test_f1="80.00%", test_roc_auc="0.8874"),
            ModelPerformanceItem(dataset="UCI Cleveland Heart", model="Fixed HQNN (M2)", model_id="M2", cv_accuracy="83.89% ± 4.54%", cv_f1="81.16% ± 5.65%", cv_roc_auc="0.9097 ± 0.0401", test_f1="80.77%", test_roc_auc="0.8808"),
            ModelPerformanceItem(dataset="UCI Cleveland Heart", model="Adaptive HQNN (M3)", model_id="M3", cv_accuracy="83.06% ± 3.63%", cv_f1="80.35% ± 4.45%", cv_roc_auc="0.9080 ± 0.0392", test_f1="80.77%", test_roc_auc="0.8755"),
            ModelPerformanceItem(dataset="UCI Cleveland Heart", model="Adaptive Classical (M4)", model_id="M4", cv_accuracy="83.47% ± 3.23%", cv_f1="80.82% ± 3.99%", cv_roc_auc="0.9118 ± 0.0360", test_f1="80.00%", test_roc_auc="0.8861"),
            
            # Diabetes
            ModelPerformanceItem(dataset="Pima Indians Diabetes", model="Classical MLP (M1)", model_id="M1", cv_accuracy="76.71% ± 3.01%", cv_f1="64.13% ± 5.25%", cv_roc_auc="0.8329 ± 0.0364", test_f1="71.74%", test_roc_auc="0.8724"),
            ModelPerformanceItem(dataset="Pima Indians Diabetes", model="Fixed HQNN (M2)", model_id="M2", cv_accuracy="77.20% ± 2.87%", cv_f1="65.34% ± 5.23%", cv_roc_auc="0.8335 ± 0.0384", test_f1="71.43%", test_roc_auc="0.8711"),
            ModelPerformanceItem(dataset="Pima Indians Diabetes", model="Adaptive HQNN (M3)", model_id="M3", cv_accuracy="76.71% ± 3.40%", cv_f1="64.71% ± 5.86%", cv_roc_auc="0.8327 ± 0.0381", test_f1="71.43%", test_roc_auc="0.8696"),
            ModelPerformanceItem(dataset="Pima Indians Diabetes", model="Adaptive Classical (M4)", model_id="M4", cv_accuracy="76.71% ± 3.01%", cv_f1="64.13% ± 5.25%", cv_roc_auc="0.8329 ± 0.0364", test_f1="71.74%", test_roc_auc="0.8724"),
        ]

    def get_workload_records(self) -> List[QuantumWorkloadItem]:
        return [
            QuantumWorkloadItem(dataset="WDBC Breast Cancer", classical_only_pct=98.02, one_qlayer_pct=1.54, two_qlayers_pct=0.44, avg_quantum_layers=0.024, workload_reduction_pct=98.8),
            QuantumWorkloadItem(dataset="UCI Cleveland Heart", classical_only_pct=79.75, one_qlayer_pct=16.12, two_qlayers_pct=4.13, avg_quantum_layers=0.244, workload_reduction_pct=87.8),
            QuantumWorkloadItem(dataset="Pima Indians Diabetes", classical_only_pct=71.01, one_qlayer_pct=17.26, two_qlayers_pct=11.73, avg_quantum_layers=0.407, workload_reduction_pct=79.6),
        ]

    def get_model_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            project_title="Adaptive Hybrid Quantum-Classical Neural Network for Early Medical Disease Diagnosis Using Explainable AI",
            version="1.0.0",
            supported_datasets=["diabetes", "heart", "breast_cancer"],
            num_qubits=4,
            max_quantum_layers=2,
            quantum_encoding="RY Angle Embedding (u in [-pi, pi]^4)",
            quantum_gates=["RY(theta)", "Rot(alpha, beta, gamma)", "Circular CNOTs"],
            entanglement_topology="Circular Ring (0->1, 1->2, 2->3, 3->0)",
            classical_encoder_dims="Linear(D -> 16) -> LayerNorm(16) -> ReLU()",
            readout_method="Expectation values of Pauli-Z observables <Z_j> on all 4 qubits",
            citation_bibtex=(
                "@article{adaptive_hqnn_2026,\n"
                "  title={Adaptive Hybrid Quantum-Classical Neural Network for Early Medical Disease Diagnosis Using Explainable AI},\n"
                "  author={Adaptive HQNN Research Consortium},\n"
                "  journal={IEEE Transactions on Quantum Engineering / Medical Informatics},\n"
                "  year={2026}\n"
                "}"
            )
        )

data_service = DataService()
