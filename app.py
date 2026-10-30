
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from io import BytesIO, StringIO
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f3f7fc;
}
h1, h2, h3 {
    color: #12345a;
}
div.stButton > button {
    background-color: #0877c9;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.title("❤️ Heart Disease Prediction System")
st.subheader("Machine Learning Based Heart Disease Risk Screening")

st.info(
    "This educational project estimates a heart disease risk "
    "classification from patient health information. "
    "It is not a medical diagnosis."
)

# ---------------- DATASET LOADING ----------------

UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

ALIASES = {
    "age": "age",
    "sex": "sex",
    "cp": "cp",
    "chestpaintype": "cp",
    "trestbps": "trestbps",
    "restingbp": "trestbps",
    "restingbloodpressure": "trestbps",
    "chol": "chol",
    "cholesterol": "chol",
    "fbs": "fbs",
    "fastingbloodsugar": "fbs",
    "restecg": "restecg",
    "restingecg": "restecg",
    "thalach": "thalach",
    "maxheartrate": "thalach",
    "maximumheartrate": "thalach",
    "exang": "exang",
    "exerciseinducedangina": "exang",
    "oldpeak": "oldpeak",
    "slope": "slope",
    "ca": "ca",
    "thal": "thal",
    "target": "target",
    "num": "target",
    "condition": "target",
    "output": "target",
    "heartdisease": "target"
}


def normalize_data(df):
    df = df.copy()

    # Recognize common UCI dataset format
    if len(df.columns) == 14 and not any(
        str(c).lower().strip() in ALIASES for c in df.columns
    ):
        df.columns = UCI_COLUMNS

    df.columns = [
        ALIASES.get(
            "".join(ch for ch in str(c).lower() if ch.isalnum()),
            str(c).strip().lower()
        )
        for c in df.columns
    ]

    if "target" not in df.columns:
        raise ValueError("Target column not found in dataset.")

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing these features: " + ", ".join(missing)
        )

    df = df[FEATURES + ["target"]].copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["target"])

    # Cleveland UCI target: 0 = no disease, 1-4 = disease.
    # Common Kaggle target: 0 = no disease, 1 = disease.
    df["target"] = (df["target"] > 0).astype(int)

    df = df.drop_duplicates()

    return df


def load_dataset():
    csv_files = list(Path(".").glob("*.csv"))

    if csv_files:
        return normalize_data(pd.read_csv(csv_files[0]))

    zip_files = list(Path(".").glob("*.zip"))

    for zip_path in zip_files:
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".csv"):
                    with archive.open(name) as file:
                        return normalize_data(pd.read_csv(file))

                if name.lower().endswith(".data"):
                    with archive.open(name) as file:
                        df = pd.read_csv(
                            file,
                            header=None,
                            names=UCI_COLUMNS,
                            na_values=["?"]
                        )
                        return normalize_data(df)

    raise FileNotFoundError(
        "No CSV dataset or supported UCI .data file was found. "
        "Upload heart.csv or the UCI dataset ZIP."
    )


# ---------------- TRAIN MACHINE LEARNING MODELS ----------------

@st.cache_resource
def train_models():

    df = load_dataset()

    X = df[FEATURES]
    y = df["target"]

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Dataset must contain both target classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ]),

        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                max_depth=4, random_state=42
            ))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            ))
        ])
    }

    results = {}
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        results[name] = {
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(
                y_test, prediction, zero_division=0
            ),
            "Recall": recall_score(
                y_test, prediction, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, prediction, zero_division=0
            ),
            "Confusion Matrix": confusion_matrix(
                y_test, prediction, labels=[0, 1]
            )
        }

        trained[name] = model

    # Select using recall first, then accuracy.
    best_name = max(
        results,
        key=lambda name: (
            results[name]["Recall"],
            results[name]["Accuracy"]
        )
    )

    return df, trained, results, best_name


# ---------------- SIDEBAR ----------------

st.sidebar.title("❤️ Heart Care")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Patient Prediction", "Model Evaluation", "About Project"]
)

# ---------------- LOAD MODEL ----------------

try:
    df, models, results, best_model = train_models()
except Exception as error:
    st.error("Dataset or model loading problem.")
    st.warning(str(error))
    st.info(
        "Please upload the correct heart disease dataset "
        "to your GitHub repository."
    )
    st.stop()


# ---------------- HOME PAGE ----------------

if page == "Home":

    st.title("Welcome to Heart Disease Prediction")

    st.write(
        "This system uses machine learning to analyze patient "
        "health information and estimate possible heart disease risk."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Patient Records", len(df))
    col2.metric("Input Features", len(FEATURES))
    col3.metric("ML Algorithms", len(models))

    st.markdown("---")

    st.subheader("Project Objectives")

    st.write("1. Collect and understand patient health data.")
    st.write("2. Preprocess the dataset.")
    st.write("3. Train machine learning classification models.")
    st.write("4. Evaluate model performance.")
    st.write("5. Predict the class for new patient information.")

    st.success("Selected model: " + best_model)

    st.info(
        "Use Patient Prediction from the sidebar to enter "
        "health information."
    )


# ---------------- PATIENT PREDICTION ----------------

elif page == "Patient Prediction":

    st.title("Patient Health Information")

    st.write("Enter the patient's clinical information below.")

    with st.form("patient_form"):

        patient_name = st.text_input("Patient Name")

        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input(
                "Age", 18, 100, 45
            )

            sex = st.selectbox(
                "Sex",
                ["Male", "Female"]
            )

            cp = st.selectbox(
                "Chest Pain Type",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Typical Angina",
                    2: "Atypical Angina",
                    3: "Non-anginal Pain",
                    4: "Asymptomatic"
                }[x]
            )

            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                70, 250, 120
            )

            chol = st.number_input(
                "Serum Cholesterol (mg/dl)",
                80, 700, 200
            )

            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                ["No", "Yes"]
            )

            restecg = st.selectbox(
                "Resting ECG",
                [0, 1, 2]
            )

        with c2:

            thalach = st.number_input(
                "Maximum Heart Rate",
                60, 250, 150
            )

            exang = st.selectbox(
                "Exercise Induced Angina",
                ["No", "Yes"]
            )

            oldpeak = st.number_input(
                "ST Depression (Oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1
            )

            slope = st.selectbox(
                "Slope of ST Segment",
                [1, 2, 3]
            )

            ca = st.selectbox(
                "Number of Major Vessels",
                [0, 1, 2, 3, 4]
            )

            thal = st.selectbox(
                "Thal",
                [3, 6, 7]
            )

        submitted = st.form_submit_button(
            "Predict Heart Disease Risk",
            use_container_width=True
        )

    if submitted:

        if not patient_name.strip():
            st.warning("Please enter the patient name.")

        else:

            patient = pd.DataFrame([{
                "age": age,
                "sex": 1 if sex == "Male" else 0,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": 1 if fbs == "Yes" else 0,
                "restecg": restecg,
                "thalach": thalach,
                "exang": 1 if exang == "Yes" else 0,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal
            }], columns=FEATURES)

            prediction = models[best_model].predict(patient)[0]

            st.markdown("---")
            st.subheader("Prediction Result")

            st.write("Patient Name:", patient_name)
            st.write("Selected ML Model:", best_model)

            if prediction == 1:
                st.error(
                    "Possible Heart Disease Risk Detected"
                )
                st.write(
                    "The model classified this patient in "
                    "the heart disease category."
                )
            else:
                st.success(
                    "No Heart Disease Class Predicted"
                )
                st.write(
                    "The model classified this patient in "
                    "the no heart disease category."
                )

            st.warning(
                "This is a machine-learning classification, "
                "not a medical diagnosis. It cannot confirm "
                "or rule out heart disease. Consult a qualified "
                "healthcare professional."
            )


# ---------------- MODEL EVALUATION ----------------

elif page == "Model Evaluation":

    st.title("Machine Learning Model Evaluation")

    st.write(
        "Comparison of Logistic Regression, Decision Tree "
        "and Random Forest."
    )

    table = pd.DataFrame(results).T

    display_table = table[
        ["Accuracy", "Precision", "Recall", "F1-score"]
    ].copy()

    st.dataframe(
        display_table.style.format("{:.3f}"),
        use_container_width=True
    )

    st.success("Selected model: " + best_model)

    st.subheader("Confusion Matrix")

    matrix = results[best_model]["Confusion Matrix"]

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual No Disease", "Actual Disease"],
        columns=["Predicted No Disease", "Predicted Disease"]
    )

    st.dataframe(matrix_df, use_container_width=True)

    st.caption(
        "Metrics are based on a held-out test split of the selected dataset."
    )


# ---------------- ABOUT PROJECT ----------------

elif page == "About Project":

    st.title("About Our Project")

    st.subheader("Heart Disease Prediction")

    st.write(
        "This project uses Data Science and Machine Learning "
        "to analyze patient health information and classify "
        "possible heart disease risk."
    )

    st.subheader("Technologies Used")

    st.write("- Python")
    st.write("- Streamlit")
    st.write("- Pandas and NumPy")
    st.write("- Scikit-learn")
    st.write("- Machine Learning Classification")

    st.subheader("Project Workflow")

    st.write("1. Dataset Collection")
    st.write("2. Data Exploration")
    st.write("3. Data Preprocessing")
    st.write("4. Model Training")
    st.write("5. Model Evaluation")
    st.write("6. Prediction")

    st.info(
        "This project is designed for educational and "
        "decision-support demonstration purposes only."
    )

st.markdown("---")
st.caption(
    "Heart Disease Prediction System | Data Science & Machine Learning | Group 9"
)

