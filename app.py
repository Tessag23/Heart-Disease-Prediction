import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background-color:white;
}

.metric-container{
    background-color:white;
    padding:15px;
    border-radius:10px;
    box-shadow:0px 0px 5px lightgray;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    models = {

        "Logistic Regression":
        joblib.load("model/logistic_regression.pkl"),

        "Decision Tree":
        joblib.load("model/decision_tree.pkl"),

        "KNN":
        joblib.load("model/knn.pkl"),

        "Gaussian Naive Bayes":
        joblib.load("model/naive_bayes.pkl"),

        "Random Forest":
        joblib.load("model/random_forest.pkl")

    }

    scaler = joblib.load("model/scaler.pkl")

    return models, scaler


models, scaler = load_models()

# ==========================================================
# FEATURE LIST
# ==========================================================

FEATURES = [

    "HighBP",

    "HighChol",

    "CholCheck",

    "BMI",

    "Smoker",

    "Stroke",

    "Diabetes",

    "PhysActivity",

    "Fruits",

    "Veggies",

    "HvyAlcoholConsump",

    "AnyHealthcare",

    "NoDocbcCost",

    "GenHlth",

    "MentHlth",

    "PhysHlth",

    "DiffWalk",

    "Sex",

    "Age",

    "Education",

    "Income"

]

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Heart Disease Prediction")

page = st.sidebar.radio(

    "Navigation",

    [

        "Home",

        "Prediction",

        "Evaluation",

        "Model Comparison",
    ]

)

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "Home":

    st.title("Heart Disease Prediction")

    st.markdown("---")

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("Project Overview")

        st.write("""

This application predicts whether a person is likely to have heart disease using five machine learning algorithms.

Implemented Models

• Logistic Regression

• Decision Tree

• K-Nearest Neighbors

• Gaussian Naive Bayes

• Random Forest

""")

    with col2:

        st.subheader("Dataset Information")

        st.write("""

Dataset : CDC BRFSS 2015

Rows : 253,680

Features : 21

Target : HeartDiseaseorAttack

Classification : Binary

""")
# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "Prediction":

    st.title("Heart Disease Prediction")

    st.write(
        """
        Upload a CSV file containing the **21 input features only**.
        The application will predict whether each person is likely
        to have heart disease.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload Feature CSV",
        type=["csv"],
        key="prediction"
    )

    model_name = st.selectbox(
        "Select Model",
        list(models.keys()),
        key="prediction_model"
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Dataset")

            st.dataframe(df.head())

            st.write("Rows:", df.shape[0])

            st.write("Columns:", df.shape[1])

            st.markdown("---")

            missing = []

            for col in FEATURES:

                if col not in df.columns:

                    missing.append(col)

            if len(missing) > 0:

                st.error("Missing Required Columns")

                st.write(missing)

                st.stop()

            X = df[FEATURES]

            model = models[model_name]

            if model_name in [
                "Logistic Regression",
                "KNN"
            ]:

                X_input = scaler.transform(X)

            else:

                X_input = X

            prediction = model.predict(X_input)

            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(X_input)[:,1]

            else:

                probability = None

            result = df.copy()

            result["Prediction"] = prediction

            if probability is not None:

                result["Probability"] = probability

            st.markdown("---")

            st.subheader("Prediction Result")

            total = len(result)

            positive = int(result["Prediction"].sum())

            negative = total - positive

            c1, c2, c3 = st.columns(3)

            c1.metric("Total Records", total)

            c2.metric("Heart Disease", positive)

            c3.metric("No Heart Disease", negative)

            st.dataframe(result.head(20))

            st.markdown("---")

            st.subheader("Prediction Distribution")

            fig, ax = plt.subplots(figsize=(5,4))

            counts = result["Prediction"].value_counts()

            ax.bar(
                counts.index.astype(str),
                counts.values
            )

            ax.set_xlabel("Prediction")

            ax.set_ylabel("Count")

            ax.set_title("Prediction Distribution")

            st.pyplot(fig)

            csv = result.to_csv(index=False)

            st.download_button(

                label="⬇ Download Prediction CSV",

                data=csv,

                file_name="predictions.csv",

                mime="text/csv"

            )

        except Exception as e:

            st.error("Error while processing the file.")

            st.exception(e)
 # ==========================================================
# EVALUATION PAGE
# ==========================================================

elif page == "Evaluation":

    st.title("Model Evaluation")

    st.write(
        """
        Upload a labeled CSV containing the 21 input features
        and the **HeartDiseaseorAttack** target column.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload Test Dataset",
        type=["csv"],
        key="evaluation"
    )

    model_name = st.selectbox(
        "Select Model",
        list(models.keys()),
        key="evaluation_model"
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.subheader("Dataset Preview")

            st.dataframe(df.head())

            st.markdown("---")

            if "HeartDiseaseorAttack" not in df.columns:

                st.error(
                    "Target column 'HeartDiseaseorAttack' not found."
                )

                st.stop()

            missing = []

            for col in FEATURES:

                if col not in df.columns:

                    missing.append(col)

            if len(missing) > 0:

                st.error("Missing Required Features")

                st.write(missing)

                st.stop()

            X = df[FEATURES]

            y = df["HeartDiseaseorAttack"]

            model = models[model_name]

            if model_name in [
                "Logistic Regression",
                "KNN"
            ]:

                X_input = scaler.transform(X)

            else:

                X_input = X

            prediction = model.predict(X_input)

            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(X_input)[:,1]

                auc = roc_auc_score(
                    y,
                    probability
                )

            else:

                probability = None

                auc = None

            accuracy = accuracy_score(
                y,
                prediction
            )

            precision = precision_score(
                y,
                prediction
            )

            recall = recall_score(
                y,
                prediction
            )

            f1 = f1_score(
                y,
                prediction
            )

            mcc = matthews_corrcoef(
                y,
                prediction
            )

            st.markdown("---")

            st.subheader("Evaluation Metrics")

            c1,c2,c3 = st.columns(3)

            c1.metric(
                "Accuracy",
                f"{accuracy:.4f}"
            )

            c2.metric(
                "Precision",
                f"{precision:.4f}"
            )

            c3.metric(
                "Recall",
                f"{recall:.4f}"
            )

            c1,c2,c3 = st.columns(3)

            c1.metric(
                "F1 Score",
                f"{f1:.4f}"
            )

            c2.metric(
                "MCC",
                f"{mcc:.4f}"
            )

            if auc is not None:

                c3.metric(
                    "AUC",
                    f"{auc:.4f}"
                )

            else:

                c3.metric(
                    "AUC",
                    "N/A"
                )

            st.markdown("---")

            st.subheader("Confusion Matrix")

            fig, ax = plt.subplots(figsize=(5,5))

            ConfusionMatrixDisplay.from_predictions(
                y,
                prediction,
                cmap="Blues",
                ax=ax
            )

            st.pyplot(fig)

            if probability is not None:

                st.markdown("---")

                st.subheader("ROC Curve")

                fig, ax = plt.subplots(figsize=(6,6))

                RocCurveDisplay.from_predictions(
                    y,
                    probability,
                    ax=ax
                )

                st.pyplot(fig)

            st.markdown("---")

            st.subheader("Classification Report")

            report = classification_report(
                y,
                prediction,
                output_dict=True
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(
                report_df,
                width='stretch' 
            )

            st.markdown("---")

            st.subheader("Prediction Results")

            output = df.copy()

            output["Prediction"] = prediction

            if probability is not None:

                output["Probability"] = probability

            st.dataframe(
                output.head(20)
            )

            csv = output.to_csv(index=False)

            st.download_button(

                "⬇ Download Results",

                csv,

                "evaluation_results.csv",

                "text/csv"

            )

        except Exception as e:

            st.error("Error while evaluating the model.")

            st.exception(e)
 # ==========================================================
# MODEL COMPARISON PAGE
# ==========================================================

elif page == "Model Comparison":

    st.title("Model Comparison")

    try:

        comparison = pd.read_csv("model/model_comparison.csv")

        st.subheader("Performance Comparison")

        st.dataframe(
            comparison,
            width='stretch'
        )

        st.markdown("---")

        st.subheader("Accuracy Comparison")

        fig, ax = plt.subplots(figsize=(10,5))

        ax.bar(
            comparison["Model"],
            comparison["Accuracy"]
        )

        ax.set_ylabel("Accuracy")

        ax.set_xlabel("Model")

        plt.xticks(rotation=20)

        st.pyplot(fig)

        st.markdown("---")

        st.subheader("Metric Comparison")

        metrics = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC"
        ]

        selected_metric = st.selectbox(
            "Select Metric",
            metrics
        )

        fig, ax = plt.subplots(figsize=(10,5))

        ax.bar(
            comparison["Model"],
            comparison[selected_metric]
        )

        ax.set_ylabel(selected_metric)

        ax.set_xlabel("Model")

        plt.xticks(rotation=20)

        st.pyplot(fig)

        st.success(
            f"Best Performing Model : {comparison.iloc[0]['Model']}"
        )

    except Exception as e:

        st.error("Unable to load model_comparison.csv")

        st.exception(e)