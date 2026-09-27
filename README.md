# Heart Disease Prediction using Machine Learning

## a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models for predicting whether a person has a history of heart disease or heart attack. The models are trained and evaluated on the same dataset using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

An interactive Streamlit application is also developed to demonstrate model predictions and evaluation results.

## b. Dataset Description

**Dataset:** Heart Disease Health Indicators Dataset

**Source:** Kaggle

**Problem Type:** Binary Classification

**Target Variable:** `HeartDiseaseorAttack`

- `0` — No reported heart disease or heart attack
- `1` — Reported heart disease or heart attack

**Dataset size:** 253,680 instances and 21 input features.

### Input Features

`HighBP`, `HighChol`, `CholCheck`, `BMI`, `Smoker`, `Stroke`, `Diabetes`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `GenHlth`, `MentHlth`, `PhysHlth`, `DiffWalk`, `Sex`, `Age`, `Education`, `Income`

## c. GitHub Repository Link

**GitHub Repository:** (https://heart-disease-prediction-ka2wmzh8vpuem7z5iy95ew.streamlit.app/)

## d. Models Used

The following classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN) Classifier
4. Gaussian Naive Bayes
5. Random Forest Classifier (Ensemble Model)

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9080934739 | 0.8348285877 | 0.4882943144 | 0.1137957911 | 0.1845764855 | 0.2031794731 |
| Decision Tree | 0.9043174694 | 0.8046118044 | 0.4050632911 | 0.09976617303 | 0.1601000625 | 0.1651745835 |
| kNN | 0.8989028213 | 0.7005181303 | 0.3534482759 | 0.1278254092 | 0.1877504293 | 0.1681238326 |
| Naive Bayes | 0.8156169849 | 0.7961343163 | 0.251617815 | 0.5151987529 | 0.3381074169 | 0.2667594024 |
| Random Forest (Ensemble) | 0.9074522656 | 0.808817469 | 0.4708029197 | 0.1005455963 | 0.1657032755 | 0.185759095 |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Logistic Regression achieved the highest Accuracy (90.81%) and highest AUC (0.835) among the five models. It also achieved relatively high Precision (0.488) and MCC (0.203). However, its Recall was low (0.114), indicating that the model identified only a small proportion of the positive heart-disease cases. Overall, it provided the strongest general performance based on Accuracy and AUC. |
| Decision Tree | The Decision Tree achieved an Accuracy of 90.43% and an AUC of 0.805. Its Precision (0.405), Recall (0.100), F1 Score (0.160), and MCC (0.165) were relatively low. Although its Accuracy was high, the low Recall indicates that it missed many positive cases. Its overall performance was slightly weaker than Logistic Regression. |
| kNN | kNN achieved an Accuracy of 89.89%, with an AUC of 0.7005, which was the lowest AUC among all five models. Its Precision was 0.353 and Recall was 0.128, resulting in an F1 Score of 0.188 and MCC of 0.168. Although its Accuracy was relatively high, its lower AUC and moderate classification metrics indicate weaker overall performance compared with Logistic Regression. |
| Naive Bayes | Naive Bayes had the lowest Accuracy (81.56%), but it achieved the highest Recall (0.515) and the highest F1 Score (0.338) and MCC (0.267) among the five models. This indicates that it was considerably better at identifying positive heart-disease cases, although this came with lower Precision (0.252) and overall Accuracy. Therefore, Naive Bayes may be more useful when detecting positive cases is prioritized over overall Accuracy. |
| Random Forest (Ensemble) | Random Forest achieved an Accuracy of 90.75%, making it the second-highest model in terms of Accuracy. Its AUC was 0.809, while Precision was 0.471 and Recall was 0.101. Its F1 Score (0.166) and MCC (0.186) were lower than Logistic Regression and Naive Bayes. Despite being an ensemble model, it did not outperform Logistic Regression on the majority of the reported metrics. |