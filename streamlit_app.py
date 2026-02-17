import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="CKD Prediction App", layout="wide")

st.title("Chronic Kidney Disease Prediction App")

# -----------------------------
# Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload CKD Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "class_notckd" not in df.columns:
        st.error("Target column 'class_notckd' not found in dataset.")
        st.stop()

    # Ensure target is numeric
    if df["class_notckd"].dtype == "object":
        df["class_notckd"] = df["class_notckd"].map({"ckd": 1, "notckd": 0})

    X = df.drop("class_notckd", axis=1)
    y = df["class_notckd"]

    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])

    # -----------------------------
    # Train-Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Logistic Regression
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(max_iter=2000)
    log_model.fit(X_train_scaled, y_train)

    y_pred_log = log_model.predict(X_test_scaled)
    acc_log = accuracy_score(y_test, y_pred_log)

    # ROC Curve
    y_prob = log_model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # -----------------------------
    # Decision Tree
    # -----------------------------
    tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree_model.fit(X_train, y_train)

    y_pred_tree = tree_model.predict(X_test)
    acc_tree = accuracy_score(y_test, y_pred_tree)

    # =============================
    # MODEL COMPARISON
    # =============================
    st.subheader("Model Comparison")

    fig_comp, ax_comp = plt.subplots()
    models = ["Logistic Regression", "Decision Tree"]
    accuracies = [acc_log, acc_tree]

    ax_comp.bar(models, accuracies)
    ax_comp.set_ylim(0, 1)
    ax_comp.set_ylabel("Accuracy")

    for i, v in enumerate(accuracies):
        ax_comp.text(i, v + 0.02, f"{v:.2f}", ha="center")

    st.pyplot(fig_comp)

    # =============================
    # ROC CURVE
    # =============================
    st.subheader("ROC Curve")

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()

    st.pyplot(fig_roc)

    # =============================
    # FEATURE IMPORTANCE
    # =============================
    st.subheader("Top Feature Importance (Decision Tree)")

    importances = tree_model.feature_importances_
    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    fig_imp, ax_imp = plt.subplots()
    ax_imp.bar(feature_df["Feature"], feature_df["Importance"])
    ax_imp.set_xticklabels(feature_df["Feature"], rotation=45, ha="right")

    st.pyplot(fig_imp)

    # =============================
    # DOWNLOAD PREDICTION REPORT
    # =============================
    st.subheader("Download Prediction Report")

    report_df = X_test.copy()
    report_df["Actual"] = y_test.values
    report_df["Logistic_Prediction"] = y_pred_log
    report_df["Tree_Prediction"] = y_pred_tree

    csv = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="ckd_prediction_report.csv",
        mime="text/csv"
    )

    # =============================
    # USER INPUT PREDICTION
    # =============================
    st.subheader("Predict CKD for New Patient")

    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(
            f"{col}",
            value=float(X[col].mean())
        )

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        prediction = log_model.predict(input_scaled)[0]

        if prediction == 1:
            st.error("Prediction: CKD Detected")
        else:
            st.success("Prediction: No CKD Detected")
