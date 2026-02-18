%%writefile streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report

# Import ucimlrepo to fetch data directly within the Streamlit app
from ucimlrepo import fetch_ucirepo

st.set_page_config(page_title="CKD Prediction App", layout="wide")

st.title("Chronic Kidney Disease Prediction App")

# -----------------------------
# Fetch Dataset directly using ucimlrepo
# -----------------------------
# This replaces st.file_uploader, as data is now fetched automatically

@st.cache_data # Cache the data fetching to avoid re-downloading on every rerun
def load_data():
    chronic_kidney_disease = fetch_ucirepo(id=336)
    X = chronic_kidney_disease.data.features
    y = chronic_kidney_disease.data.targets
    df = pd.concat([X, y], axis=1)
    return df

df = load_data()

if df is not None:

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # --- Preprocessing steps from the notebook --- 
    # 1. Replace '?' with NaN and clean 'class' column
    df.replace('?', np.nan, inplace=True)

    if 'class' not in df.columns:
        st.error("Target column 'class' not found in dataset. Please ensure your dataset has a 'class' column.")
        st.stop()

    # Ensure the 'class' column is cleaned before mapping
    df['class'] = df['class'].astype(str).str.strip()

    # Map 'ckd' to 0 and 'notckd' to 1, consistent with 'class_notckd' being 1 for not-CKD in the notebook's interpretation
    # Handle cases where 'class' might be numeric or other types after ucimlrepo fetch
    # Explicitly check unique values to ensure proper mapping
    unique_classes = df['class'].unique()
    if 'ckd' in unique_classes and 'notckd' in unique_classes:
        df['class_encoded'] = df['class'].map({'ckd': 0, 'notckd': 1})
    else:
        st.error(f"Unexpected unique values in 'class' column: {unique_classes}. Expected 'ckd' and 'notckd'.")
        st.stop()

    if df['class_encoded'].isnull().any():
        st.warning("Some values in 'class' column could not be encoded. Please check for typos.")
        # Optionally, you might want to drop these rows or re-map them
        # For now, we'll stop if there are unhandled values
        st.stop()

    # Separate features and target
    y = df['class_encoded']
    X = df.drop(['class', 'class_encoded'], axis=1) # Drop original 'class' and temporary 'class_encoded' from features

    # Identify initial column types in X
    categorical_cols_X = X.select_dtypes(include='object').columns
    numerical_cols_X = X.select_dtypes(exclude='object').columns

    # Convert numerical columns stored as object to numeric, coercing errors
    # This handles cases where numbers might be stored as strings due to '?' removal or original data issues
    for col in numerical_cols_X:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Re-identify numerical and categorical columns after potential conversions to ensure accuracy
    numerical_cols_X = X.select_dtypes(include=[np.number]).columns
    categorical_cols_X = X.select_dtypes(include='object').columns


    # Impute missing numerical values with median
    for col in numerical_cols_X:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    # Impute missing categorical values with mode
    for col in categorical_cols_X:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode()[0])

    # One-Hot Encode categorical features
    X = pd.get_dummies(X, columns=categorical_cols_X, drop_first=True)

    # --- End of Preprocessing steps --- 

    # -----------------------------
    # Train-Test Split (with stratification for target variable)
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Logistic Regression
    # -----------------------------
    st.subheader("Logistic Regression Model")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(max_iter=2000, random_state=42) # Added random_state for reproducibility
    log_model.fit(X_train_scaled, y_train)

    y_pred_log = log_model.predict(X_test_scaled)
    acc_log = accuracy_score(y_test, y_pred_log)
    st.write(f"Logistic Regression Accuracy: {acc_log:.2f}")
    st.text("Classification Report (Logistic Regression):")
    st.code(classification_report(y_test, y_pred_log))


    # ROC Curve
    y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]
    fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
    roc_auc_log = auc(fpr_log, tpr_log)

    # -----------------------------
    # Decision Tree
    # -----------------------------
    st.subheader("Decision Tree Model")
    tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree_model.fit(X_train, y_train) # Using unscaled data for Decision Tree

    y_pred_tree = tree_model.predict(X_test)
    acc_tree = accuracy_score(y_test, y_pred_tree)
    st.write(f"Decision Tree Accuracy: {acc_tree:.2f}")
    st.text("Classification Report (Decision Tree):")
    st.code(classification_report(y_test, y_pred_tree))

    # =============================
    # MODEL COMPARISON
    # =============================
    st.subheader("Model Comparison")

    fig_comp, ax_comp = plt.subplots(figsize=(8, 5))
    models = ["Logistic Regression", "Decision Tree"]
    accuracies = [acc_log, acc_tree]

    ax_comp.bar(models, accuracies, color=['skyblue', 'lightcoral'])
    ax_comp.set_ylim(0.7, 1.0) # Adjusted ylim to better visualize differences
    ax_comp.set_ylabel("Accuracy")
    ax_comp.set_title("Model Accuracies")

    for i, v in enumerate(accuracies):
        ax_comp.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    st.pyplot(fig_comp)

    # =============================
    # ROC CURVE
    # =============================
    st.subheader("ROC Curve (Logistic Regression)")

    fig_roc, ax_roc = plt.subplots(figsize=(7, 7))
    ax_roc.plot(fpr_log, tpr_log, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_log:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Receiver Operating Characteristic for Logistic Regression")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # =============================
    # FEATURE IMPORTANCE
    # =============================
    st.subheader("Top Feature Importance (Decision Tree)")

    importances = tree_model.feature_importances_
    feature_names = X.columns # Use X.columns for feature names
    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    ax_imp.bar(feature_df["Feature"], feature_df["Importance"], color='lightgreen')
    ax_imp.set_ylabel("Importance")
    ax_imp.set_title("Top 10 Feature Importances from Decision Tree")
    ax_imp.set_xticklabels(feature_df["Feature"], rotation=45, ha="right")
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    st.pyplot(fig_imp)

    # =============================
    # DOWNLOAD PREDICTION REPORT
    # =============================
    st.subheader("Download Prediction Report")

    # Create a report DataFrame with original X_test features, actual, and predicted values
    report_df = X_test.copy()
    report_df['Actual_Class'] = y_test
    report_df['Predicted_Logistic_Regression'] = y_pred_log
    report_df['Predicted_Decision_Tree'] = y_pred_tree

    # Add predicted probabilities for Logistic Regression
    report_df['Probability_NotCKD_Logistic_Regression'] = y_prob_log

    # Convert boolean columns from get_dummies to integers for cleaner CSV export
    for col in report_df.columns:
        if report_df[col].dtype == 'bool':
            report_df[col] = report_df[col].astype(int)

    csv_output = report_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Prediction Report as CSV",
        data=csv_output,
        file_name="ckd_prediction_report.csv",
        mime="text/csv",
    )
