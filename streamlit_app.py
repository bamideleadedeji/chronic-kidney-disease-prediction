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

st.title("Chronic Kidney Disease Prediction App")

# -------------------------------
# Upload dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload CKD Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    if 'class_notckd' not in df.columns:
        st.error("Target column 'class_notckd' not found.")
    else:
        X = df.drop('class_notckd', axis=1)
        y = df['class_notckd']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # -------------------------------
        # Logistic Regression
        # -------------------------------
        log_model = LogisticRegression(max_iter=2000)
        log_model.fit(X_train_scaled, y_train)
        y_pred_log = log_model.predict(X_test_scaled)
        acc_log = accuracy_score(y_test, y_pred_log)

        # ROC curve
        y_prob = log_model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # -------------------------------
        # Decision Tree
        # -------------------------------
        tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
        tree_model.fit(X_train, y_train)
        y_pred_tree = tree_model.predict(X_test)
        acc_tree = accuracy_score(y_test, y_pred_tree)

        # -------------------------------
        # Model Performance Section
        # -------------------------------
        st.subheader("Model Comparison")

        fig_comp, ax_comp = plt.subplots()
        ax_comp.bar(["Logistic Regression", "Decision Tree"], [acc_log, acc_tree], color=["skyblue","salmon"])
        ax_comp.set_ylim(0,1)
        ax_comp.set_ylabel("Accuracy")
        ax_comp.set_title("Model Accuracy Comparison")
        for i, v in enumerate([acc_log, acc_tree]):
            ax_comp.text(i, v + 0.01, f"{v:.2f}", ha='center')
        st.pyplot(fig_comp)

        # -------------------------------
        # Feature Importance (Decision Tree)
        # -------------------------------
        st.subheader("Top Features (Decision Tree)")
        importances = tree_model.feature_importances_
        feature_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        fig_imp, ax_imp = plt.subplots()
        ax_imp.bar(feature_df["Feature"][:10], feature_df["Importance"][:10], color="green")
        ax_imp.set_xticklabels(feature_df["Feature"][:10], rotation=45, ha='right')
        st.pyplot(fig_imp)

        # -------------------------------
        # Decision Tree Visualization
        # -------------------------------
        st.subheader("Decision Tree Visualization")
        fig_tree, ax_tree = plt.subplots(figsize=(12, 6))
        plot_tree(tree_model, feature_names=X.columns, class_names=["CKD","Not CKD"], filled=True)
        st.pyplot(fig_tree)

        # -------------------------------
        # User Input Prediction
        # -------------------------------
        st.subheader("Predict CKD for New Patient")

        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction = log_model.predict(input_scaled)[0]

            if prediction == 1:
                st.error("Prediction: CKD Detected")
            else:
                st.success("Prediction: No CKD Detected")

            # -------------------------------
            # Download Prediction Report
            # -------------------------------
            st.subheader("Download Prediction Report")
            report_df = X_test.copy()
            report_df['Actual'] = y_test.values
            report_df['Logistic_Pred'] = y_pred_log
            report_df['Tree_Pred'] = y_pred_tree

            towrite = BytesIO()
            report_df.to_csv(towrite, index=False)
            towrite.seek(0)
            st.download_button(
                label="Download Predictions CSV",
                data=towrite,
                file_name="ckd_prediction_report.csv",
                mime="text/csv"
            )
