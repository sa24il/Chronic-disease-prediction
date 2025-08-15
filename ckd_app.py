import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import streamlit as st
import matplotlib.pyplot as plt

# Set page configuration for better appearance
st.set_page_config(page_title="Chronic Kidney Disease Prediction System", layout="wide")

# Define feature columns globally
feature_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'sc', 'hemo', 'pcv', 'wc', 'rc']

# Load and preprocess the dataset
# Load and preprocess the dataset
@st.cache_data
def load_data():
    cleaning_logs = []  # Store cleaning messages here
    try:
        data = pd.read_csv("kidney_disease.csv")
    except FileNotFoundError:
        st.error(
            "Dataset 'kidney_disease.csv' not found in the project directory. "
            "Please download it from Kaggle: https://www.kaggle.com/datasets/mansoordanish/chronic-kidney-disease"
        )
        st.stop()

    # Determine target column
    target_col = 'class' if 'class' in data.columns else 'classification' if 'classification' in data.columns else None
    if target_col is None:
        st.error("Target column ('class' or 'classification') not found in dataset.")
        st.stop()

    # Standardize target values
    data[target_col] = data[target_col].astype(str).str.strip().str.lower()
    data['Outcome'] = data[target_col].map({'ckd': 1, 'notckd': 0})
    data = data.dropna(subset=['Outcome'])

    # Select only feature columns
    data_features = data[feature_cols].copy()

    # Convert features to numeric and fill missing
    for col in feature_cols:
        data_features[col] = pd.to_numeric(data_features[col], errors='coerce')
        if data_features[col].isnull().any():
            cleaning_logs.append(f"Filled missing values in '{col}' with median.")
            data_features[col].fillna(data_features[col].median(), inplace=True)

    # Remove outliers
    initial_rows = len(data_features)
    for col in feature_cols:
        col_mean = data_features[col].mean()
        col_std = data_features[col].std()
        data_features = data_features[
            (data_features[col] <= col_mean + 3 * col_std) &
            (data_features[col] >= col_mean - 3 * col_std)
        ]
    if len(data_features) < initial_rows:
        cleaning_logs.append(f"Removed {initial_rows - len(data_features)} rows with outliers.")

    # Align target
    data = data.loc[data_features.index]
    for col in feature_cols:
        data[col] = data_features[col]

    return data, cleaning_logs

try:
    data, cleaning_logs = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Split features and target
X = data[feature_cols]
y = data['Outcome']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Streamlit app
st.title("Chronic Kidney Disease Prediction System")
st.markdown("""
This tool predicts chronic kidney disease risk based on health metrics using a Random Forest model.
Enter your details below to get a prediction. Explore the data analysis tabs for insights.
Dataset source: https://www.kaggle.com/datasets/mansoordanish/chronic-kidney-disease
""")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "Model Performance"])

with tab1:
    # Input form and prediction
    st.subheader("Enter Health Metrics")
    with st.form(key="ckd_form"):
        age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, step=1.0, value=40.0)
        bp = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, step=1.0, value=80.0)
        sg = st.number_input("Specific Gravity (1.005 to 1.030)", min_value=1.005, max_value=1.030, step=0.005, value=1.020)
        al = st.number_input("Albumin (0 to 5)", min_value=0.0, max_value=5.0, step=1.0, value=0.0)
        su = st.number_input("Sugar (0 to 5)", min_value=0.0, max_value=5.0, step=1.0, value=0.0)
        bgr = st.number_input("Blood Glucose Random (mg/dL)", min_value=0.0, max_value=500.0, step=1.0, value=100.0)
        sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=15.0, step=0.1, value=1.2)
        hemo = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, step=0.1, value=15.0)
        pcv = st.number_input("Packed Cell Volume (%)", min_value=0.0, max_value=60.0, step=1.0, value=40.0)
        wc = st.number_input("White Blood Cell Count (cells/cmm)", min_value=0.0, max_value=20000.0, step=100.0, value=8000.0)
        rc = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=0.0, max_value=8.0, step=0.1, value=5.0)
        submit_button = st.form_submit_button(label="Predict")

    st.subheader("Prediction Results")
    if submit_button:
        # Create input array
        user_data = np.array([[age, bp, sg, al, su, bgr, sc, hemo, pcv, wc, rc]])
        # Scale input
        user_data_scaled = scaler.transform(user_data)
        # Predict
        prediction = rf_model.predict(user_data_scaled)[0]
        probability = rf_model.predict_proba(user_data_scaled)[0]

        # Display prediction
        if prediction == 1:
            st.error("**Prediction**: Chronic Kidney Disease")
        else:
            st.success("**Prediction**: No Chronic Kidney Disease")
        st.write(f"**Confidence**: {probability[1]:.2%} (CKD), {probability[0]:.2%} (No CKD)")

        # Plot feature importance
        feature_importance = rf_model.feature_importances_
        plt.figure(figsize=(8, 6))
        plt.barh(feature_cols, feature_importance, color='skyblue')
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance in CKD Prediction')
        st.pyplot(plt)

with tab2:
    # Data Analysis Visualizations
    with tab2:
        st.subheader("Data Analysis")

    with st.expander("Data Cleaning Log"):
        if cleaning_logs:
            for log in cleaning_logs:
                st.write(f"- {log}")
        else:
            st.write("No cleaning actions were needed.")

    # Correlation Heatmap
    st.write("**Correlation Heatmap**")
    st.write("This heatmap shows the correlation between features. Values close to 1 or -1 indicate strong relationships.")
    plt.figure(figsize=(10, 8))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    st.pyplot(plt)

    # Distribution Plots for Key Features
    st.write("**Distribution of Key Features**")
    key_features = ['sc', 'hemo', 'bgr']
    for feature in key_features:
        plt.figure(figsize=(8, 4))
        sns.histplot(X[feature], kde=True, color='skyblue')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        st.pyplot(plt)

with tab3:
    # Model Performance Visualizations
    st.subheader("Model Performance")

    # Confusion Matrix
    st.write("**Confusion Matrix**")
    st.write("This matrix shows the model's performance on the test set (True Positives, False Positives, etc.).")
    y_pred = rf_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No CKD', 'CKD'])
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# Footer note
st.markdown("---")
st.markdown("""
**Note**: This is a basic model for educational purposes. Consult a healthcare professional for accurate diagnosis.
Dataset source: https://www.kaggle.com/datasets/mansoordanish/chronic-kidney-disease
""")