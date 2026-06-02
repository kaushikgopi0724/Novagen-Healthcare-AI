import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title = "Novagen AI",
    page_icon = "🏥",
    layout = "wide"
)

model = joblib.load("models/health_risk_model.pkl")
features = joblib.load("models/model_features.pkl")
explainer = shap.TreeExplainer(model)

st.title("🏥 Novagen Healthcare AI")
st.markdown(
"""
### Explainable Healthcare Risk Prediction System

This AI system predicts potential health risk
using Machine Learnign and XGBoost.
""")

st.sidebar.title("About")
st.sidebar.info("""
Novagen AI is a Machine Learning powered
healthcare risk prediction system built using: 

- XGBoost
- SHAP Explainability
- Streamlit
""")
tab1, tab2, tab3 = st.tabs([
    "Prediction",
    "Model Insights",
    "About Project"
])
with tab1: 
    col1, col2 = st.columns(2)
    
    with col1: 
        age = st.slider("Age", 1, 100, 25)
        bmi = st.slider("BMI", 10.0, 50.0, 22.0)
        blood_pressure = st.slider(
            "Blood Pressure",
            80,
            200,
            120
        )
    with col2: 
        cholesterol = st.slider(
            "Cholesterol",
            100,
            400,
            180
        )
        glucose = st.slider(
            "Glucose Level",
            50,
            300,
            100
        )
        stress = st.slider(
            "Stresss Level",
            1, 
            10,
            5
        )
    with st.expander("🩺 Advanced Health Metrics"):
        heart_rate = st.slider(
            "Heart Rate",
            40,
            180,
            72
        )
        sleep_hours = st.slider(
            "Sleep Hours",
            0.0,
            12.0,
            7.0
        )
        exercise_hours = st.slider(
            "Exercise Hours",
            0.0,
            5.0,
            1.0
        )
        water_intake = st.slider(
            "Water Intake (Liters)",
            0.0,
            10.0,
            2.5
        )
        smoking = st.selectbox(
            "Smoking",
            [0,1]
        )
        alcohol = st.selectbox(
            "Alcohol Consumption",
            [0,1]
        )
    st.divider()
    if st.button(
        "🔍 Analyze Health Risk",
        use_container_width = True        
    ):
        input_dict = dict.fromkeys(features, 0)
        
        input_dict["Age"] = age
        input_dict["BMI"] = bmi
        input_dict["Blood_Pressure"] = blood_pressure
        input_dict["Cholesterol"] = cholesterol
        input_dict["Glucose_Level"] = glucose
        input_dict["Stress_Level"] = stress
        input_dict["Heart_Rate"] = heart_rate
        input_dict["Sleep_Hours"] = sleep_hours
        input_dict["Exercise_Hours"] = exercise_hours
        input_dict["Water_Intake"] = water_intake
        input_dict["Smoking"] = smoking
        input_dict["Alcohol"] = alcohol
        
        input_data = pd.DataFrame([input_dict])
    
        prediction = model.predict(input_data)[0]
        
        probability = model.predict_proba(input_data)[0][1]
    
        shap_values = explainer.shap_values(input_data)
        
        st.subheader("Prediction Result")
    
        if prediction == 1:
            st.error(
                f"⚠️ High Health Risk ({probability*100:.2f}% confidence)"
            )
        else :
            st.success(
                f"✅ Low Health Risk ({(1-probability)*100:.2f}% confidence)"
            )
    
        st.progress(float(probability))
    
        st.metric(
            "Risk Probability",
            f"{probability*100:.2f}%"
        )
        if probability < 0.30:
            st.success("🟢 Low Risk Category")
        elif probability < 0.70:
            st.warning("🟡 Moderate Risk Category")
        else:
            st.error("🔴 High Risk Category")
    
        st.subheader("SHAP Explanation")

        st.divider()
        shap_exp = shap.Explanation(
            values = shap_values[0],
            base_values = explainer.expected_value,
            data = input_data.iloc[0],
            feature_names=input_data.columns
        )
        shap.plots.waterfall(shap_exp, show=False)
        st.pyplot(plt.gcf())

with tab2: 
    st.header("📊 Model Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "94.97%")
    with col2:
        st.metric("ROC-AUC", "0.984")
    with col3:
        st.metric("False Negative", "52")
    st.divider()
    st.write(
        """
        ### Best Performing Model
        Tuned XGBoost Classifier
        
        ### Key Features
        - BMI
        - Blood Pressure
        - Cholesterol
        - Stress Level

        ### Explainability
        SHAP Explainable AI integrated
        """
    )
with tab3: 
    st.header("ℹ️ About Novagen AI")
    st.write("""
    Novagen AI is an Explainable Healthcare
    Risk Prediction System built using:

    - XGBoost
    - SHAP Explainability
    - Streamlit

    The system predicts potential health risks
    using patient health metrics.
    """)







