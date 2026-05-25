import streamlit as st
import pandas as pd
import joblib

# TÍTULO
st.title("Predicción de Cáncer de Mama")
st.write("Aplicación de Machine Learning con múltiples modelos para la clasificación benigna o maligna de masas tumorales mamarias ")

# SELECCIÓN DE MODELO
modelo_opcion = st.selectbox(
    "Selecciona un modelo",
    (
        "AdaBoost",
        "Gradient Boosting",
        "Random Forest",
        "Stacking",
        "Regresión Logística",
        "SVM",
        "Voting Hard",
        "XGBoost"
    )
)

# CARGAR MODELO SEGÚN SELECCIÓN
if modelo_opcion == "AdaBoost":
    modelo = joblib.load("best_ada_model.sav")

elif modelo_opcion == "Gradient Boosting":
    modelo = joblib.load("best_gb_model.sav")

elif modelo_opcion == "Random Forest":
    modelo = joblib.load("best_RF_modelo.sav")

elif modelo_opcion == "Stacking":
    modelo = joblib.load("best_stack_model.sav")

elif modelo_opcion == "Regresión Logística":
    modelo = joblib.load("mejor_RL_modelo.sav")

elif modelo_opcion == "SVM":
    modelo = joblib.load("mejor_SVM_modelo.sav")

elif modelo_opcion == "Voting Hard":
    modelo = joblib.load("voting_hard.sav")

elif modelo_opcion == "XGBoost":
    modelo = joblib.load("xgb_model.sav")

# INPUTS
radius_worst = st.number_input("Radius Worst, (xx.xx)", value=0.0)
texture_worst = st.number_input("Texture Worst, (xx.xx)", value=0.0)
area_worst = st.number_input("Area Worst,(xxxx.xx)", value=0.0)
concave_points_worst = st.number_input("Concave Points Worst, (0.xxxx)", value=0.0)
compactness_worst = st.number_input("Compactness Worst, (0.xxxx)", value=0.0)
smoothness_worst = st.number_input("Smoothness Worst, (0.xxxx)", value=0.0)
symmetry_worst = st.number_input("Symmetry Worst, (0.xxxx)", value=0.0)

# BOTÓN
if st.button("Predecir"):

    datos = pd.DataFrame({
        'radius_worst': [radius_worst],
        'texture_worst': [texture_worst],
        'area_worst': [area_worst],
        'concave points_worst': [concave_points_worst],
        'compactness_worst': [compactness_worst],
        'smoothness_worst': [smoothness_worst],
        'symmetry_worst': [symmetry_worst]
    })

    prediccion = modelo.predict(datos)

    if prediccion[0] == 1:
        st.error("Resultado: Maligno")
    else:
        st.success("Resultado: Benigno")