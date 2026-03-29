import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predicción Mn", page_icon="🔬", layout="centered")

st.title("🔬 Predicción de Concentración de Manganeso")
st.subheader("Electroobtención de Zinc - Modelo XGBoost")

# Cargar modelos
@st.cache_resource
def load_models():
    modelo = joblib.load('modelo_xgboost_mn.pkl')
    scaler = joblib.load('scaler_mn.pkl')
    detector = joblib.load('detector_outliers_mn.pkl')
    features = joblib.load('features_mn.pkl')
    return modelo, scaler, detector, features

modelo, scaler, detector_outliers, features = load_models()

st.sidebar.header("Parámetros de Entrada")

temp = st.sidebar.slider("Temperatura (°C)", 31.0, 46.0, 38.0)
acidez = st.sidebar.slider("Acidez (%)", 161.0, 189.0, 175.0)
ph = st.sidebar.slider("pH Electrolito", 0.8, 2.2, 1.5)
densidad = st.sidebar.slider("Densidad (g/L)", 1285.0, 1305.0, 1295.0)
zn_ea = st.sidebar.slider("Zn EA (g/L)", 38.0, 48.0, 42.0)
peso = st.sidebar.slider("Peso Depósito (kg)", 75.0, 90.0, 82.0)

zn_pura = st.sidebar.number_input("Zn Sol Pura (g/L)", 145.0, 160.0, 150.0)
fe = st.sidebar.number_input("Fe (mg/L)", 0.0, 10.0, 3.0)
sb = st.sidebar.number_input("Sb (mg/L)", 0.0, 0.1, 0.03)
cu = st.sidebar.number_input("Cu (mg/L)", 0.0, 0.5, 0.1)

if st.button("Predecir Concentración de Mn", type="primary"):
    input_data = {
        'T de EA (°C)': temp,
        'Zn Sol Pura (g/L)': zn_pura,
        'Fe (mg/L)': fe,
        'Sb (mg/L)': sb,
        'Cu (mg/L)': cu,
        'Densidad (g/L)': densidad,
        'Zn EA (g/L)': zn_ea,
        'Acidez (%)': acidez,
        'Horas de depósito': 48,
        'Peso Depósito (kg)': peso,
        'pH_electrolito': ph
    }

    df_input = pd.DataFrame([input_data])
    df_input = df_input[features]

    scaled_input = scaler.transform(df_input)
    prediccion = modelo.predict(scaled_input)[0]
    es_outlier = detector_outliers.predict(scaled_input)[0]

    st.success(f"**Concentración de Mn predicha: {prediccion:.2f} g/L**")

    if es_outlier == 1:
        st.info("✅ Parámetros dentro del rango de datos. Predicción confiable.")
    else:
        st.warning("⚠️ Parámetros fuera del rango. La predicción puede no ser confiable.")

    with st.expander("Ver parámetros ingresados"):
        st.json(input_data)

st.caption("Modelo XGBoost desarrollado para la tesis de Adriana Cardenas Carbajal")