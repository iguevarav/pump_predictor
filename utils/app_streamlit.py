import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# Para normalizar los datos
from sklearn.preprocessing import MinMaxScaler

# Cargar el modelo entrenado
modelo = joblib.load("models/modelo_bombas.pkl")

# Columnas esperadas
columnas = ['footfall', 'tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature']

# Rangos usados para normalizar (del entrenamiento)
rangos = {
    'footfall': (0, 7300),
    'tempMode': (0, 7),
    'AQ': (1, 7),
    'USS': (1, 7),
    'CS': (1, 7),
    'VOC': (0, 6),
    'RP': (19, 91),
    'IP': (1, 7),
    'Temperature': (1, 24)
}

# Función para normalizar dict (modo manual)
def normalizar_dict(datos):
    return {
        col: (valor - rangos[col][0]) / (rangos[col][1] - rangos[col][0])
        for col, valor in datos.items()
    }

# Función para normalizar CSV
def normalizar_csv(df, columnas):
    df_norm = df.copy()
    for col in columnas:
        min_val, max_val = rangos[col]
        df_norm[col] = (df[col] - min_val) / (max_val - min_val)
    return df_norm

# Inicializar historial si no existe
if "historial_manual" not in st.session_state:
    st.session_state.historial_manual = []

# Configuración inicial
st.set_page_config(page_title=" Predicción de Fallas", page_icon="🛠️")
st.title("Predicción de Fallas en Bombas Centrífugas")

# Selección de modo
modo = st.radio("Selecciona el modo de entrada:", ["🎛️ Manual (datos reales)", "📁 Archivo CSV"])

# ------------------- MODO MANUAL -------------------
if modo == "🎛️ Manual (datos reales)":
    st.markdown("### Ingresar valores reales de sensores (sin normalizar)")

    valores = {}
    valores['footfall'] = st.number_input("👣 Footfall", min_value=0, max_value=7300, value=100)
    valores['tempMode'] = st.number_input("🌡️ TempMode", min_value=0, max_value=7, value=3)
    valores['AQ'] = st.number_input("💨 Calidad del Aire (AQ)", min_value=1, max_value=7, value=4)
    valores['USS'] = st.number_input("📏 Proximidad (USS)", min_value=1, max_value=7, value=3)
    valores['CS'] = st.number_input("⚡ Corriente (CS)", min_value=1, max_value=7, value=4)
    valores['VOC'] = st.number_input("🧪 VOC", min_value=0, max_value=6, value=2)
    valores['RP'] = st.number_input("🔄 RPM", min_value=19, max_value=91, value=40)
    valores['IP'] = st.number_input("💧 Presión Entrada (IP)", min_value=1, max_value=7, value=4)
    valores['Temperature'] = st.number_input("🌡️ Temperatura", min_value=1, max_value=24, value=14)

    if st.button("🔍 Predecir Falla (Manual)"):
        valores_norm = normalizar_dict(valores)
        entrada_df = pd.DataFrame([valores_norm])
        pred = modelo.predict(entrada_df)[0]
        proba = modelo.predict_proba(entrada_df)[0][1]

        resultado = valores.copy()
        resultado['Predicción'] = int(pred)
        resultado['Probabilidad de Falla'] = round(proba, 4)
        st.session_state.historial_manual.append(resultado)

        st.subheader("📊 Resultado de esta predicción:")
        if pred == 1:
            st.error(f"⚠️ ¡Falla predicha! (Probabilidad: {proba:.2%})")
        else:
            st.success(f"✅ No se predice falla. (Probabilidad de falla: {proba:.2%})")

    if st.session_state.historial_manual:
        st.markdown("### 📋 Datos Ingresados")
        historial_df = pd.DataFrame(st.session_state.historial_manual)
        st.dataframe(historial_df)

        csv_result = historial_df.to_csv(index=False).encode("utf-8")
        fecha_actual = datetime.now().strftime("%d-%m-%Y_%H:%M")
        nombre_archivo = f"data_{fecha_actual}.csv"

        st.download_button("📥 Descargar datos en CSV", data=csv_result, file_name=nombre_archivo)



# ------------------- MODO CSV -------------------
elif modo == "📁 Archivo CSV":
    st.markdown("### Carga un archivo `.csv` con las siguientes columnas:")
    st.code(", ".join(columnas), language="csv")

    archivo = st.file_uploader("Selecciona tu archivo CSV", type="csv")

    if archivo:
        try:
            df = pd.read_csv(archivo)

            if all(col in df.columns for col in columnas):
                st.success("✅ Archivo válido. Listo para predecir.")

                if st.button("🔍 Predecir Falla (CSV)"):
                    df_normalizado = normalizar_csv(df, columnas)
                    predicciones = modelo.predict(df_normalizado[columnas])
                    probabilidades = modelo.predict_proba(df_normalizado[columnas])[:, 1]

                    df_resultado = df.copy()
                    df_resultado["Predicción"] = predicciones
                    df_resultado["Probabilidad de Falla"] = np.round(probabilidades, 4)

                    st.subheader("📊 Resultados del archivo:")
                    st.dataframe(df_resultado)

                    csv_result = df_resultado.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Descargar resultados CSV", data=csv_result, file_name="resultados_predicciones.csv")
            else:
                st.error("❌ El archivo no contiene todas las columnas requeridas.")
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {e}")
