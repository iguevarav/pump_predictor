import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
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

# ------------------- CARGA DE CSV -------------------
st.markdown("### Carga un archivo `.csv` con las siguientes columnas:")
st.code(", ".join(columnas), language="csv")

archivo = st.file_uploader("Selecciona tu archivo CSV", type="csv")

if archivo:
    try:
        df = pd.read_csv(archivo)

        # Verificar si las columnas necesarias están presentes
        if all(col in df.columns for col in columnas):
            st.success("✅ Archivo válido. Listo para predecir.")

            if st.button("🔍 Predecir Falla (CSV)"):
                # Mostrar una barra de progreso mientras se realiza el análisis
                progreso = st.progress(0)
                df_normalizado = normalizar_csv(df, columnas)
                progreso.progress(25)

                predicciones = modelo.predict(df_normalizado[columnas])
                progreso.progress(50)

                probabilidades = modelo.predict_proba(df_normalizado[columnas])[:, 1]
                progreso.progress(75)

                df_resultado = df.copy()
                df_resultado["Predicción"] = predicciones
                df_resultado["Probabilidad de Falla"] = np.round(probabilidades, 4)
                progreso.progress(100)

                st.subheader("📊 Resultados del archivo:")
                st.dataframe(df_resultado)

                # -------------------- Primer gráfico: Serie Temporal de Probabilidad de Falla (Plotly) --------------------
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=df_resultado['timestamp'],
                    y=df_resultado['Probabilidad de Falla'],
                    mode='lines',
                    name='Probabilidad de Falla',
                    line=dict(color='red')
                ))

                fig.update_layout(
                    title="Probabilidad de Falla a lo largo del tiempo",
                    xaxis_title="Hora",
                    yaxis_title="Probabilidad de Falla",
                    xaxis=dict(tickformat="%H:%M"),
                    template="plotly_dark"
                )

                st.plotly_chart(fig)

                # Cálculo de la probabilidad promedio de falla
                probabilidad_media = np.mean(df_resultado['Probabilidad de Falla'])
                st.metric("Probabilidad general de falla", f"{probabilidad_media * 100:.2f}%", delta=f"{(probabilidad_media * 100 - 50):.2f}%")

                # Estado de la máquina y semáforo
                if probabilidad_media >= 0.7:
                    estado = "⚠️ **Peligro de Falla**"
                    color_estado = "red"
                    semaforo = "🟥"
                elif probabilidad_media >= 0.4:
                    estado = "🔶 **Mantenimiento Necesario**"
                    color_estado = "orange"
                    semaforo = "🟧"
                else:
                    estado = "✅ **Máquina Operativa**"
                    color_estado = "green"
                    semaforo = "🟩"

                st.markdown(f"**Estado de la máquina:** {estado} {semaforo}", unsafe_allow_html=True)

                # Descargar los resultados como CSV
                csv_result = df_resultado.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Descargar resultados CSV", data=csv_result, file_name="resultados_predicciones.csv")
        else:
            st.error("❌ El archivo no contiene todas las columnas requeridas.")
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")

# ------------------- INGRESO MANUAL DE DATOS -------------------
st.markdown("### Ingreso Manual de Datos 📥")

# Crear un apartado exclusivo para ingresar datos manualmente
st.info("Puedes ingresar los valores manualmente para predecir la probabilidad de falla de la máquina. Ingresa los datos en los campos siguientes:")

# Crear formularios interactivos con detalles visuales y iconos
with st.form(key="manual_form"):
    st.markdown("### Ingresar datos para predicción:")
    footfall = st.number_input("👥 Número de personas (footfall)", min_value=0, step=1, help="Cantidad de personas detectadas en la máquina")
    tempMode = st.number_input("🌡️ Modo de temperatura (tempMode)", min_value=0, max_value=7, step=1, help="Modo de temperatura de la máquina")
    AQ = st.number_input("💨 Calidad del aire (AQ)", min_value=1, max_value=7, step=1, help="Calificación de calidad del aire")
    USS = st.number_input("⚙️ Nivel de desgaste (USS)", min_value=1, max_value=7, step=1, help="Nivel de desgaste de la máquina")
    CS = st.number_input("🔒 Condiciones de seguridad (CS)", min_value=1, max_value=7, step=1, help="Condiciones de seguridad de la máquina")
    VOC = st.number_input("💨 Comp. orgánicos volátiles (VOC)", min_value=0, max_value=6, step=1, help="Nivel de compuestos volátiles")
    RP = st.number_input("🔧 Rendimiento de la máquina (RP)", min_value=19, max_value=91, step=1, help="Rendimiento de la máquina")
    IP = st.number_input("⚡ Índice de presión (IP)", min_value=1, max_value=7, step=1, help="Índice de presión de la máquina")
    Temperature = st.number_input("🌡️ Temperatura (Temperature)", min_value=1, max_value=24, step=1, help="Temperatura en grados Celsius")

    # Cuando el usuario envía el formulario
    submit_button = st.form_submit_button(label="Predecir Falla")

    if submit_button:
        # Crear un diccionario con los valores ingresados
        datos_manual = {
            'footfall': footfall,
            'tempMode': tempMode,
            'AQ': AQ,
            'USS': USS,
            'CS': CS,
            'VOC': VOC,
            'RP': RP,
            'IP': IP,
            'Temperature': Temperature
        }

        # Normalizar los datos manuales
        datos_normalizados = normalizar_dict(datos_manual)

        # Hacer la predicción
        prediccion = modelo.predict([list(datos_normalizados.values())])
        probabilidad = modelo.predict_proba([list(datos_normalizados.values())])[:, 1][0]

        # Mostrar los resultados de la predicción
        st.subheader(f"Predicción para los datos ingresados manualmente:")
        st.write(f"Predicción: {'Falla' if prediccion[0] == 1 else 'Operativo'}")
        st.write(f"Probabilidad de Falla: {probabilidad * 100:.2f}%")

        # Mostrar el estado con semáforo
        if probabilidad >= 0.7:
            estado = "⚠️ **Peligro de Falla**"
            semaforo = "🟥"
        elif probabilidad >= 0.4:
            estado = "🔶 **Mantenimiento Necesario**"
            semaforo = "🟧"
        else:
            estado = "✅ **Máquina Operativa**"
            semaforo = "🟩"

        st.markdown(f"**Estado de la máquina:** {estado} {semaforo}", unsafe_allow_html=True)
