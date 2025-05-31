import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler 

st.set_page_config(
    page_title="Predicción de RUL (C-MAPSS Turbofan)",
    page_icon="✈️",
    layout="wide"
)

# --------------------------------------------------------------------------------
# CONFIGURACIÓN ESPECÍFICA PARA C-MAPSS (FD001)
# --------------------------------------------------------------------------------

MODEL_PATH = "models/modelo1.pkl"
try:
    modelo = joblib.load(MODEL_PATH)
    st.sidebar.success(f"Modelo cargado: {MODEL_PATH}")
except FileNotFoundError:
    st.sidebar.error(f"Error: Modelo no encontrado en {MODEL_PATH}. Asegúrate de haberlo entrenado y guardado correctamente.")
    st.stop() 

# IMPORTANTE: Características que el modelo espera (basado en el entrenamiento)
# El modelo fue entrenado solo con estas características específicas
feature_cols_model = ['setting1', 'setting2', 'setting3', 's2', 's3', 's4', 's7', 's8', 's9', 
                      's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']

# Todas las columnas en el archivo C-MAPSS original
all_sensor_cols = ['setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]

# Columnas para archivos de TEST (sin RUL)
cmaps_cols_test = ['unit_id', 'cycle'] + all_sensor_cols

# Columnas para archivos de ENTRENAMIENTO (con RUL)
cmaps_cols_train = cmaps_cols_test + ['RUL']

# RANGOS DE NORMALIZACIÓN PARA LAS CARACTERÍSTICAS USADAS EN EL MODELO
# Solo incluimos los rangos para las características que el modelo usa
cmaps_ranges = {
    'setting1': (-0.008, 0.008),      # op_setting_1
    'setting2': (-0.0006, 0.0006),    # op_setting_2
    'setting3': (100.0, 100.0),       # op_setting_3
    's2': (641.71, 644.52),           # sensor_2
    's3': (1571.71, 1616.91),         # sensor_3
    's4': (1384.39, 1441.47),         # sensor_4
    's7': (549.85, 556.09),           # sensor_7
    's8': (2387.90, 2388.56),         # sensor_8
    's9': (9039.69, 9221.76),         # sensor_9
    's11': (47.01, 48.53),            # sensor_11
    's12': (518.67, 523.57),          # sensor_12
    's13': (2387.89, 2388.81),        # sensor_13
    's14': (8110.14, 8293.72),        # sensor_14
    's15': (8.3244, 8.5724),          # sensor_15
    's17': (390, 400),                # sensor_17
    's20': (38.41, 39.46),            # sensor_20
    's21': (22.9564, 23.8202)         # sensor_21
}

# Función para normalizar un DataFrame completo usando los rangos predefinidos
def normalizar_df_cmaps(df, feature_cols, ranges):
    df_norm = df.copy()
    for col in feature_cols:
        if col in ranges:
            min_val, max_val = ranges[col]
            if max_val == min_val: # Evitar división por cero si la columna es constante
                df_norm[col] = 0.0 # O cualquier valor consistente con tu normalización de constantes
            else:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            st.warning(f"La columna '{col}' no tiene rangos de normalización definidos. Se mantendrá sin normalizar.")
    return df_norm

# --------------------------------------------------------------------------------
# INTERFAZ DE STREAMLIT
# --------------------------------------------------------------------------------

st.title("✈️ Predicción de Vida Útil Restante (RUL) de Motores Turbofan")
st.markdown("Carga archivos de prueba o entrenamiento de C-MAPSS (`.txt` o `.csv`) para obtener predicciones de RUL por cada motor.")

# Información sobre las columnas
with st.expander("ℹ️ Información sobre el modelo y las características"):
    st.markdown(f"""
    **El modelo fue entrenado con las siguientes características:**
    - `setting1, setting2, setting3`: Configuraciones operacionales
    - Sensores utilizados: `s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21`
    
    **Nota:** El modelo no utiliza todos los sensores del dataset C-MAPSS, solo un subconjunto seleccionado.
    
    **Estructura del archivo esperada:**
    - Debe contener todas las columnas C-MAPSS estándar (26 o 27 columnas)
    - El modelo extraerá automáticamente las características que necesita
    """)

# ------------------- CARGA DE ARCHIVO -------------------
st.markdown("### Carga un archivo de datos (`.txt` o `.csv`)")

archivo_cargado = st.file_uploader("Selecciona tu archivo", type=["txt", "csv"])

if archivo_cargado:
    # Limpiar session_state cuando se carga un nuevo archivo
    if 'ultimo_archivo' not in st.session_state or st.session_state.ultimo_archivo != archivo_cargado.name:
        st.session_state.clear()
        st.session_state.ultimo_archivo = archivo_cargado.name
    
    try:
        # Detectar el separador automáticamente
        if archivo_cargado.name.endswith('.txt'):
            # Leer el archivo sin asignar columnas todavía
            df_raw = pd.read_csv(archivo_cargado, sep='\s+', header=None)
            
            # Determinar si es archivo de entrenamiento o prueba por el número de columnas
            num_columnas = df_raw.shape[1]
            
            if num_columnas == 26:  # Archivo de TEST
                df_raw.columns = cmaps_cols_test
                st.info("📄 Archivo detectado como archivo de PRUEBA (sin columna RUL)")
            elif num_columnas == 27:  # Archivo de ENTRENAMIENTO
                df_raw.columns = cmaps_cols_train
                st.info("📄 Archivo detectado como archivo de ENTRENAMIENTO (con columna RUL)")
            else:
                st.error(f"❌ El archivo tiene {num_columnas} columnas. Se esperaban 26 (prueba) o 27 (entrenamiento) columnas.")
                st.stop()
                
        else: # Asumir CSV, que es delimitado por comas
            df_raw = pd.read_csv(archivo_cargado)
            
            # Verificar si el CSV ya tiene los nombres de columnas correctos
            if 'unit_id' not in df_raw.columns or 'cycle' not in df_raw.columns:
                # Si no tiene headers, asignar basándose en el número de columnas
                if df_raw.shape[1] == 26:
                    df_raw.columns = cmaps_cols_test
                elif df_raw.shape[1] == 27:
                    df_raw.columns = cmaps_cols_train
                else:
                    st.error("❌ El archivo CSV no tiene el número correcto de columnas.")
                    st.stop()

        # Verificar si las columnas necesarias para el modelo están presentes
        missing_cols = [col for col in feature_cols_model if col not in df_raw.columns]
        if missing_cols:
            st.error(f"❌ El archivo no contiene las siguientes columnas requeridas por el modelo: {', '.join(missing_cols)}")
            st.stop()
        
        st.success("✅ Archivo válido. Todas las características requeridas están presentes.")
        
        # Mostrar estadísticas básicas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Número de motores", df_raw['unit_id'].nunique())
        with col2:
            st.metric("Total de registros", len(df_raw))
        with col3:
            st.metric("Máximo ciclo", df_raw['cycle'].max())
        
        # Mostrar las primeras filas (solo columnas relevantes)
        st.subheader("Vista previa de los datos:")
        display_cols = ['unit_id', 'cycle'] + feature_cols_model
        if 'RUL' in df_raw.columns:
            display_cols.append('RUL')
        st.dataframe(df_raw[display_cols].head(10))
        
        # Si es archivo de entrenamiento, mostrar también el RUL real
        if 'RUL' in df_raw.columns:
            st.info("ℹ️ Este archivo contiene valores RUL reales que se pueden comparar con las predicciones.")

        if st.button("🔍 Predecir RUL", type="primary"):
            with st.spinner("Realizando predicciones... Esto puede tomar un momento."):
                progreso = st.progress(0)

                # Extraer solo las características que el modelo necesita
                df_features = df_raw[feature_cols_model].copy()
                progreso.progress(20)

                # Normalizar los datos de características
                df_normalizado = normalizar_df_cmaps(df_features, feature_cols_model, cmaps_ranges)
                progreso.progress(40)

                # Realizar predicciones
                predicciones_rul = modelo.predict(df_normalizado)
                progreso.progress(80)

                # Asegurar que las predicciones de RUL no sean negativas
                predicciones_rul = np.maximum(0, predicciones_rul)

                df_resultado = df_raw[['unit_id', 'cycle']].copy()
                df_resultado["RUL_Predicha"] = predicciones_rul
                
                # Si hay RUL real, agregarlo también
                if 'RUL' in df_raw.columns:
                    df_resultado['RUL'] = df_raw['RUL']
                
                # Guardar resultados en session_state
                st.session_state.df_resultado = df_resultado
                st.session_state.predicciones_realizadas = True
                
                progreso.progress(100)

            st.success("✅ Predicciones completadas")
        
        # Mostrar resultados si existen en session_state
        if 'predicciones_realizadas' in st.session_state and st.session_state.predicciones_realizadas:
            df_resultado = st.session_state.df_resultado
            
            # Tabs para organizar los resultados
            tab1, tab2, tab3 = st.tabs(["📊 Tabla de Resultados", "📈 Visualización", "📝 Resumen por Motor"])
            
            with tab1:
                st.subheader("Resultados de RUL Predicha por ciclo:")
                
                # Si hay RUL real, mostrar comparación
                if 'RUL' in df_resultado.columns:
                    df_resultado['Error'] = df_resultado['RUL'] - df_resultado['RUL_Predicha']
                    df_resultado['Error_Absoluto'] = np.abs(df_resultado['Error'])
                    columnas_mostrar = ['unit_id', 'cycle', 'RUL', 'RUL_Predicha', 'Error', 'Error_Absoluto']
                else:
                    columnas_mostrar = ['unit_id', 'cycle', 'RUL_Predicha']
                
                # Filtro por motor
                motor_seleccionado = st.selectbox(
                    "Filtrar por motor (opcional):",
                    ["Todos"] + list(df_resultado['unit_id'].unique())
                )
                
                if motor_seleccionado != "Todos":
                    df_mostrar = df_resultado[df_resultado['unit_id'] == motor_seleccionado]
                else:
                    df_mostrar = df_resultado
                
                st.dataframe(df_mostrar[columnas_mostrar], use_container_width=True)

            with tab2:
                st.subheader("RUL Predicha por Unidad a lo largo del Ciclo")
                
                # Selector de motores para visualizar
                unique_units = df_resultado['unit_id'].unique()
                
                # Inicializar el estado si no existe
                if 'selected_units' not in st.session_state:
                    if len(unique_units) > 10:
                        st.session_state.selected_units = list(unique_units[:5])
                    else:
                        st.session_state.selected_units = list(unique_units)
                
                # Widget de selección con key único
                selected_units = st.multiselect(
                    "Selecciona los motores a visualizar (máx 10):",
                    options=unique_units,
                    default=st.session_state.selected_units,
                    max_selections=10,
                    key="motor_selector"
                )
                
                # Actualizar el estado
                st.session_state.selected_units = selected_units
                
                if selected_units:  # Solo graficar si hay motores seleccionados
                    fig = go.Figure()

                    for unit_id in selected_units:
                        df_unit = df_resultado[df_resultado['unit_id'] == unit_id].sort_values('cycle')
                        
                        # Agregar línea de RUL predicha
                        fig.add_trace(go.Scatter(
                            x=df_unit['cycle'],
                            y=df_unit['RUL_Predicha'],
                            mode='lines+markers',
                            name=f'Motor {unit_id} (Predicha)',
                            hovertemplate=f"Motor: {unit_id}<br>Ciclo: %{{x}}<br>RUL Predicha: %{{y:.2f}}<extra></extra>"
                        ))
                        
                        # Si hay RUL real, agregar también esa línea
                        if 'RUL' in df_unit.columns:
                            fig.add_trace(go.Scatter(
                                x=df_unit['cycle'],
                                y=df_unit['RUL'],
                                mode='lines',
                                name=f'Motor {unit_id} (Real)',
                                line=dict(dash='dash'),
                                hovertemplate=f"Motor: {unit_id}<br>Ciclo: %{{x}}<br>RUL Real: %{{y:.2f}}<extra></extra>"
                            ))

                    fig.update_layout(
                        title="RUL vs Ciclo por Motor",
                        xaxis_title="Ciclo",
                        yaxis_title="RUL (Vida Útil Restante)",
                        hovermode="x unified",
                        template="plotly_white",
                        height=600,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("👆 Selecciona al menos un motor para visualizar")

            with tab3:
                st.subheader("Resumen de RUL para el Último Ciclo de Cada Motor")
                
                # Encontrar el último ciclo de cada unidad
                df_last_cycle = df_resultado.loc[df_resultado.groupby('unit_id')['cycle'].idxmax()].reset_index(drop=True)
                
                if 'RUL' in df_last_cycle.columns:
                    # Mostrar métricas de error
                    col1, col2, col3 = st.columns(3)
                    mae = df_last_cycle['Error_Absoluto'].mean()
                    rmse = np.sqrt((df_last_cycle['Error'] ** 2).mean())
                    
                    # Evitar división por cero en MAPE
                    df_last_cycle['MAPE_individual'] = np.where(
                        df_last_cycle['RUL'] > 0,
                        (df_last_cycle['Error_Absoluto'] / df_last_cycle['RUL']) * 100,
                        np.nan
                    )
                    mape = df_last_cycle['MAPE_individual'].mean()
                    
                    with col1:
                        st.metric("MAE", f"{mae:.2f} ciclos")
                    with col2:
                        st.metric("RMSE", f"{rmse:.2f} ciclos")
                    with col3:
                        st.metric("MAPE", f"{mape:.1f}%")
                    
                    # Tabla resumen
                    columnas_resumen = ['unit_id', 'cycle', 'RUL', 'RUL_Predicha', 'Error', 'Error_Absoluto']
                    st.dataframe(
                        df_last_cycle[columnas_resumen].style.highlight_min(
                            subset=['Error_Absoluto'], color='lightgreen'
                        ).highlight_max(
                            subset=['Error_Absoluto'], color='lightcoral'
                        ),
                        use_container_width=True
                    )
                    
                    # Gráfico de dispersión: Predicho vs Real
                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(go.Scatter(
                        x=df_last_cycle['RUL'],
                        y=df_last_cycle['RUL_Predicha'],
                        mode='markers',
                        marker=dict(size=10, color='blue', opacity=0.6),
                        text=df_last_cycle['unit_id'],
                        hovertemplate="Motor: %{text}<br>RUL Real: %{x}<br>RUL Predicha: %{y}<extra></extra>"
                    ))
                    
                    # Línea diagonal perfecta
                    max_val = max(df_last_cycle['RUL'].max(), df_last_cycle['RUL_Predicha'].max())
                    fig_scatter.add_trace(go.Scatter(
                        x=[0, max_val],
                        y=[0, max_val],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Predicción Perfecta'
                    ))
                    
                    fig_scatter.update_layout(
                        title="RUL Predicha vs RUL Real (Último Ciclo)",
                        xaxis_title="RUL Real",
                        yaxis_title="RUL Predicha",
                        template="plotly_white",
                        height=500
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.dataframe(df_last_cycle[['unit_id', 'cycle', 'RUL_Predicha']], use_container_width=True)

            # Descargar los resultados como CSV
            st.markdown("---")
            csv_result = df_resultado.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Descargar todos los resultados (CSV)",
                data=csv_result,
                file_name=f"predicciones_rul_{archivo_cargado.name.split('.')[0]}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {str(e)}")
        with st.expander("Ver detalles del error"):
            st.code(str(e))
            st.markdown("**Sugerencias:**")
            st.markdown("""
            - Verifica que el archivo tenga 26 columnas (prueba) o 27 columnas (entrenamiento)
            - Para archivos .txt, asegúrate de que estén separados por espacios
            - Para archivos .csv, verifica que estén separados por comas
            - Las columnas deben seguir el formato estándar de C-MAPSS
            """)