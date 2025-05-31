import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar funciones de los diferentes módulos

from utils.data_preprocessing import load_data, clean_data, normalize_data, calculate_rul, load_test_rul
from utils.visualization import plot_feature_importance, plot_correlation_heatmap, plot_feature_correlation # Asegúrate de que estas funciones sean compatibles con series de tiempo/RUL
from models.model import train_random_forest, grid_search_random_forest, evaluate_with_cross_validation, final_evaluation, save_model

base_data_path = r'D:\UNT CICLOS\VII CICLO\GESTION DE TI\PROYECTO - I UNIDAD\PUMP_PREDICTOR\data' 

train_files = [os.path.join(base_data_path, 'train_FD001.txt')]
test_files = [os.path.join(base_data_path, 'test_FD001.txt')]
rul_files = [os.path.join(base_data_path, 'RUL_FD001.txt')]


#1. CARGAR Y EXPLORAR LOS DATOS
print("Cargando y combinando datos de entrenamiento...")
train_df = load_data(train_files, is_train=True)
print(f"Tamaño del dataset de entrenamiento combinado: {train_df.shape}")
print("Columnas del dataset de entrenamiento:", train_df.columns.tolist())

#2: CALCULAR LA RUL PARA EL DATASET DE ENTRENAMIENTO
print("Calculando RUL para el dataset de entrenamiento...")
train_df = calculate_rul(train_df, rul_clip_value=125) # Puedes ajustar el valor de clipping
print(f"Columnas después de calcular RUL: {train_df.columns.tolist()}")

#3: IDENTIFICAR CARACTERÍSTICAS Y NORMALIZAR LOS DATOS
feature_cols_to_normalize = [col for col in train_df.columns if col not in ['unit_id', 'cycle', 'RUL']]
print(f"Características a normalizar: {feature_cols_to_normalize}")

#4: NORMALIZAR LOS DATOS DE ENTRENAMIENTO
print("Normalizando datos de entrenamiento...")
train_df_normalized, scaler = normalize_data(train_df.copy(), feature_cols_to_normalize, fit=True)
print("Normalización de entrenamiento completa.")

# 5. CARGAR Y PREPARAR LOS DATOS DE PRUEBA
print("Cargando y preparando datos de prueba...")
test_df_raw = load_data(test_files, is_train=False)
true_rul_df = load_test_rul(rul_files)

test_df_last_cycle = test_df_raw.groupby('unit_id').last().reset_index()

# Normalizar los datos de prueba usando el scaler ENTRENADO con los datos de entrenamiento
test_df_normalized, _ = normalize_data(test_df_last_cycle.copy(), feature_cols_to_normalize, scaler=scaler, fit=False)
print("Datos de prueba preparados.")
print(f"Tamaño del dataset de prueba (último ciclo de cada unidad): {test_df_normalized.shape}")

# 6. VISUALIZACIÓN DE CARACTERÍSTICAS
print("\nGenerando visualizaciones...")
# Asegúrate de que 'train_df_normalized' contenga 'RUL' para las funciones de correlación
plot_feature_correlation(train_df_normalized) 
plot_correlation_heatmap(train_df_normalized)

# 7. ENTRENAR EL MODELO
print("\nIniciando entrenamiento del modelo Random Forest Regressor...")
# train_random_forest ahora devuelve X, y de entrenamiento para el proceso de validación
model, X_train_full, y_train_full = train_random_forest(train_df_normalized)
print("Entrenamiento inicial completo.")

# 8. AJUSTE DE HIPERPARÁMETROS CON GRID SEARCH
print("\nRealizando ajuste de hiperparámetros con Grid Search...")
best_model = grid_search_random_forest(X_train_full, y_train_full) 
print("Ajuste de hiperparámetros completo.")


# 9. EVALUAR EL MODELO USANDO VALIDACIÓN CRUZADA EN EL CONJUNTO DE ENTRENAMIENTO
print("\nEvaluando el modelo con validación cruzada...")
evaluate_with_cross_validation(best_model, X_train_full, y_train_full)
print("Evaluación con validación cruzada completa.")


# 10. EVALUAR EL MODELO FINAL SOBRE EL CONJUNTO DE PRUEBA
print("\nEvaluando el modelo final sobre el conjunto de prueba...")
# Pasamos el DataFrame de prueba ya preprocesado (con las últimas filas) y las RUL reales
final_evaluation(best_model, test_df_normalized, true_rul_df)
print("Evaluación final completa.")



# 11. GUARDAR MODELO ENTRENADO
save_model(best_model)















