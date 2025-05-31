import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

#DATA: CMAPSS NASA - Dataset para la determinacion de la vida util restante en motores a reaccion

#Columnas : 26
#ID de la maquina : 1
#Ciclo de tiempo : 2 
#Parametros de operacion : 3 - 5 
#Lecturas de sensores : 6 - 26

#Asignación de nombres a las columnas
#Las 3 utlimas columnas son descartadas por no ser relevantes para el modelo
feature_names = ['unit_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
                ['s' + str(i) for i in range(1, 22)] + \
                ['sensor_discard_' + str(i) for i in range(1, 4)]

# Función para cargar los datos
def load_data(file_paths, is_train=True):
    all_data = []
    for path in file_paths:
        df = pd.read_csv(path, sep='\s+', header=None)
        df.columns = feature_names[:len(df.columns)]
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Descartar columnas no relevantes

    if 'sensor_discard_1' in combined_df.columns:
        combined_df = combined_df.drop(columns=[col for col in combined_df.columns if col.startswith('sensor_discard_')])

    # Columnas con valores constantes (sin variación) se eliminan
    constant_sensor_cols = ['s1', 's5', 's6', 's10', 's16', 's18', 's19'] 

    # Verificar si las columnas constantes existen en el DataFrame y eliminarlas
    cols_to_drop = [col for col in constant_sensor_cols if col in combined_df.columns]
    combined_df = combined_df.drop(columns=cols_to_drop, errors='ignore')

    return combined_df

# Función para eliminar filas con valores nulos si existen
def clean_data(data):
    return data.dropna()

# Función para normalizar los datos (escalado de características)
def normalize_data(data, features_to_normalize, scaler=None, fit=True):
    if scaler is None: 
        scaler = MinMaxScaler()

    if fit: 
        data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])
    else: 
        data[features_to_normalize] = scaler.transform(data[features_to_normalize])
    return data, scaler

# Funcion para calcular la vida util restante (RUL)
def calculate_rul(df, rul_clip_value=125):
    # Calcular el ciclo maximo para cada unidad
    max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']

    #Combinar el ciclo maximo con el DataFrame original
    df = df.merge(max_cycles, on='unit_id', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']

    # Aplicar el clip a la RUL
    df['RUL'] = df['RUL'].apply(lambda x: min(x, rul_clip_value))

    #Eliminar la columna de ciclo maximo
    df = df.drop(columns=['max_cycle'])
    return df

# Función para cargar los RUL reales
def load_test_rul(file_paths):
    all_rul_dfs = [] 
    for path in file_paths:
        rul_df = pd.read_csv(path, sep='\s+', header=None)
        rul_df.columns = ['RUL']
        all_rul_dfs.append(rul_df) 
    return pd.concat(all_rul_dfs, ignore_index=True)