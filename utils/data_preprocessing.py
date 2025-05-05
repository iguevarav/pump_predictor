import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Función para cargar los datos
def load_data(file_path):
    return pd.read_csv(file_path)

# Función para limpiar los datos (eliminar valores nulos)
def clean_data(data):
    return data.dropna()

# Función para normalizar los datos (escalado de características)
def normalize_data(data):
    scaler = MinMaxScaler()
    # Normalización de las características
    data[['footfall', 'tempMode', 'AQ', 'VOC', 'Temperature', 'USS', 'CS', 'RP', 'IP']] = scaler.fit_transform(
        data[['footfall', 'tempMode', 'AQ', 'VOC', 'Temperature', 'USS', 'CS', 'RP', 'IP']])
    return data