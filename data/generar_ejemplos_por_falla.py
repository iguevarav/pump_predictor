import pandas as pd
import joblib
import numpy as np

# Cargar archivo
df = pd.read_csv("data/data.csv")
modelo = joblib.load("models/modelo_bombas.pkl")

# Rangos de entrenamiento
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
columnas = list(rangos.keys())

def normalizar_csv(df, columnas):
    df_norm = df.copy()
    for col in columnas:
        min_val, max_val = rangos[col]
        df_norm[col] = (df[col] - min_val) / (max_val - min_val)
    return df_norm

# Normalizar y predecir
df_norm = normalizar_csv(df, columnas)
df['Probabilidad'] = modelo.predict_proba(df_norm[columnas])[:, 1]
df['Predicción'] = modelo.predict(df_norm[columnas])

# Separar en 3 grupos
bajo = df[df['Probabilidad'] < 0.2].head(5)
medio = df[(df['Probabilidad'] >= 0.39) & (df['Probabilidad'] <= 0.41)].head(5)
alto = df[df['Probabilidad'] > 0.5].head(5)

# Guardar como CSV
bajo.to_csv("data/falla_baja.csv", index=False)
medio.to_csv("data/falla_40.csv", index=False)
alto.to_csv("data/falla_alta.csv", index=False)

print("✅ Archivos generados: falla_baja.csv, falla_40.csv, falla_alta.csv")
