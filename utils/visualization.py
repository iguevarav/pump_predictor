import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor # CAMBIO: Regressor para RUL
import numpy as np # Para np.corrcoef si es necesario, o solo df.corr()

# Define las columnas que NO son características, pero son importantes para el contexto del dataset
NON_FEATURE_COLS = ['unit_id', 'cycle', 'RUL'] # RUL será la variable objetivo

# Función para mostrar la correlación de las características con la RUL
def plot_feature_correlation(data_df):

    if 'RUL' not in data_df.columns:
        print("Error: La columna 'RUL' no se encuentra en el DataFrame. No se puede calcular la correlación con RUL.")
        return

    # Excluir las columnas que no son características ni la RUL
    feature_cols = [col for col in data_df.columns if col not in NON_FEATURE_COLS]

    # Calcular la correlación de las características con 'RUL'
    rul_correlacion = data_df[feature_cols + ['RUL']].corr()['RUL'].drop('RUL').sort_values(ascending=False)

    # Visualización
    plt.figure(figsize=(12, 7)) 
    sns.barplot(x=rul_correlacion.index, y=rul_correlacion.values, palette='viridis') 
    plt.title('Correlación de las Características con la RUL (Remaining Useful Life)')
    plt.xlabel('Características')
    plt.ylabel('Coeficiente de Correlación (Pearson)')
    plt.xticks(rotation=60, ha='right') 
    plt.tight_layout()
    plt.show()

# Función para mostrar el mapa de calor de correlación entre las características
def plot_correlation_heatmap(data_df):

    # Excluir las columnas que no son características (incluyendo RUL, si no quieres su correlación interna)
    feature_cols = [col for col in data_df.columns if col not in NON_FEATURE_COLS]
    
    # Calcular la matriz de correlación solo entre las características
    corr_matrix = data_df[feature_cols].corr()
    
    # Visualizar la correlación
    plt.figure(figsize=(14, 12)) 
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5, annot_kws={"size": 8}) # Ajustar tamaño de anotaciones
    plt.title("Mapa de Calor de Correlación entre las Características")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Función para obtener y mostrar la importancia de las características
def plot_feature_importance(model, X_train, y_train):

    if not hasattr(model, 'feature_importances_'):
        print("El modelo proporcionado no tiene el atributo 'feature_importances_'.")
        temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
        temp_model.fit(X_train, y_train)
        importances = temp_model.feature_importances_
        print("Advertencia: Se entrenó un modelo temporal para la importancia de características.")
    else:
        importances = model.feature_importances_

    # Crear un DataFrame para almacenar las características y su importancia
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    })

    # Ordenar las características por su importancia en orden descendente
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Crear gráfico de barras
    plt.figure(figsize=(12, 7)) # Ajustar tamaño
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='magma') # Cambio a barh y paleta
    plt.title("Importancia de las Características (Random Forest Regressor)")
    plt.xlabel("Importancia")
    plt.ylabel("Características")
    plt.tight_layout()
    plt.show()