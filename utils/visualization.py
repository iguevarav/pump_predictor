import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Función para mostrar la correlacion de las características
def plot_feature_correlation(data):
    # Excluir la columna 'fail' para calcular la correlación solo entre las características
    correlacion_matrix = data.drop(columns=['fail']).corr()  # Excluir 'fail'
    
    # Calcular la correlación de las características con 'fail'
    fail_correlacion = data.corr()['fail'].drop('fail').sort_values(ascending=False)  # Excluye 'fail' de las correlaciones

    # Visualización
    plt.figure(figsize=(10, 6))
    sns.barplot(x=fail_correlacion.index, y=fail_correlacion.values)
    plt.title('Correlación de las características vs Estado de la máquina')
    plt.xlabel('Características')
    plt.ylabel('Correlación con Fail')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Función para mostrar el mapa de calor de correlación entre las características
def plot_correlation_heatmap(data):
    # Excluir la columna 'fail' para que no se incluya en la matriz de correlación
    corr_matrix = data.drop(columns=['fail']).corr()  # Excluir 'fail' de las características
    
    # Visualizar la correlación
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
    plt.title("Mapa de calor de correlación entre las características")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#Funcion para obtener la importancia de las caracteristicas
def plot_feature_importance(data):
    X = data.drop('fail', axis=1)  # Excluimos la columna 'fail' como objetivo
    y = data['fail']  # Variable objetivo

    # Dividimos el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Obtener la importancia de las características
    importances = model.feature_importances_

    # Crear un DataFrame para almacenar las características y su importancia
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    })

    # Ordenar las características por su importancia en orden descendente
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Crear gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.title("Importancia de las Características")
    plt.xlabel("Importancia")
    plt.ylabel("Características")
    plt.show()
