import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar funciones de los diferentes módulos

from utils.data_preprocessing import load_data,clean_data, normalize_data
from utils.visualization import plot_feature_importance, plot_correlation_heatmap, plot_feature_correlation
from models.model_backup import train_random_forest, evaluate_model
from models.model_backup import train_random_forest, grid_search_random_forest, evaluate_with_cross_validation, final_evaluation
from models.model import save_model

#1. CARGAR Y EXPLORAR LOS DATOS
data = load_data(r'C:\Users\Crishtian Paz\Desktop\Software GPTI\pump_predictor-ingrid\data\data.csv')

#2: Limpiar los datos (si es necesario)
data = clean_data(data)

#3: Normalizar los datos
data = normalize_data(data)

#4: Visualizar la correlacion de las características
plot_feature_correlation(data)

#5: Visualizar mapa de calor de correlacion
plot_correlation_heatmap(data)

#6: Visualizar la importancia de las caracterisiticas
plot_feature_importance(data)


#7: Entrenar el modelo
model, X_train, X_test, y_train, y_test = train_random_forest(data)

#8: Evaluar el modelo / 9 caracteristicas
#evaluate_model(model, X_test, y_test)


#9: Evaluar el model / Ajuste de hiperparametros
best_model = grid_search_random_forest(X_train, y_train)

    # Evaluar el modelo usando validación cruzada
evaluate_with_cross_validation(best_model, X_train, y_train)

    # Evaluar el modelo final sobre el conjunto de prueba
final_evaluation(best_model, X_test, y_test)

    # Guardar modelo entrenado
save_model(best_model)

