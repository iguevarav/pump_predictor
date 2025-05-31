from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor # CAMBIO IMPORTANTE: Regressor
from sklearn.metrics import mean_squared_error, r2_score # Nuevas métricas para regresión
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np

# Función para obtener las características del modelo
def get_features_for_model(df):
    feature_cols = [col for col in df.columns if col not in ['unit_id', 'cycle', 'RUL']]
    return feature_cols


def train_random_forest(data_df): 
    feature_cols = get_features_for_model(data_df)
    X = data_df[feature_cols]
    y = data_df['RUL']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X, y


def grid_search_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestRegressor(random_state=42)
    
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv=5, n_jobs=-1,scoring = 'neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("Mejores parámetros encontrados: ", grid_search.best_params_)
    return best_model

def evaluate_with_cross_validation(best_model, X_train, y_train):
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    print("RMSE de validación cruzada:", rmse_scores)
    print("RMSE promedio de validación cruzada:", np.mean(rmse_scores))

    cv_r2_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
    print("R2 de validación cruzada:", cv_r2_scores)
    print("R2 promedio de validación cruzada:", np.mean(cv_r2_scores))


def final_evaluation(best_model, test_df_processed, true_rul_df):
    feature_cols = get_features_for_model(test_df_processed)
    X_test = test_df_processed[feature_cols]
    y_pred = best_model.predict(X_test)
    y_true = true_rul_df['RUL'].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"RMSE Final en conjunto de prueba: {rmse:.4f}")
    print(f"R2 SCORE FINAL en el conjunto de prueba: {r2:.4f}")

    #Calcular el score de la NADA 
    def calculate_nasa_score(y_true, y_pred):
        d = y_pred - y_true
        s = np.zeros(len(d))
        for i, val in enumerate(d):
            if val < 0: # Predicción más baja que la real (falla antes de lo predicho)
                s[i] = np.exp(-val / 13) - 1 # alpha = 13 (parámetro común para C-MAPSS)
            else: # Predicción más alta que la real (falla después de lo predicho)
                s[i] = np.exp(val / 10) - 1 # beta = 10 (parámetro común para C-MAPSS)
        return np.sum(s)

    nasa_score = calculate_nasa_score(y_true, y_pred)
    print(f"NASA Score (S_k) Final en Test Set: {nasa_score:.2f}")


    # Visualización de predichos vs reales (puede ser un scatter plot)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2) # Línea de referencia ideal
    plt.title("RUL Predicha vs. RUL Real (Test Set)")
    plt.xlabel("RUL Real")
    plt.ylabel("RUL Predicha")
    plt.grid(True)
    plt.show()

    # Visualización de errores
    errors = y_pred - y_true
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title("Distribución de Errores de Predicción de RUL")
    plt.xlabel("Error (Predicción - Real)")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.show()


# ✅ NUEVA FUNCIÓN - debe estar aquí y no dentro de otra
def save_model(model, path="models/modelo1.pkl"): # Nombre de archivo actualizado
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Modelo guardado en: {path}")

