from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score

# Evaluacion inicial - 9 caracteristicas
def train_random_forest(data): 
    X = data[['footfall','tempMode','AQ','USS','CS','VOC','RP','IP','Temperature']]  # Características
    y = data['fail']  # Etiqueta

     # Dividir el conjunto de datos en entrenamiento y prueba (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear y entrenar el modelo RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    # Realizar las predicciones
    y_pred = model.predict(X_test)
    
    # Evaluar el rendimiento del modelo
    print("\nExactitud:", accuracy_score(y_test, y_pred))
    print("Informe de clasificacion:")
    print(classification_report(y_test, y_pred))

    # Matriz de Confusión
    conf_mat = confusion_matrix(y_test, y_pred)
    
    # Visualización de la matriz de confusión
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap="YlGnBu", xticklabels=["No Fail", "Fail"], yticklabels=["No Fail", "Fail"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()



# Evaluacion final - Ajuste de Hiperparametros

def grid_search_random_forest(X_train, y_train):
    # Definir el rango de parámetros que se quiere evaluar
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Crear el modelo
    rf = RandomForestClassifier(random_state=42)
    
    # Configurar GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
    
    # Entrenar con el Grid Search
    grid_search.fit(X_train, y_train)
    
    # Obtener el mejor modelo
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    
    return best_model
# validación cruzada
def evaluate_with_cross_validation(best_model, X_train, y_train):
    # Realizar validación cruzada
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    print("\nCross-validation results: ", cv_scores)
    print("Mean cross-validation score: ", cv_scores.mean())

def final_evaluation(best_model, X_test, y_test):
    # Predicciones finales
    y_pred = best_model.predict(X_test)
    
    # Evaluación del rendimiento
    print("Exactitud:", accuracy_score(y_test, y_pred))
    print("Informe de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusión
    conf_mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap="YlGnBu", xticklabels=["No Fail", "Fail"], yticklabels=["No Fail", "Fail"])
    plt.title("Matriz de Confusion - FINAL")
    plt.xlabel("Prediccion")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()