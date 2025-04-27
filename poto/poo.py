# Importamos las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Configuración para visualizaciones
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("viridis")

# Semilla para reproducibilidad
np.random.seed(42)

# Cargamos el dataset
data = pd.read_csv('dataset\Student_Performance.csv')

# 1. Descripción del dataset
# --------------------------

print("1. DESCRIPCIÓN DEL DATASET")
print("--------------------------")
print(f"Número de muestras: {data.shape[0]}")
print(f"Número de variables: {data.shape[1]}")
print("\nPrimeras 5 filas del dataset:")
print(data.head())

print("\nEstadísticas descriptivas básicas:")
print(data.describe())

print("\nInformación sobre tipos de datos:")
print(data.info())

print("\nVariables independientes (predictores):")
print("- Hours Studied: Horas dedicadas al estudio")
print("- Previous Scores: Calificaciones previas")
print("- Extracurricular Activities: Participación en actividades extracurriculares (categórica)")
print("- Sleep Hours: Horas de sueño")
print("- Sample Question Papers Practiced: Número de exámenes de práctica realizados")

print("\nVariable dependiente:")
print("- Performance Index: Índice de rendimiento académico")

print("\nValores únicos en Extracurricular Activities:")
print(data['Extracurricular Activities'].unique())

# Verificamos si hay valores nulos
print("\nValores nulos en el dataset:")
print(data.isnull().sum())

# 2. Relación entre características seleccionadas y variable de salida
# -------------------------------------------------------------------

print("\n\n2. RELACIÓN ENTRE CARACTERÍSTICAS SELECCIONADAS Y VARIABLE DE SALIDA")
print("-------------------------------------------------------------------")

# Seleccionamos las tres características numéricas para el análisis
features_to_analyze = ['Hours Studied', 'Previous Scores', 'Sleep Hours']
selected_features = features_to_analyze + ['Performance Index']

# Creamos scatter matrix para visualizar relaciones
print("\nCreando scatter matrix para visualizar relaciones entre variables...")
sns.set(style="ticks")
scatter_matrix = pd.plotting.scatter_matrix(data[selected_features], figsize=(15, 15), 
                                           diagonal='kde', alpha=0.8)
plt.savefig('scatter_matrix.png')
plt.close()

# Visualizamos la relación de cada característica con la variable dependiente
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, feature in enumerate(features_to_analyze):
    sns.regplot(x=feature, y='Performance Index', data=data, ax=axes[i])
    axes[i].set_title(f'Relación entre {feature} y Performance Index')
plt.tight_layout()
plt.savefig('feature_relationships.png')
plt.close()

# Analizamos correlaciones
correlation_matrix = data[selected_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación')
plt.savefig('correlation_matrix.png')
plt.close()

print("\nMatriz de correlación:")
print(correlation_matrix)

print("\nInterpretación de las relaciones observadas:")
print("- Hours Studied vs. Performance Index: Muestra la relación entre el tiempo de estudio y el rendimiento.")
print("- Previous Scores vs. Performance Index: Indica cómo las calificaciones anteriores predicen el rendimiento actual.")
print("- Sleep Hours vs. Performance Index: Refleja el impacto de las horas de sueño en el rendimiento académico.")

# Visualizamos la distribución de la variable dependiente
plt.figure(figsize=(10, 6))
sns.histplot(data['Performance Index'], kde=True)
plt.title('Distribución de Performance Index')
plt.savefig('performance_distribution.png')
plt.close()

# Analizamos la relación de la variable categórica con la variable dependiente
plt.figure(figsize=(10, 6))
sns.boxplot(x='Extracurricular Activities', y='Performance Index', data=data)
plt.title('Performance Index por Tipo de Actividad Extracurricular')
plt.xticks(rotation=45)
plt.savefig('extracurricular_boxplot.png')
plt.close()

# 3. Ajuste de modelo de regresión lineal
# ---------------------------------------

print("\n\n3. AJUSTE DE MODELO DE REGRESIÓN LINEAL")
print("---------------------------------------")

# Preparación de datos
X = data.drop('Performance Index', axis=1)
y = data['Performance Index']

# Preprocesamiento para manejar la variable categórica
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['Extracurricular Activities'])
    ],
    remainder='passthrough'
)

# 3.1 Entrenamiento con diferentes proporciones train/test
# --------------------------------------------------------

print("\n3.1 ENTRENAMIENTO CON DIFERENTES PROPORCIONES TRAIN/TEST")
print("--------------------------------------------------------")

# Definimos las proporciones a probar
split_ratios = [(0.7, 0.3), (0.5, 0.5), (0.4, 0.6)]
results_splits = []

for train_size, test_size in split_ratios:
    # Dividimos los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Creamos el pipeline con preprocesamiento y modelo
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Entrenamos el modelo
    pipeline.fit(X_train, y_train)
    
    # Evaluamos el modelo
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Guardamos resultados
    results_splits.append({
        'Train-Test Split': f"{train_size:.1f}-{test_size:.1f}",
        'MSE': mse,
        'R²': r2
    })

# Mostramos resultados
splits_df = pd.DataFrame(results_splits)
print("\nResultados con diferentes proporciones de train/test:")
print(splits_df)

print("\nInterpretación:")
print("Observamos cómo diferentes proporciones de train/test afectan el desempeño del modelo.")
print("Una proporción adecuada equilibra la cantidad de datos disponibles para entrenar el modelo")
print("y la cantidad necesaria para evaluarlo de manera confiable.")

# 3.2 Cambio de método de optimización
# -----------------------------------

print("\n\n3.2 CAMBIO DE MÉTODO DE OPTIMIZACIÓN")
print("-----------------------------------")

# Usamos la mejor proporción del experimento anterior (asumamos 70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definimos los optimizadores a probar
optimizers = {
    'Linear Regression (Normal Equation)': LinearRegression(),
    'SGD (alpha=0.01, max_iter=1000)': SGDRegressor(alpha=0.01, max_iter=1000, random_state=42),
    'SGD (alpha=0.001, max_iter=1000)': SGDRegressor(alpha=0.001, max_iter=1000, random_state=42),
    'SGD (alpha=0.0001, max_iter=1000)': SGDRegressor(alpha=0.0001, max_iter=1000, random_state=42)
}

results_optimizers = []

for name, optimizer in optimizers.items():
    # Creamos el pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', optimizer)
    ])
    
    # Entrenamos el modelo
    pipeline.fit(X_train, y_train)
    
    # Evaluamos el modelo
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Guardamos resultados
    results_optimizers.append({
        'Optimizer': name,
        'MSE': mse,
        'R²': r2
    })

# Mostramos resultados
optimizers_df = pd.DataFrame(results_optimizers)
print("\nResultados con diferentes métodos de optimización:")
print(optimizers_df)

print("\nInterpretación:")
print("Comparamos la regresión lineal estándar (que usa la ecuación normal) con")
print("el Descenso de Gradiente Estocástico (SGD) con diferentes tasas de aprendizaje.")
print("Esto nos permite observar cómo diferentes enfoques de optimización afectan la convergencia")
print("y la calidad del modelo final.")

# 3.3 Métodos de regularización
# ----------------------------

print("\n\n3.3 MÉTODOS DE REGULARIZACIÓN")
print("----------------------------")

# Definimos los métodos de regularización a probar
regularizers = {
    'No Regularization': LinearRegression(),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0, random_state=42),
    'Ridge (alpha=0.1)': Ridge(alpha=0.1, random_state=42),
    'Lasso (alpha=1.0)': Lasso(alpha=1.0, random_state=42),
    'Lasso (alpha=0.1)': Lasso(alpha=0.1, random_state=42),
    'ElasticNet (alpha=1.0, l1_ratio=0.5)': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
    'ElasticNet (alpha=0.1, l1_ratio=0.5)': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
}

results_regularizers = []

for name, regularizer in regularizers.items():
    # Creamos el pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regularizer)
    ])
    
    # Entrenamos el modelo
    pipeline.fit(X_train, y_train)
    
    # Evaluamos el modelo
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Guardamos resultados
    results_regularizers.append({
        'Regularizer': name,
        'MSE': mse,
        'R²': r2
    })

# Mostramos resultados
regularizers_df = pd.DataFrame(results_regularizers)
print("\nResultados con diferentes métodos de regularización:")
print(regularizers_df)

print("\nInterpretación:")
print("La regularización ayuda a prevenir el sobreajuste penalizando la magnitud de los coeficientes.")
print("Ridge (L2) penaliza el cuadrado de los coeficientes, reduciendo su magnitud pero sin llevarlos a cero.")
print("Lasso (L1) puede llevar coeficientes a cero, actuando como selector de características.")
print("ElasticNet combina ambos enfoques, ofreciendo un equilibrio entre reducción y selección.")

# 3.4 Modelo final y análisis de parámetros
# ----------------------------------------

print("\n\n3.4 MODELO FINAL Y ANÁLISIS DE PARÁMETROS")
print("----------------------------------------")

# Basado en los resultados anteriores, seleccionamos el mejor modelo
# Por ejemplo, supongamos que Ridge con alpha=0.1 fue el mejor
best_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=0.1, random_state=42))
])

# Entrenamos el modelo final con la mejor proporción (70-30)
best_model.fit(X_train, y_train)

# Obtenemos los coeficientes
feature_names = list(best_model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(['Extracurricular Activities']))
feature_names.extend(['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced'])

coefficients = best_model.named_steps['regressor'].coef_
intercept = best_model.named_steps['regressor'].intercept_

# Creamos un DataFrame para visualizar los coeficientes
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})
coef_df = coef_df.sort_values('Coefficient', ascending=False)

print("\nIntercepción (valor base):", intercept)
print("\nCoeficientes del modelo:")
print(coef_df)

# Visualizamos los coeficientes
plt.figure(figsize=(12, 8))
sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.title('Coeficientes del Modelo Final')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.tight_layout()
plt.savefig('model_coefficients.png')
plt.close()

print("\nInterpretación de los coeficientes:")
print("- Magnitud: Indica la importancia relativa de cada característica.")
print("  Un coeficiente mayor (en valor absoluto) tiene un mayor impacto en la predicción.")
print("- Signo: Indica la dirección de la relación con la variable dependiente.")
print("  - Positivo: Al aumentar la característica, aumenta el Performance Index.")
print("  - Negativo: Al aumentar la característica, disminuye el Performance Index.")

# 3.5 Métricas de evaluación final
# -------------------------------

print("\n\n3.5 MÉTRICAS DE EVALUACIÓN FINAL")
print("-------------------------------")

# Evaluamos el modelo final en el conjunto de prueba
y_pred_final = best_model.predict(X_test)
final_mse = mean_squared_error(y_test, y_pred_final)
final_r2 = r2_score(y_test, y_pred_final)

print(f"\nMean Squared Error (MSE): {final_mse:.4f}")
print(f"R² Score: {final_r2:.4f}")

# Visualizamos predicciones vs valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_final, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')
plt.savefig('predictions_vs_actual.png')
plt.close()

# Visualizamos residuos
residuals = y_test - y_pred_final
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_final, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos')
plt.savefig('residuals_plot.png')
plt.close()

print("\nInterpretación de las métricas:")
print("- MSE (Mean Squared Error): Mide el promedio de los errores al cuadrado.")
print("  Un valor más bajo indica un mejor ajuste del modelo.")
print("- R² Score: Indica la proporción de la varianza en la variable dependiente")
print("  que es predecible a partir de las variables independientes.")
print("  - R² = 1: Ajuste perfecto")
print("  - R² = 0: El modelo no explica nada de la variabilidad")
print("  - R² < 0: El modelo es peor que predecir la media")

# 4. Conclusiones
# --------------

print("\n\n4. CONCLUSIONES")
print("--------------")

print("\nA partir del análisis realizado, podemos concluir que:")
print("1. Las variables más influyentes en el rendimiento académico son...")
print("2. La proporción óptima de train/test es...")
print("3. El método de optimización más efectivo fue...")
print("4. La regularización tuvo el siguiente impacto...")
print("5. El modelo final logra explicar aproximadamente un {:.1f}% de la variabilidad en el rendimiento académico.".format(final_r2 * 100))