import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Configuración de la página
st.set_page_config(
    page_title="Laboratorio 2: Algoritmos de Optimización en ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
<style>
    .title {
        text-align: center;
        font-weight: bold;
        color: #3366ff;
    }
    .subtitle {
        font-weight: bold;
        color: #0099cc;
    }
    .centered {
        display: flex;
        justify-content: center;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .caption {
        font-size: 0.85em;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("<h1 class='title'>Laboratorio 2: Algoritmos de Optimización en Machine Learning</h1>", unsafe_allow_html=True)
st.markdown("### Análisis del rendimiento académico de estudiantes")

# Función para cargar datos
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('Student_Performance.csv')
        return data
    except FileNotFoundError:
        st.error("No se encontró el archivo 'Student_Performance.csv'. Por favor, asegúrese de que el archivo esté en el mismo directorio que esta aplicación.")
        return None

# Barra lateral para navegación
st.sidebar.title("Navegación")
pages = st.sidebar.radio("Ir a:", [
    "1. Descripción del Dataset",
    "2. Análisis Exploratorio",
    "3. Visualización 3D",
    "4. Modelado y Optimización",
    "5. Métricas y Evaluación",
    "6. Conclusiones"
])

# Cargar el dataset
data = load_data()

if data is not None:
    # 1. DESCRIPCIÓN DEL DATASET
    if pages == "1. Descripción del Dataset":
        st.markdown("<h2 class='subtitle'>1. Descripción del Dataset</h2>", unsafe_allow_html=True)
        
        # Información general
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Número de muestras", data.shape[0])
        with col2:
            st.metric("Número de variables", data.shape[1])
        
        # Primeras filas
        st.subheader("Primeras filas del dataset")
        st.dataframe(data.head())
        
        # Estadísticas descriptivas
        st.subheader("Estadísticas descriptivas")
        st.dataframe(data.describe())
        
        
        def formato_tipo_dato(tipo):
            """Convierte tipos de datos técnicos a términos más comprensibles en español"""
            if 'int' in str(tipo):
                return "entero"
            elif 'float' in str(tipo):
                return "decimal"
            elif 'object' in str(tipo):
                return "categórico"
            elif 'bool' in str(tipo):
                return "booleano"
            elif 'datetime' in str(tipo):
                return "fecha/hora"
            else:
                return str(tipo)
        # Información sobre los tipos de datos
        st.subheader("Información sobre tipos de datos")
        
        # Crea una tabla personalizada para mostrar la información
        info_datos = []
        for col in data.columns:
            no_nulos = data[col].count() # Cuenta valores no-nulo
            tipo = formato_tipo_dato(data[col].dtype) # Obtiene el tipo de dato y lo formatea
            porcentaje = 100 * (no_nulos / len(data)) # Porcentaje de completitud
            
            info_datos.append({
                "Columna": col,
                "Tipo de Dato": tipo,
                "Valores No-Nulos": f"{no_nulos} de {len(data)} ({porcentaje:.1f}%)"
            })

        # Convierte a DataFrame para mostrar como tabla
        info_df = pd.DataFrame(info_datos)
        st.table(info_df)

        # Información adicional del dataset
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Número de filas:** {data.shape[0]}")
            st.write(f"**Número de columnas:** {data.shape[1]}")

        with col2:
            memo_kb = data.memory_usage(deep=True).sum() / 1024
            st.write(f"**Uso de memoria:** {memo_kb:.1f} KB")
        
        # Valores únicos en variables categóricas
        st.subheader("Valores únicos en Extracurricular Activities")
        st.write(data['Extracurricular Activities'].unique())
        
        # Valores nulos
        st.subheader("Valores nulos en el dataset")
        st.write(data.isnull().sum())
        
        # Descripción de las variables
        st.subheader("Descripción de las variables")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4>Variables independientes (predictores)</h4>", unsafe_allow_html=True)
            st.markdown("- **Hours Studied**: Horas dedicadas al estudio")
            st.markdown("- **Previous Scores**: Calificaciones previas")
            st.markdown("- **Extracurricular Activities**: Participación en actividades extracurriculares (categórica)")
            st.markdown("- **Sleep Hours**: Horas de sueño")
            st.markdown("- **Sample Question Papers Practiced**: Número de exámenes de práctica realizados")
        
        with col2:
            st.markdown("<h4>Variable dependiente</h4>", unsafe_allow_html=True)
            st.markdown("- **Performance Index**: Índice de rendimiento académico")
    
    # 2. ANÁLISIS EXPLORATORIO
    elif pages == "2. Análisis Exploratorio":
        st.markdown("<h2 class='subtitle'>2. Análisis Exploratorio de Datos</h2>", unsafe_allow_html=True)
        
        # Selección de características para visualización
        st.subheader("Visualización de relaciones entre variables")
        
        features_options = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']
        selected_features = st.multiselect(
            "Seleccione características para analizar (máximo 3):",
            features_options,
            default=['Hours Studied', 'Previous Scores', 'Sleep Hours'],
            max_selections=3
        )
        
        if len(selected_features) > 0:
            # Matriz de correlación
            selected_features_with_target = selected_features + ['Performance Index']
            correlation_matrix = data[selected_features_with_target].corr()
            
            st.subheader("Matriz de Correlación")
            fig_corr = plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Matriz de Correlación')
            st.pyplot(fig_corr)
            
            # Interpretación de correlaciones
            st.subheader("Interpretación de correlaciones")
            for feature in selected_features:
                corr_value = correlation_matrix.loc[feature, 'Performance Index']
                correlacion = ""

                if abs(corr_value) > 0.7:
                    correlacion = "**Correlación fuerte**"
                elif abs(corr_value) > 0.3:
                    correlacion = "**Correlación moderada**"
                else:
                    correlacion = "**Correlación débil**"

                st.markdown(f"- **{feature} vs. Performance Index**: {corr_value:.4f} ({correlacion})")
            
            # Scatter plots individuales
            st.subheader("Relaciones entre variables seleccionadas y Performance Index")
            
            col1, col2 = st.columns(2)
            
            for i, feature in enumerate(selected_features):
                fig = plt.figure(figsize=(10, 6))
                sns.regplot(x=feature, y='Performance Index', data=data)
                plt.title(f'Relación entre {feature} y Performance Index')
                plt.xlabel(feature)
                plt.ylabel('Performance Index')
                plt.tight_layout()
                
                if i % 2 == 0:
                    with col1:
                        st.pyplot(fig)
                else:
                    with col2:
                        st.pyplot(fig)
            
            # Distribución de la variable objetivo
            st.subheader("Distribución de Performance Index")
            fig_dist = plt.figure(figsize=(10, 6))
            sns.histplot(data['Performance Index'], kde=True)
            plt.title('Distribución de Performance Index')
            st.pyplot(fig_dist)
            
            # Boxplot para variable categórica
            st.subheader("Performance Index por Tipo de Actividad Extracurricular")
            fig_box = plt.figure(figsize=(10, 6))
            sns.boxplot(x='Extracurricular Activities', y='Performance Index', data=data)
            plt.title('Performance Index por Tipo de Actividad Extracurricular')
            st.pyplot(fig_box)
        else:
            st.warning("Por favor, seleccione al menos una característica para visualizar.")
    
    # 3. VISUALIZACIÓN 3D
    elif pages == "3. Visualización 3D":
        st.markdown("<h2 class='subtitle'>3. Visualización 3D de Relaciones</h2>", unsafe_allow_html=True)
        
        st.write("Explore cómo dos predictores se relacionan con el rendimiento académico en un espacio tridimensional.")
        
        # Selección de características para visualización 3D
        col1, col2 = st.columns(2)
        
        with col1:
            feature_x = st.selectbox(
                "Seleccione la primera característica (eje X):",
                ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced'],
                index=1  # Previous Scores por defecto
            )
        
        with col2:
            # Excluimos la característica ya seleccionada
            remaining_features = [f for f in ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced'] if f != feature_x]
            feature_y = st.selectbox(
                "Seleccione la segunda característica (eje Y):",
                remaining_features,
                index=0  # Primera opción disponible
            )
        
        # Función para crear visualización 3D con Plotly
        @st.cache_data
        def create_3d_visualization(data, x1_name, x2_name, y_name):
            # Extraemos los datos
            x1 = data[x1_name]
            x2 = data[x2_name]
            y = data[y_name]
            
            # Ajustamos un modelo de regresión para obtener el plano
            X = data[[x1_name, x2_name]]
            model = LinearRegression()
            model.fit(X, y)
            
            # Creamos una malla para el plano de regresión
            x1_range = np.linspace(x1.min(), x1.max(), 50)
            x2_range = np.linspace(x2.min(), x2.max(), 50)
            x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
            
            # Predicciones para la malla
            X_mesh = np.column_stack((x1_mesh.flatten(), x2_mesh.flatten()))
            y_pred = model.predict(X_mesh).reshape(x1_mesh.shape)
            
            # Ecuación del plano
            intercept = model.intercept_
            coef1 = model.coef_[0]
            coef2 = model.coef_[1]
            equation = f'{y_name} = {coef1:.2f}*{x1_name} + {coef2:.2f}*{x2_name} + {intercept:.2f}'
            r2 = model.score(X, y)
            
            fig = go.Figure() # Creamos la figura de Plotly
            
            # Añadimos los puntos de datos
            fig.add_trace(
                go.Scatter3d(
                    x=x1,
                    y=x2,
                    z=y,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=y,
                        colorscale='Viridis',
                        opacity=0.7,
                        colorbar=dict(title=y_name)
                    ),
                    name='Datos'
                )
            )
            
            # Añadimos el plano de regresión
            fig.add_trace(
                go.Surface(
                    x=x1_mesh,
                    y=x2_mesh,
                    z=y_pred,
                    colorscale='RdBu',
                    opacity=0.7,
                    name='Plano de Regresión',
                    showscale=False
                )
            )
            
            # Configuramos el layout
            fig.update_layout(
                title=f'Relación 3D: {x1_name} y {x2_name} vs {y_name}',
                scene=dict(
                    xaxis_title=x1_name,
                    yaxis_title=x2_name,
                    zaxis_title=y_name,
                    aspectratio=dict(x=1, y=1, z=0.7)
                ),
                width=800,
                height=700,
                margin=dict(l=0, r=0, b=0, t=40)
            )
            
            # Añadimos anotación con la ecuación
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"Ecuación: {equation}<br>R² = {r2:.4f}",
                showarrow=False,
                font=dict(size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                align="left"
            )
            
            return fig
        
        # Creamos y mostramos la visualización 3D
        fig_3d = create_3d_visualization(data, feature_x, feature_y, 'Performance Index')
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Visualización por categoría
        st.subheader("Visualización 3D por Actividad Extracurricular")
        
        show_categorical = st.checkbox("Mostrar análisis por actividad extracurricular", value=False)
        
        if show_categorical:
            # Función para crear visualización 3D por categoría
            @st.cache_data
            def create_3d_categorical(data, x1_name, x2_name, y_name, cat_name):
                fig = go.Figure()
                
                # Obtener categorías únicas
                categories = data[cat_name].unique()
                colors = ['red', 'blue']  # Para Yes/No
                
                equations = []
                
                for i, category in enumerate(categories):
                    # Filtrar datos para esta categoría
                    cat_data = data[data[cat_name] == category]
                    
                    # Extraer datos
                    x1 = cat_data[x1_name]
                    x2 = cat_data[x2_name]
                    y = cat_data[y_name]
                    
                    # Ajustar modelo
                    X = cat_data[[x1_name, x2_name]]
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Crear malla
                    x1_range = np.linspace(x1.min(), x1.max(), 30)
                    x2_range = np.linspace(x2.min(), x2.max(), 30)
                    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
                    
                    # Predicciones
                    X_mesh = np.column_stack((x1_mesh.flatten(), x2_mesh.flatten()))
                    y_pred = model.predict(X_mesh).reshape(x1_mesh.shape)
                    
                    # Ecuación
                    intercept = model.intercept_
                    coef1 = model.coef_[0]
                    coef2 = model.coef_[1]
                    r2 = model.score(X, y)
                    equation = f"{cat_name}={category}: {y_name} = {coef1:.2f}*{x1_name} + {coef2:.2f}*{x2_name} + {intercept:.2f} (R²={r2:.4f})"
                    equations.append(equation)
                    
                    # Añadir puntos
                    fig.add_trace(
                        go.Scatter3d(
                            x=x1,
                            y=x2,
                            z=y,
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=colors[i],
                                opacity=0.7
                            ),
                            name=f'{cat_name}={category}'
                        )
                    )
                    
                    # Añadir plano
                    fig.add_trace(
                        go.Surface(
                            x=x1_mesh,
                            y=x2_mesh,
                            z=y_pred,
                            colorscale=[[0, colors[i]], [1, colors[i]]],
                            opacity=0.4,
                            showscale=False,
                            name=f'Plano ({category})'
                        )
                    )
                
                # Configurar layout
                fig.update_layout(
                    title=f'Efecto de {cat_name} en la relación entre {x1_name}, {x2_name} y {y_name}',
                    scene=dict(
                        xaxis_title=x1_name,
                        yaxis_title=x2_name,
                        zaxis_title=y_name,
                        aspectratio=dict(x=1, y=1, z=0.7)
                    ),
                    width=800,
                    height=700,
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                
                # Añadir ecuaciones
                for i, eq in enumerate(equations):
                    fig.add_annotation(
                        x=0.02,
                        y=0.98 - i*0.08,
                        xref="paper",
                        yref="paper",
                        text=eq,
                        showarrow=False,
                        font=dict(size=12, color=colors[i]),
                        bgcolor="white",
                        bordercolor=colors[i],
                        borderwidth=1,
                        borderpad=4,
                        align="left"
                    )
                
                return fig
            
            fig_cat = create_3d_categorical(data, feature_x, feature_y, 'Performance Index', 'Extracurricular Activities')
            st.plotly_chart(fig_cat, use_container_width=True)
            
            st.markdown("""
            <div class='caption'>
            Esta visualización muestra cómo la participación en actividades extracurriculares afecta la relación entre las variables seleccionadas y el rendimiento académico.
            Los diferentes colores representan las categorías (Sí/No) y cada plano muestra la relación lineal para esa categoría específica.
            </div>
            """, unsafe_allow_html=True)
    
    # 4. MODELADO Y OPTIMIZACIÓN
    elif pages == "4. Modelado y Optimización":
        st.markdown("<h2 class='subtitle'>4. Modelado y Optimización</h2>", unsafe_allow_html=True)
        
        # Dividimos la página en pestañas para los diferentes experimentos
        tabs = st.tabs([
            "Proporciones Train/Test", 
            "Métodos de Optimización", 
            "Métodos de Regularización"
        ])
        
        # Preparación de datos (común para todos los experimentos)
        X = data.drop('Performance Index', axis=1)
        y = data['Performance Index']
        
        # Preprocesamiento para la variable categórica
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first'), ['Extracurricular Activities'])
            ],
            remainder='passthrough'
        )
        
        # Pestaña 1: Proporciones Train/Test
        with tabs[0]:
            st.subheader("Experimentación con Diferentes Proporciones Train/Test")
            
            st.write("""
            En esta sección, evaluamos cómo diferentes proporciones de datos de entrenamiento y prueba
            afectan el desempeño del modelo de regresión lineal.
            """)
            
            # Permitir al usuario seleccionar proporciones
            st.markdown("### Seleccione proporciones a evaluar")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                split1 = st.slider("Proporción 1 (% Train)", 
                                   min_value=10, max_value=90, value=70, step=5)
            with col2:
                split2 = st.slider("Proporción 2 (% Train)", 
                                   min_value=10, max_value=90, value=50, step=5)
            with col3:
                split3 = st.slider("Proporción 3 (% Train)", 
                                   min_value=10, max_value=90, value=40, step=5)
            
            # Botón para ejecutar el experimento
            if st.button("Ejecutar Experimento de Proporciones"):
                # Convertimos los porcentajes a proporciones
                splits = [(split1/100, 1-split1/100), 
                          (split2/100, 1-split2/100), 
                          (split3/100, 1-split3/100)]
                
                results_splits = []
                
                with st.spinner("Ejecutando experimento..."):
                    for train_size, test_size in splits:
                        # División de datos
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                        
                        # Pipeline
                        pipeline = Pipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', LinearRegression())
                        ])
                        
                        # Entrenamiento
                        pipeline.fit(X_train, y_train)
                        
                        # Evaluación
                        y_pred = pipeline.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        # Resultados
                        results_splits.append({
                            'Train-Test Split': f"{train_size:.2f}-{test_size:.2f}",
                            'Muestras Train': int(train_size * len(X)),
                            'Muestras Test': int(test_size * len(X)),
                            'MSE': mse,
                            'R²': r2
                        })
                
                # Mostrar resultados en tabla
                splits_df = pd.DataFrame(results_splits)
                st.subheader("Resultados")
                st.dataframe(splits_df)
                
                # Visualizar resultados
                fig, ax = plt.subplots(1, 2, figsize=(14, 5))
                
                # Gráfico de MSE
                sns.barplot(x='Train-Test Split', y='MSE', data=splits_df, ax=ax[0])
                ax[0].set_title('MSE por Proporción Train-Test')
                ax[0].set_ylabel('Mean Squared Error')
                ax[0].set_ylim(bottom=0)
                
                # Gráfico de R²
                sns.barplot(x='Train-Test Split', y='R²', data=splits_df, ax=ax[1])
                ax[1].set_title('R² por Proporción Train-Test')
                ax[1].set_ylabel('R² Score')
                ax[1].set_ylim(0, 1)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Interpretación
                st.subheader("Interpretación")
                
                best_split = splits_df.iloc[splits_df['R²'].argmax()]
                
                st.markdown(f"""
                - La mejor proporción de train/test según R² es **{best_split['Train-Test Split']}** con R²={best_split['R²']:.4f}
                - El MSE más bajo se obtuvo con la proporción **{splits_df.iloc[splits_df['MSE'].argmin()]['Train-Test Split']}** (MSE={splits_df['MSE'].min():.4f})
                - Una proporción adecuada equilibra la cantidad de datos disponibles para entrenar el modelo y la cantidad necesaria para evaluarlo de manera confiable.
                """)
                
                # Guardar el mejor valor para usar en otras pestañas
                st.session_state['best_split'] = float(best_split['Train-Test Split'].split('-')[0])
        
        # Pestaña 2: Métodos de Optimización
        with tabs[1]:
            st.subheader("Experimentación con Diferentes Métodos de Optimización")
            
            st.write("""
            En esta sección, evaluamos cómo diferentes métodos de optimización afectan el desempeño
            del modelo de regresión lineal. Comparamos la ecuación normal (método estándar) con
            diferentes configuraciones del Descenso de Gradiente Estocástico (SGD).
            """)
            
            # Usar la mejor proporción si está disponible, sino usar 0.7 (70%)
            best_split = st.session_state.get('best_split', 0.7)
            
            # Permitir ajustar parámetros de SGD
            st.markdown("### Configuración de SGD")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                alpha1 = st.number_input("Alpha 1", 
                                         min_value=0.00001, max_value=1.0, 
                                         value=0.01, format="%.5f", step=0.001)
            with col2:
                alpha2 = st.number_input("Alpha 2", 
                                         min_value=0.00001, max_value=1.0, 
                                         value=0.001, format="%.5f", step=0.0001)
            with col3:
                alpha3 = st.number_input("Alpha 3", 
                                         min_value=0.00001, max_value=1.0, 
                                         value=0.0001, format="%.5f", step=0.00001)
            
            max_iter = st.slider("Iteraciones máximas", 
                                 min_value=100, max_value=10000, value=1000, step=100)
            
            # Botón para ejecutar el experimento
            if st.button("Ejecutar Experimento de Optimización"):
                # División de datos
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=(1-best_split), random_state=42
                )
                
                # Definir optimizadores
                optimizers = {
                    'Linear Regression (Normal Equation)': LinearRegression(),
                    f'SGD (alpha={alpha1}, max_iter={max_iter})': SGDRegressor(
                        alpha=alpha1, max_iter=max_iter, random_state=42, learning_rate='constant', eta0=0.01),
                    f'SGD (alpha={alpha2}, max_iter={max_iter})': SGDRegressor(
                        alpha=alpha2, max_iter=max_iter, random_state=42, learning_rate='constant', eta0=0.01),
                    f'SGD (alpha={alpha3}, max_iter={max_iter})': SGDRegressor(
                        alpha=alpha3, max_iter=max_iter, random_state=42, learning_rate='constant', eta0=0.01)
                }
                
                results_optimizers = []
                
                with st.spinner("Ejecutando experimento..."):
                    for name, optimizer in optimizers.items():
                        try:
                            # Pipeline
                            pipeline = Pipeline([
                                ('preprocessor', preprocessor),
                                ('regressor', optimizer)
                            ])
                            
                            # Entrenamiento
                            pipeline.fit(X_train, y_train)
                            
                            # Evaluación
                            y_pred = pipeline.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            # Resultados
                            results_optimizers.append({
                                'Optimizer': name,
                                'MSE': mse,
                                'R²': r2
                            })
                        except Exception as e:
                            results_optimizers.append({
                                'Optimizer': name,
                                'MSE': 'Error',
                                'R²': 'Error',
                                'Error': str(e)
                            })
                
                # Mostrar resultados en tabla
                optimizers_df = pd.DataFrame(results_optimizers)
                st.subheader("Resultados")
                st.dataframe(optimizers_df)
                
                # Filtrar resultados con errores
                valid_results = optimizers_df[optimizers_df['MSE'] != 'Error'].copy()
                
                if not valid_results.empty:
                    # Convertir a numérico para visualización
                    valid_results['MSE'] = pd.to_numeric(valid_results['MSE'])
                    valid_results['R²'] = pd.to_numeric(valid_results['R²'])
                    
                    # Visualizar resultados
                    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Gráfico de MSE
                    sns.barplot(x='Optimizer', y='MSE', data=valid_results, ax=ax[0])
                    ax[0].set_title('MSE por Método de Optimización')
                    ax[0].set_ylabel('Mean Squared Error')
                    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')
                    ax[0].set_ylim(bottom=0)
                    
                    # Gráfico de R²
                    sns.barplot(x='Optimizer', y='R²', data=valid_results, ax=ax[1])
                    ax[1].set_title('R² por Método de Optimización')
                    ax[1].set_ylabel('R² Score')
                    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')
                    ax[1].set_ylim(0, 1)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Interpretación
                st.subheader("Interpretación")
                
                if 'Error' in optimizers_df['MSE'].values:
                    st.warning("Algunos métodos de optimización generaron errores. Intente con diferentes valores de alpha o iteraciones.")
                
                if not valid_results.empty:
                    best_opt = valid_results.iloc[valid_results['R²'].argmax()]
                    
                    st.markdown(f"""
                    - El mejor método de optimización según R² es **{best_opt['Optimizer']}** con R²={best_opt['R²']:.4f}
                    - El MSE más bajo se obtuvo con **{valid_results.iloc[valid_results['MSE'].argmin()]['Optimizer']}** (MSE={valid_results['MSE'].min():.4f})
                    - La regresión lineal estándar (ecuación normal) y el SGD pueden tener diferentes comportamientos dependiendo de los parámetros y la naturaleza de los datos.
                    """)
                    
                    # Guardar el mejor optimizador para usar en otras pestañas
                    st.session_state['best_optimizer'] = best_opt['Optimizer']
                else:
                    st.error("No se obtuvieron resultados válidos. Intente con diferentes parámetros.")
        
        # Pestaña 3: Métodos de Regularización
        with tabs[2]:
            st.subheader("Experimentación con Diferentes Métodos de Regularización")
            
            st.write("""
            En esta sección, evaluamos cómo diferentes métodos de regularización afectan el desempeño
            del modelo de regresión lineal. Comparamos el modelo sin regularización con Ridge (L2),
            Lasso (L1) y ElasticNet (combinación de L1 y L2).
            """)
            
            # Usar la mejor proporción si está disponible, sino usar 0.7 (70%)
            best_split = st.session_state.get('best_split', 0.7)
            
            # Permitir ajustar parámetros de regularización
            st.markdown("### Configuración de Regularización")
            
            col1, col2 = st.columns(2)
            
            with col1:
                alpha_high = st.number_input("Alpha (alto)", 
                                            min_value=0.01, max_value=10.0, 
                                            value=1.0, format="%.2f", step=0.1)
            with col2:
                alpha_low = st.number_input("Alpha (bajo)", 
                                           min_value=0.001, max_value=1.0, 
                                           value=0.1, format="%.3f", step=0.01)
            
            l1_ratio = st.slider("L1 Ratio (para ElasticNet)", 
                                min_value=0.0, max_value=1.0, value=0.5, step=0.1)
            
            # Botón para ejecutar el experimento
            if st.button("Ejecutar Experimento de Regularización"):
                # División de datos
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=(1-best_split), random_state=42
                )
                
                # Definir regularizadores
                regularizers = {
                    'No Regularization': LinearRegression(),
                    f'Ridge (alpha={alpha_high})': Ridge(alpha=alpha_high, random_state=42),
                    f'Ridge (alpha={alpha_low})': Ridge(alpha=alpha_low, random_state=42),
                    f'Lasso (alpha={alpha_high})': Lasso(alpha=alpha_high, random_state=42),
                    f'Lasso (alpha={alpha_low})': Lasso(alpha=alpha_low, random_state=42),
                    f'ElasticNet (alpha={alpha_high}, l1_ratio={l1_ratio})': ElasticNet(
                        alpha=alpha_high, l1_ratio=l1_ratio, random_state=42),
                    f'ElasticNet (alpha={alpha_low}, l1_ratio={l1_ratio})': ElasticNet(
                        alpha=alpha_low, l1_ratio=l1_ratio, random_state=42)
                }
                
                results_regularizers = []
                
                with st.spinner("Ejecutando experimento..."):
                    for name, regularizer in regularizers.items():
                        try:
                            # Pipeline
                            pipeline = Pipeline([
                                ('preprocessor', preprocessor),
                                ('regressor', regularizer)
                            ])
                            
                            # Entrenamiento
                            pipeline.fit(X_train, y_train)
                            
                            # Evaluación
                            y_pred = pipeline.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            # Resultados
                            results_regularizers.append({
                                'Regularizer': name,
                                'MSE': mse,
                                'R²': r2
                            })
                        except Exception as e:
                            results_regularizers.append({
                                'Regularizer': name,
                                'MSE': 'Error',
                                'R²': 'Error',
                                'Error': str(e)
                            })
                
                # Mostrar resultados en tabla
                regularizers_df = pd.DataFrame(results_regularizers)
                st.subheader("Resultados")
                st.dataframe(regularizers_df)
                
                # Filtrar resultados con errores
                valid_results = regularizers_df[regularizers_df['MSE'] != 'Error'].copy()
                
                if not valid_results.empty:
                    # Convertir a numérico para visualización
                    valid_results['MSE'] = pd.to_numeric(valid_results['MSE'])
                    valid_results['R²'] = pd.to_numeric(valid_results['R²'])
                    
                    # Visualizar resultados
                    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Gráfico de MSE
                    sns.barplot(x='Regularizer', y='MSE', data=valid_results, ax=ax[0])
                    ax[0].set_title('MSE por Método de Regularización')
                    ax[0].set_ylabel('Mean Squared Error')
                    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')
                    ax[0].set_ylim(bottom=0)
                    
                    # Gráfico de R²
                    sns.barplot(x='Regularizer', y='R²', data=valid_results, ax=ax[1])
                    ax[1].set_title('R² por Método de Regularización')
                    ax[1].set_ylabel('R² Score')
                    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')
                    ax[1].set_ylim(0, 1)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Interpretación
                st.subheader("Interpretación")
                
                if not valid_results.empty:
                    best_reg = valid_results.iloc[valid_results['R²'].argmax()]
                    
                    st.markdown(f"""
                    - El mejor método de regularización según R² es **{best_reg['Regularizer']}** con R²={best_reg['R²']:.4f}
                    - El MSE más bajo se obtuvo con **{valid_results.iloc[valid_results['MSE'].argmin()]['Regularizer']}** (MSE={valid_results['MSE'].min():.4f})
                    - La regularización puede ayudar a prevenir el sobreajuste penalizando la magnitud de los coeficientes.
                    - Ridge (L2) penaliza el cuadrado de los coeficientes, reduciendo su magnitud pero sin llevarlos a cero.
                    - Lasso (L1) puede llevar coeficientes a cero, actuando como selector de características.
                    - ElasticNet combina ambos enfoques, ofreciendo un equilibrio entre reducción y selección.
                    """)
                    
                    # Guardar el mejor regularizador para usar en otras pestañas
                    st.session_state['best_regularizer'] = best_reg['Regularizer']
                else:
                    st.error("No se obtuvieron resultados válidos. Intente con diferentes parámetros.")
    
    # 5. MÉTRICAS Y EVALUACIÓN
    elif pages == "5. Métricas y Evaluación":
        st.markdown("<h2 class='subtitle'>5. Métricas y Evaluación del Modelo Final</h2>", unsafe_allow_html=True)
        
        # Verificar si tenemos los mejores parámetros de los experimentos anteriores
        best_split = st.session_state.get('best_split', 0.7)
        best_regularizer = st.session_state.get('best_regularizer', 'Ridge (alpha=0.1)')
        
        st.write(f"""
        En esta sección, analizamos el modelo final utilizando los mejores parámetros 
        encontrados en los experimentos anteriores:
        - Proporción de train/test: {best_split:.2f}-{1-best_split:.2f}
        - Método de regularización: {best_regularizer}
        """)
        
        # Configuración del modelo final
        st.subheader("Configuración del Modelo Final")
        
        # Permitir ajustes finales
        use_best_params = st.checkbox("Usar los mejores parámetros de los experimentos anteriores", value=True)
        
        if use_best_params:
            split = best_split # Usar los mejores parámetros encontrados
            
            # Determinar el regularizador basado en el nombre
            if 'Ridge' in best_regularizer:
                alpha = float(best_regularizer.split('alpha=')[1].split(')')[0])
                regularizer = Ridge(alpha=alpha, random_state=42)
                regularizer_name = f'Ridge (alpha={alpha})'
            elif 'Lasso' in best_regularizer:
                alpha = float(best_regularizer.split('alpha=')[1].split(')')[0])
                regularizer = Lasso(alpha=alpha, random_state=42)
                regularizer_name = f'Lasso (alpha={alpha})'
            elif 'ElasticNet' in best_regularizer:
                alpha = float(best_regularizer.split('alpha=')[1].split(',')[0])
                l1_ratio = float(best_regularizer.split('l1_ratio=')[1].split(')')[0])
                regularizer = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                regularizer_name = f'ElasticNet (alpha={alpha}, l1_ratio={l1_ratio})'
            else:
                regularizer = LinearRegression()
                regularizer_name = 'Linear Regression (No Regularization)'
        else:
            # Permitir al usuario configurar manualmente
            col1, col2 = st.columns(2)
            
            with col1:
                split = st.slider("Proporción de entrenamiento", 
                                 min_value=0.1, max_value=0.9, value=0.7, step=0.05)
            
            with col2:
                reg_method = st.selectbox("Método de regularización", 
                                        ['Sin regularización', 'Ridge', 'Lasso', 'ElasticNet'])
            
            if reg_method != 'Sin regularización':
                alpha = st.number_input("Alpha", 
                                        min_value=0.001, max_value=10.0, 
                                        value=0.1, format="%.3f", step=0.01)
                
                if reg_method == 'ElasticNet':
                    l1_ratio = st.slider("L1 Ratio", 
                                        min_value=0.0, max_value=1.0, value=0.5, step=0.1)
                    regularizer = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                    regularizer_name = f'ElasticNet (alpha={alpha}, l1_ratio={l1_ratio})'
                elif reg_method == 'Ridge':
                    regularizer = Ridge(alpha=alpha, random_state=42)
                    regularizer_name = f'Ridge (alpha={alpha})'
                else:  # Lasso
                    regularizer = Lasso(alpha=alpha, random_state=42)
                    regularizer_name = f'Lasso (alpha={alpha})'
            else:
                regularizer = LinearRegression()
                regularizer_name = 'Linear Regression (No Regularization)'
        
        # Botón para ejecutar el análisis del modelo final
        if st.button("Analizar Modelo Final"):
            # Preparación de datos
            X = data.drop('Performance Index', axis=1)
            y = data['Performance Index']
            
            # Preprocesamiento
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(drop='first'), ['Extracurricular Activities'])
                ],
                remainder='passthrough'
            )
            
            # División de datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=(1-split), random_state=42
            )
            
            with st.spinner("Entrenando el modelo final..."):
                # Pipeline
                best_model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', regularizer)
                ])
                
                # Entrenamiento
                best_model.fit(X_train, y_train)
                
                # Evaluación
                y_pred = best_model.predict(X_test)
                final_mse = mean_squared_error(y_test, y_pred)
                final_r2 = r2_score(y_test, y_pred)
            
            # Mostrar métricas finales
            st.subheader("Métricas del Modelo Final")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean Squared Error (MSE)", f"{final_mse:.4f}")
            
            with col2:
                st.metric("R² Score", f"{final_r2:.4f}")
            
            # Visualizar predicciones vs valores reales
            st.subheader("Predicciones vs Valores Reales")
            
            fig = px.scatter(
                x=y_test, y=y_pred,
                labels={'x': 'Valores Reales', 'y': 'Predicciones'},
                title='Predicciones vs Valores Reales'
            )
            
            # Añadir línea de referencia perfecta
            fig.add_trace(
                go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Predicción Perfecta'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Visualizar residuos
            st.subheader("Análisis de Residuos")
            
            residuals = y_test - y_pred
            
            fig_res = px.scatter(
                x=y_pred, y=residuals,
                labels={'x': 'Predicciones', 'y': 'Residuos'},
                title='Gráfico de Residuos'
            )
            
            fig_res.add_hline(
                y=0, line_dash="dash", line_color="red",
                annotation_text="Residuo cero",
                annotation_position="bottom right"
            )
            
            st.plotly_chart(fig_res, use_container_width=True)
            
            # Histograma de residuos
            fig_hist = px.histogram(
                residuals, nbins=30,
                labels={'value': 'Residuos'},
                title='Distribución de Residuos'
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Análisis de coeficientes
            st.subheader("Análisis de Coeficientes")
            
            try:
                # Obtener nombres de características
                feature_names = list(best_model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(['Extracurricular Activities']))
                feature_names.extend(['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced'])
                
                # Obtener coeficientes
                coefficients = best_model.named_steps['regressor'].coef_
                intercept = best_model.named_steps['regressor'].intercept_
                
                # Crear DataFrame para visualización
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefficients
                })
                coef_df = coef_df.sort_values('Coefficient', ascending=False)
                
                # Mostrar intercepción
                st.write(f"**Intercepción (valor base):** {intercept:.4f}")
                
                # Visualizar coeficientes
                fig_coef = px.bar(
                    coef_df,
                    x='Coefficient', y='Feature',
                    orientation='h',
                    title='Coeficientes del Modelo Final',
                    labels={'Coefficient': 'Valor del Coeficiente', 'Feature': 'Característica'}
                )
                
                fig_coef.add_vline(
                    x=0, line_dash="dash", line_color="black",
                    annotation_text="Cero",
                    annotation_position="top"
                )
                
                st.plotly_chart(fig_coef, use_container_width=True)
                
                # Interpretación de coeficientes
                st.subheader("Interpretación de Coeficientes")
                
                st.markdown("""
                #### Significado de la magnitud y signo de los coeficientes:
                
                - **Magnitud**: Indica la importancia relativa de cada característica.
                  Un coeficiente mayor (en valor absoluto) tiene un mayor impacto en la predicción.
                
                - **Signo**: Indica la dirección de la relación con la variable dependiente.
                  - Positivo: Al aumentar la característica, aumenta el Performance Index.
                  - Negativo: Al aumentar la característica, disminuye el Performance Index.
                """)
                
                # Análisis detallado de los coeficientes principales
                top_positive = coef_df[coef_df['Coefficient'] > 0].head(3)
                top_negative = coef_df[coef_df['Coefficient'] < 0].head(3)
                
                if not top_positive.empty:
                    st.markdown("#### Características con mayor impacto positivo:")
                    for _, row in top_positive.iterrows():
                        st.markdown(f"- **{row['Feature']}** (coef: {row['Coefficient']:.4f}): Por cada unidad de aumento en {row['Feature']}, el Performance Index aumenta en {row['Coefficient']:.4f} unidades, manteniendo las demás variables constantes.")
                
                if not top_negative.empty:
                    st.markdown("#### Características con mayor impacto negativo:")
                    for _, row in top_negative.iterrows():
                        st.markdown(f"- **{row['Feature']}** (coef: {row['Coefficient']:.4f}): Por cada unidad de aumento en {row['Feature']}, el Performance Index disminuye en {abs(row['Coefficient']):.4f} unidades, manteniendo las demás variables constantes.")
            
            except Exception as e:
                st.error(f"Error al analizar coeficientes: {str(e)}")
            
            # Interpretación de métricas
            st.subheader("Interpretación de Métricas")
            
            st.markdown(f"""
            #### Mean Squared Error (MSE): {final_mse:.4f}
            - Mide el promedio de los errores al cuadrado entre las predicciones y los valores reales.
            - Un valor más bajo indica un mejor ajuste del modelo.
            - En este caso, el error cuadrático medio es de {final_mse:.4f}, lo que significa que, en promedio, 
              nuestras predicciones tienen un error de aproximadamente {np.sqrt(final_mse):.4f} unidades de Performance Index.
            
            #### R² Score: {final_r2:.4f}
            - Indica la proporción de la varianza en la variable dependiente que es predecible a partir de las variables independientes.
            - R² = 1: Ajuste perfecto
            - R² = 0: El modelo no explica nada de la variabilidad
            - R² < 0: El modelo es peor que predecir la media
            - En este caso, nuestro modelo explica aproximadamente el {final_r2*100:.2f}% de la variabilidad en el rendimiento académico,
              lo que indica un {final_r2 > 0.9 and "excelente" or final_r2 > 0.7 and "buen" or "moderado"} ajuste a los datos.
            """)
    
    # 6. CONCLUSIONES
    elif pages == "6. Conclusiones":
        st.markdown("<h2 class='subtitle'>6. Conclusiones</h2>", unsafe_allow_html=True)
        
        # Verificar si tenemos los mejores parámetros de los experimentos anteriores
        best_split = st.session_state.get('best_split', 0.7)
        best_regularizer = st.session_state.get('best_regularizer', 'Ridge (alpha=0.1)')
        best_r2 = st.session_state.get('best_r2', 0.989)
        
        st.markdown("""
        A partir del análisis realizado, podemos extraer las siguientes conclusiones:
        """)
        
        # Conclusiones sobre el dataset
        st.subheader("Sobre el Dataset")
        st.markdown("""
        - El dataset de rendimiento estudiantil contiene 10,000 muestras y 6 variables, incluyendo 5 predictores y 1 variable objetivo.
        - La variable dependiente "Performance Index" representa el rendimiento académico de los estudiantes.
        - Las principales variables predictoras incluyen horas de estudio, calificaciones previas, actividades extracurriculares, horas de sueño y exámenes de práctica.
        """)
        
        # Conclusiones sobre las relaciones entre variables
        st.subheader("Sobre las Relaciones entre Variables")
        st.markdown("""
        - **Previous Scores** muestra la correlación más fuerte con el rendimiento académico (r ≈ 0.915), indicando que el rendimiento pasado es un fuerte predictor del rendimiento futuro.
        - **Hours Studied** tiene una correlación moderada (r ≈ 0.374), demostrando que el tiempo dedicado al estudio influye significativamente en el rendimiento.
        - **Sleep Hours** presenta una correlación débil pero positiva (r ≈ 0.048), sugiriendo que las horas de sueño tienen un impacto menor pero existente.
        - La participación en **Extracurricular Activities** muestra un efecto positivo en el rendimiento académico, lo que podría indicar beneficios de desarrollo integral.
        """)
        
        # Conclusiones sobre el modelado
        st.subheader("Sobre el Modelado y Optimización")
        st.markdown(f"""
        - La proporción óptima de train/test fue {best_split:.2f}-{1-best_split:.2f}, balanceando adecuadamente el tamaño del conjunto de entrenamiento con la necesidad de una evaluación confiable.
        - El método de optimización más efectivo fue la Regresión Lineal con Ecuación Normal, superando significativamente al Descenso de Gradiente Estocástico (SGD) en este problema específico.
        - La regularización {best_regularizer} resultó ser la más adecuada, indicando que el modelo no sufría de sobreajuste significativo.
        - El modelo final logra explicar aproximadamente un {best_r2*100:.1f}% de la variabilidad en el rendimiento académico, demostrando un ajuste excelente.
        """)
        
        # Conclusiones sobre los coeficientes
        st.subheader("Sobre los Coeficientes del Modelo")
        st.markdown("""
        - **Previous Scores** tiene el mayor impacto en términos de correlación, pero **Hours Studied** tiene el coeficiente más alto, indicando que cada hora adicional de estudio contribuye significativamente al rendimiento.
        - Todos los predictores mostraron coeficientes positivos, indicando que cada uno contribuye positivamente al rendimiento académico.
        - La participación en actividades extracurriculares tiene un efecto positivo moderado, sugiriendo beneficios que complementan el estudio académico tradicional.
        """)
        
        # Conclusiones generales y aplicaciones
        st.subheader("Conclusiones Generales")
        st.markdown("""
        - Los algoritmos de optimización pueden tener rendimientos muy diferentes dependiendo de la naturaleza de los datos y el problema.
        - La regularización es una herramienta valiosa, pero su impacto depende del problema específico y la calidad de los datos.
        - Este modelo podría ser utilizado por instituciones educativas para:
          - Identificar factores clave que influyen en el rendimiento académico
          - Desarrollar intervenciones tempranas para estudiantes en riesgo
          - Proporcionar recomendaciones personalizadas para mejorar el rendimiento
        
        Este estudio subraya la importancia de seleccionar cuidadosamente los algoritmos de optimización y ajustar sus parámetros para obtener el mejor rendimiento en aplicaciones de machine learning.
        """)
        
        # Información del equipo
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Información del Equipo")
        
        # Permitir ingresar información del equipo
        # num_members = st.number_input("Número de integrantes:", min_value=1, max_value=4, value=1, step=1)
        num_members = 3
        st.markdown("### Integrantes")
        lista_integrantes = [
            ["Flavio Arregoces", "200182105"],
            ["Jorge Sanchez", "Pendiente"],
            ["Cristian Gonzales", "200182438"],
        ]

        for i in range(num_members):
            col1, col2 = st.columns(2)
            with col1:
                st.text_input(f"Nombre del integrante {i+1}:", key=f"name_{i}", value=lista_integrantes[i][0], )
            with col2:
                st.text_input(f"Código del integrante {i+1}:", key=f"code_{i}", value=lista_integrantes[i][1], )
        
        # Añadir fecha y curso
        st.text_input("Curso:", value="Optimización",disabled=False)
        st.date_input("Fecha de entrega:", value=pd.to_datetime("2025-05-01"), disabled=False)
else:
    st.error("No se pudo cargar el dataset. Por favor, verifica que el archivo 'Student_Performance.csv' esté disponible.")