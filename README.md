# 📊 Infraestructura de Minería de Datos Educativos

Este proyecto implementa una infraestructura completa para el análisis de datos educativos utilizando series temporales, con enfoque en la detección de patrones, predicciones y análisis de riesgo estudiantil.

## 🏗️ Arquitectura del Proyecto

```
.venv/
├── src/
│   ├── config/
│   │   └── database.py          # Configuración de MongoDB
│   ├── core/
│   │   ├── base_model.py        # Modelos de series temporales
│   │   └── metrics.py           # Métricas de evaluación
│   └── utils/
│       ├── data_loader.py       # Cargador de datos
│       ├── preproccesor.py      # Preprocesamiento
│       └── visualization.py     # Visualizaciones
├── notebooks/                   # Jupyter notebooks
├── models/                      # Modelos guardados
├── requirements.txt             # Dependencias
└── example_usage.py            # Ejemplo de uso
```

## 🚀 Características Principales

### 📈 Modelos de Series Temporales
- **Holt-Winters**: Para series con estacionalidad
- **ARIMA**: Para tendencias y autocorrelación
- **Ensemble**: Combinación de múltiples modelos
- **Auto-ARIMA**: Selección automática de parámetros

### 📊 Análisis Educativo
- **Métricas de Engagement**: Participación, preocupaciones, bullying
- **Análisis de Citas**: Asistencia, cancelaciones, patrones temporales
- **Completado de Tareas**: Tasas de éxito, tiempo de completado
- **Score de Riesgo**: Identificación de estudiantes en riesgo

### 📈 Visualizaciones
- **Series Temporales**: Componentes, tendencias, estacionalidad
- **Mapas de Calor**: Actividad por hora y día
- **Dashboards**: Métricas educativas y riesgo estudiantil
- **Comparaciones**: Múltiples modelos y pronósticos

## 🛠️ Instalación

1. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

2. **Configurar MongoDB**:
```python
# Las credenciales están configuradas por defecto
# mongodb://admin:secret123@3.223.236.109:27017
```

## 📖 Uso Básico

### 1. Cargar y Preprocesar Datos

```python
from src.config.database import db_config
from src.utils.data_loader import DataLoader
from src.utils.preproccesor import DataPreprocessor

# Conectar a MongoDB
database = db_config.connect()
collection = db_config.get_collection('ai_analysis')

# Cargar datos
loader = DataLoader(collection)
data = loader.load_ai_analysis_data(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)

# Preprocesar
preprocessor = DataPreprocessor()
clean_data = preprocessor.clean_ai_analysis_data(data)
```

### 2. Entrenar Modelos de Series Temporales

```python
from src.core.base_model import HoltWintersModel, ARIMAModel

# Preparar datos
daily_data = preprocessor.aggregate_by_time_period(clean_data, period='D')
concern_series = daily_data['concern_mean']

# Holt-Winters
hw_model = HoltWintersModel(seasonal_periods=7)
hw_model.fit(concern_series)
hw_forecast = hw_model.predict(steps=7)

# ARIMA
arima_model = ARIMAModel(order=(1, 1, 1))
arima_model.fit(concern_series)
arima_forecast = arima_model.predict(steps=7)
```

### 3. Evaluar Modelos

```python
from src.core.metrics import TimeSeriesMetrics, ModelComparison

# Calcular métricas
hw_metrics = TimeSeriesMetrics.calculate_forecast_accuracy(actual, hw_forecast)
arima_metrics = TimeSeriesMetrics.calculate_forecast_accuracy(actual, arima_forecast)

# Comparar modelos
models_results = {
    'Holt-Winters': hw_metrics,
    'ARIMA': arima_metrics
}
comparison_df = ModelComparison.compare_models(models_results)
best_model = ModelComparison.get_best_model(comparison_df, metric='rmse')
```

### 4. Visualizar Resultados

```python
from src.utils.visualization import TimeSeriesVisualizer, EducationalVisualizer

# Visualizador de series temporales
ts_viz = TimeSeriesVisualizer()
ts_viz.plot_time_series(concern_series, "Tasa de Preocupaciones")
ts_viz.plot_multiple_forecasts(actual, forecasts, "Comparación de Pronósticos")

# Visualizador educativo
edu_viz = EducationalVisualizer()
edu_viz.plot_engagement_trends(clean_data)
edu_viz.plot_student_activity_heatmap(clean_data)
```

## 📊 Análisis Específicos

### 🎯 Predicciones de Riesgo Académico

```python
from src.core.metrics import EducationalMetrics

# Calcular score de riesgo por estudiante
risk_scores = EducationalMetrics.calculate_student_risk_score(clean_data)

# Identificar estudiantes en riesgo
high_risk_students = risk_scores[risk_scores > risk_scores.quantile(0.8)]
```

### 📅 Análisis de Citas

```python
# Cargar datos de citas
appointments_data = loader.load_appointments_data(
    start_date=start_date,
    end_date=end_date
)

# Calcular métricas
appointment_metrics = EducationalMetrics.calculate_appointment_metrics(appointments_data)

# Visualizar
edu_viz.plot_appointment_analysis(appointments_data)
```

### 📝 Análisis de Tareas

```python
# Cargar datos de tareas
tasks_data = loader.load_tasks_data(
    start_date=start_date,
    end_date=end_date
)

# Calcular métricas
task_metrics = EducationalMetrics.calculate_task_completion_metrics(tasks_data)

# Visualizar
edu_viz.plot_task_completion_analysis(tasks_data)
```

## 🔍 Métricas Disponibles

### Series Temporales
- **RMSE**: Error cuadrático medio
- **MAE**: Error absoluto medio
- **MAPE**: Error porcentual absoluto medio
- **SMAPE**: Error porcentual absoluto simétrico
- **R²**: Coeficiente de determinación
- **Theil's U**: Estadístico de Theil

### Educativas
- **Engagement**: Tasa de participación, longitud de mensajes
- **Preocupaciones**: Tasa de preocupaciones académicas
- **Bullying**: Tasa de detección de bullying
- **Asistencia**: Tasa de asistencia a citas
- **Completado**: Tasa de completado de tareas

## 📈 Predicciones Efectivas

### 🎯 Para Implementar

1. **Deserción Estudiantil**:
   - Patrones de disminución en participación
   - Aumento en preocupaciones académicas
   - Reducción en asistencia a citas

2. **Bajo Rendimiento**:
   - Correlación entre engagement y resultados
   - Patrones de completado de tareas
   - Tendencias en comportamiento constructivo

3. **Problemas de Comportamiento**:
   - Detección temprana de bullying
   - Patrones de comunicación problemática
   - Cambios en patrones de actividad

4. **Optimización de Recursos**:
   - Mejores horarios para citas
   - Carga de trabajo de tutores
   - Efectividad de intervenciones

## 🚀 Ejecutar Ejemplo Completo

```bash
cd .venv
python src/example_usage.py
```

## 📋 Próximos Pasos

1. **Desarrollar notebooks específicos**:
   - `01_data_exploration.ipynb`: Exploración inicial
   - `02_forecasting_development.ipynb`: Desarrollo de modelos
   - `03_clustering_development.ipynb`: Clustering de estudiantes
   - `04_integration_pipeline.ipynb`: Pipeline completo

2. **Implementar modelos adicionales**:
   - Prophet (Facebook)
   - LSTM/RNN para patrones complejos
   - Modelos de clasificación para riesgo

3. **Mejorar visualizaciones**:
   - Dashboards interactivos con Streamlit
   - Alertas automáticas
   - Reportes automáticos

## 🔧 Configuración Avanzada

### Variables de Entorno
```bash
MONGODB_URI=mongodb://admin:secret123@3.223.236.109:27017
DATABASE_NAME=tutoring_platform
```

### Personalización de Modelos
```python
# Holt-Winters personalizado
hw_model = HoltWintersModel(
    seasonal_periods=7,  # Semanal
    trend='add',         # Tendencia aditiva
    seasonal='add'       # Estacionalidad aditiva
)

# ARIMA con parámetros específicos
arima_model = ARIMAModel(
    order=(2, 1, 2),     # p=2, d=1, q=2
    seasonal_order=(1, 1, 1, 7)  # SARIMA
)
```

## 📞 Soporte

Para preguntas o problemas:
1. Revisar el archivo `example_usage.py`
2. Verificar la conexión a MongoDB
3. Instalar todas las dependencias
4. Ejecutar con datos de prueba

---

**¡Listo para analizar datos educativos y hacer predicciones efectivas! 🎓📊** 