"""
Ejemplo de uso de la infraestructura de minería de datos educativos
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.database import db_config
from src.utils.data_loader import DataLoader
from src.utils.preproccesor import DataPreprocessor
from src.core.base_model import HoltWintersModel, ARIMAModel, EnsembleModel
from src.core.metrics import TimeSeriesMetrics, EducationalMetrics, ModelComparison
from src.utils.visualization import TimeSeriesVisualizer, EducationalVisualizer
from datetime import datetime, timedelta
import pandas as pd

def main():
    """Ejemplo principal de uso"""
    print("🚀 Iniciando análisis de datos educativos...")
    
    # 1. Conectar a la base de datos
    try:
        database = db_config.connect()
        print("✅ Conexión a MongoDB establecida")
    except Exception as e:
        print(f"❌ Error conectando a MongoDB: {e}")
        return
    
    # 2. Cargar datos
    print("\n📊 Cargando datos...")
    
    # Análisis de IA
    ai_collection = db_config.get_collection('ai_analysis')
    ai_loader = DataLoader(ai_collection)
    
    # Cargar últimos 30 días de datos
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    ai_data = ai_loader.load_ai_analysis_data(
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"📈 Datos de análisis de IA cargados: {len(ai_data)} registros")
    
    # 3. Preprocesar datos
    print("\n🔧 Preprocesando datos...")
    preprocessor = DataPreprocessor()
    
    # Limpiar datos de análisis de IA
    ai_clean = preprocessor.clean_ai_analysis_data(ai_data)
    
    # Crear features de series temporales
    ai_ts = preprocessor.create_time_series_features(ai_clean)
    
    # Agregar por día
    daily_ai = preprocessor.aggregate_by_time_period(
        ai_ts, 
        period='D',
        agg_functions={
            'bullying': ['sum', 'mean'],
            'concern': ['sum', 'mean'],
            'academic_constructive': ['sum', 'mean'],
            'message_length': ['mean', 'std']
        }
    )
    
    print(f"📊 Datos agregados por día: {len(daily_ai)} días")
    
    # 4. Análisis de métricas educativas
    print("\n📚 Calculando métricas educativas...")
    
    engagement_metrics = EducationalMetrics.calculate_engagement_metrics(ai_clean)
    print("📊 Métricas de engagement:")
    for metric, value in engagement_metrics.items():
        print(f"  - {metric}: {value:.2f}")
    
    # 5. Modelado de series temporales
    print("\n🤖 Entrenando modelos de series temporales...")
    
    # Usar tasa de preocupaciones como ejemplo
    if 'concern_mean' in daily_ai.columns:
        concern_series = daily_ai['concern_mean'].fillna(0)
        
        # Dividir en train/test
        train_size = int(len(concern_series) * 0.8)
        train_data = concern_series[:train_size]
        test_data = concern_series[train_size:]
        
        print(f"📈 Datos de entrenamiento: {len(train_data)} días")
        print(f"📈 Datos de prueba: {len(test_data)} días")
        
        # Modelo Holt-Winters
        hw_model = HoltWintersModel(seasonal_periods=7)
        hw_model.fit(train_data)
        hw_forecast = hw_model.predict(len(test_data))
        
        # Modelo ARIMA
        arima_model = ARIMAModel(order=(1, 1, 1))
        arima_model.fit(train_data)
        arima_forecast = arima_model.predict(len(test_data))
        
        # Evaluar modelos
        hw_metrics = TimeSeriesMetrics.calculate_forecast_accuracy(test_data, hw_forecast)
        arima_metrics = TimeSeriesMetrics.calculate_forecast_accuracy(test_data, arima_forecast)
        
        print("\n📊 Resultados de modelos:")
        print("Holt-Winters:")
        for metric, value in hw_metrics.items():
            print(f"  - {metric}: {value:.4f}")
        
        print("ARIMA:")
        for metric, value in arima_metrics.items():
            print(f"  - {metric}: {value:.4f}")
        
        # 6. Visualizaciones
        print("\n📊 Generando visualizaciones...")
        
        # Visualizador de series temporales
        ts_viz = TimeSeriesVisualizer()
        
        # Graficar serie original
        ts_viz.plot_time_series(concern_series, "Tasa de Preocupaciones Diarias")
        
        # Comparar pronósticos
        forecasts = {
            'Holt-Winters': hw_forecast,
            'ARIMA': arima_forecast
        }
        ts_viz.plot_multiple_forecasts(test_data, forecasts, "Comparación de Pronósticos")
        
        # Visualizador educativo
        edu_viz = EducationalVisualizer()
        
        # Graficar tendencias de engagement
        edu_viz.plot_engagement_trends(ai_clean)
        
        # Graficar mapa de calor de actividad
        edu_viz.plot_student_activity_heatmap(ai_clean)
        
        # 7. Análisis de riesgo estudiantil
        print("\n⚠️ Calculando scores de riesgo estudiantil...")
        
        risk_scores = EducationalMetrics.calculate_student_risk_score(ai_clean)
        
        if not risk_scores.empty:
            print(f"📊 Scores de riesgo calculados para {len(risk_scores)} estudiantes")
            print("Top 5 estudiantes en riesgo:")
            top_risk = risk_scores.nlargest(5)
            for student_id, score in top_risk.items():
                print(f"  - Estudiante {student_id}: {score:.3f}")
            
            # Graficar dashboard de riesgo
            edu_viz.plot_student_risk_dashboard(risk_scores)
        
        # 8. Comparación de modelos
        print("\n🔍 Comparando modelos...")
        
        models_results = {
            'Holt-Winters': hw_metrics,
            'ARIMA': arima_metrics
        }
        
        comparison_df = ModelComparison.compare_models(models_results)
        print("\n📊 Comparación de modelos:")
        print(comparison_df.to_string(index=False))
        
        best_model = ModelComparison.get_best_model(comparison_df, metric='rmse')
        print(f"\n🏆 Mejor modelo según RMSE: {best_model}")
        
    else:
        print("⚠️ No hay suficientes datos para modelado de series temporales")
    
    # 9. Cerrar conexión
    db_config.close()
    print("\n✅ Análisis completado exitosamente!")

def example_with_appointments():
    """Ejemplo con datos de citas"""
    print("\n📅 Ejemplo con datos de citas...")
    
    try:
        database = db_config.connect()
        
        # Cargar datos de citas
        appointments_collection = db_config.get_collection('appointments')
        appointments_loader = DataLoader(appointments_collection)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        appointments_data = appointments_loader.load_appointments_data(
            start_date=start_date,
            end_date=end_date
        )
        
        if not appointments_data.empty:
            print(f"📅 Datos de citas cargados: {len(appointments_data)} registros")
            
            # Preprocesar
            preprocessor = DataPreprocessor()
            appointments_clean = preprocessor.clean_appointments_data(appointments_data)
            
            # Calcular métricas
            appointment_metrics = EducationalMetrics.calculate_appointment_metrics(appointments_clean)
            print("📊 Métricas de citas:")
            for metric, value in appointment_metrics.items():
                print(f"  - {metric}: {value:.2f}")
            
            # Visualizar
            edu_viz = EducationalVisualizer()
            edu_viz.plot_appointment_analysis(appointments_clean)
            
        else:
            print("⚠️ No hay datos de citas disponibles")
            
    except Exception as e:
        print(f"❌ Error en análisis de citas: {e}")
    finally:
        db_config.close()

def example_with_tasks():
    """Ejemplo con datos de tareas"""
    print("\n📝 Ejemplo con datos de tareas...")
    
    try:
        database = db_config.connect()
        
        # Cargar datos de tareas
        tasks_collection = db_config.get_collection('tasks')
        tasks_loader = DataLoader(tasks_collection)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        tasks_data = tasks_loader.load_tasks_data(
            start_date=start_date,
            end_date=end_date
        )
        
        if not tasks_data.empty:
            print(f"📝 Datos de tareas cargados: {len(tasks_data)} registros")
            
            # Preprocesar
            preprocessor = DataPreprocessor()
            tasks_clean = preprocessor.clean_tasks_data(tasks_data)
            
            # Calcular métricas
            task_metrics = EducationalMetrics.calculate_task_completion_metrics(tasks_clean)
            print("📊 Métricas de tareas:")
            for metric, value in task_metrics.items():
                print(f"  - {metric}: {value:.2f}")
            
            # Visualizar
            edu_viz = EducationalVisualizer()
            edu_viz.plot_task_completion_analysis(tasks_clean)
            
        else:
            print("⚠️ No hay datos de tareas disponibles")
            
    except Exception as e:
        print(f"❌ Error en análisis de tareas: {e}")
    finally:
        db_config.close()

if __name__ == "__main__":
    # Ejecutar análisis principal
    main()
    
    # Ejemplos adicionales
    example_with_appointments()
    example_with_tasks() 