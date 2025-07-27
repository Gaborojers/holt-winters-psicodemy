import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de matplotlib
plt.style.use('default')
sns.set_palette("husl")

class TimeSeriesVisualizer:
    """Visualizaciones para series temporales"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
    
    def plot_time_series(
        self, 
        data: pd.Series, 
        title: str = "Serie Temporal",
        show_trend: bool = True,
        show_seasonal: bool = True
    ) -> None:
        """Grafica una serie temporal con componentes"""
        _, axes = plt.subplots(2, 1, figsize=self.figsize)
        
        # Serie original
        axes[0].plot(data.index, data.values, linewidth=2, alpha=0.8)
        axes[0].set_title(f"{title} - Serie Original")
        axes[0].grid(True, alpha=0.3)
        
        # Componentes
        if show_trend or show_seasonal:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            try:
                # Descomposición estacional
                decomposition = seasonal_decompose(data, period=min(7, len(data)//2))
                
                if show_trend:
                    axes[1].plot(data.index, decomposition.trend, 'r-', linewidth=2, label='Tendencia')
                
                if show_seasonal:
                    axes[1].plot(data.index, decomposition.seasonal, 'g-', linewidth=1, alpha=0.7, label='Estacionalidad')
                
                axes[1].plot(data.index, data.values, 'b-', linewidth=1, alpha=0.5, label='Original')
                axes[1].set_title("Componentes de la Serie")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"Error en descomposición: {e}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_forecast_comparison(
        self, 
        actual: pd.Series, 
        predicted: pd.Series,
        title: str = "Comparación de Pronósticos"
    ) -> None:
        """Compara valores actuales vs predichos"""
        _, axes = plt.subplots(2, 1, figsize=self.figsize)
        
        # Serie temporal
        axes[0].plot(actual.index, actual.values, 'b-', linewidth=2, label='Actual', alpha=0.8)
        axes[0].plot(predicted.index, predicted.values, 'r--', linewidth=2, label='Predicho', alpha=0.8)
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuos
        residuals = actual - predicted
        axes[1].plot(residuals.index, residuals.values, 'g-', linewidth=1, alpha=0.7)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_title("Residuos")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_multiple_forecasts(
        self, 
        actual: pd.Series, 
        forecasts: Dict[str, pd.Series],
        title: str = "Múltiples Pronósticos"
    ) -> None:
        """Grafica múltiples pronósticos en una sola figura"""
        plt.figure(figsize=self.figsize)
        
        # Serie actual
        plt.plot(actual.index, actual.values, 'k-', linewidth=3, label='Actual', alpha=0.9)
        
        # Pronósticos
        colors = plt.cm.Set3(np.linspace(0, 1, len(forecasts)))
        for i, (name, forecast) in enumerate(forecasts.items()):
            plt.plot(forecast.index, forecast.values, '--', linewidth=2, 
                    label=name, color=colors[i], alpha=0.8)
        
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_autocorrelation(self, data: pd.Series, lags: int = 40) -> None:
        """Grafica autocorrelación y autocorrelación parcial"""
        _, axes = plt.subplots(2, 1, figsize=self.figsize)
        
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        # Autocorrelación
        plot_acf(data, lags=lags, ax=axes[0])
        axes[0].set_title("Autocorrelación")
        
        # Autocorrelación parcial
        plot_pacf(data, lags=lags, ax=axes[1])
        axes[1].set_title("Autocorrelación Parcial")
        
        plt.tight_layout()
        plt.show()
    
    def plot_rolling_statistics(self, data: pd.Series, window: int = 30) -> None:
        """Grafica estadísticas móviles"""
        _, axes = plt.subplots(3, 1, figsize=self.figsize)
        
        # Media móvil
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        
        axes[0].plot(data.index, data.values, 'b-', linewidth=1, alpha=0.7, label='Original')
        axes[0].plot(rolling_mean.index, rolling_mean.values, 'r-', linewidth=2, label=f'Media móvil ({window})')
        axes[0].set_title("Media Móvil")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Desviación estándar móvil
        axes[1].plot(rolling_std.index, rolling_std.values, 'g-', linewidth=2)
        axes[1].set_title("Desviación Estándar Móvil")
        axes[1].grid(True, alpha=0.3)
        
        # Volatilidad
        volatility = data.pct_change().rolling(window=window).std()
        axes[2].plot(volatility.index, volatility.values, 'm-', linewidth=2)
        axes[2].set_title("Volatilidad Móvil")
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class EducationalVisualizer:
    """Visualizaciones específicas para análisis educativo"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
    
    def plot_engagement_trends(self, data: pd.DataFrame, date_column: str = 'created_at') -> None:
        """Grafica tendencias de engagement"""
        if data.empty:
            print("No hay datos para visualizar")
            return
        
        # Agregar por fecha
        daily_data = data.groupby(pd.Grouper(key=date_column, freq='D')).agg({
            'concern': 'mean',
            'bullying': 'mean',
            'academic_constructive': 'mean',
            'message_length': 'mean'
        }).fillna(0)
        
        _, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Preocupaciones
        axes[0, 0].plot(daily_data.index, daily_data['concern'] * 100, 'r-', linewidth=2)
        axes[0, 0].set_title("Tasa de Preocupaciones (%)")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Bullying
        axes[0, 1].plot(daily_data.index, daily_data['bullying'] * 100, 'orange', linewidth=2)
        axes[0, 1].set_title("Tasa de Bullying (%)")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Comportamiento constructivo
        axes[1, 0].plot(daily_data.index, daily_data['academic_constructive'] * 100, 'g-', linewidth=2)
        axes[1, 0].set_title("Tasa de Comportamiento Constructivo (%)")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Longitud de mensajes
        axes[1, 1].plot(daily_data.index, daily_data['message_length'], 'b-', linewidth=2)
        axes[1, 1].set_title("Longitud Promedio de Mensajes")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_student_activity_heatmap(self, data: pd.DataFrame) -> None:
        """Grafica mapa de calor de actividad estudiantil por hora y día"""
        if data.empty or 'created_at' not in data.columns:
            print("No hay datos de fecha para visualizar")
            return
        
        # Crear features de tiempo
        data_copy = data.copy()
        data_copy['hour'] = data_copy['created_at'].dt.hour
        data_copy['day_of_week'] = data_copy['created_at'].dt.dayofweek
        
        # Crear matriz de actividad
        activity_matrix = data_copy.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        
        # Nombres de días
        day_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        activity_matrix.index = [day_names[i] for i in activity_matrix.index]
        
        plt.figure(figsize=self.figsize)
        sns.heatmap(activity_matrix, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Número de Mensajes'})
        plt.title("Mapa de Calor de Actividad Estudiantil")
        plt.xlabel("Hora del Día")
        plt.ylabel("Día de la Semana")
        plt.tight_layout()
        plt.show()
    
    def plot_appointment_analysis(self, data: pd.DataFrame) -> None:
        """Grafica análisis de citas"""
        if data.empty:
            print("No hay datos de citas para visualizar")
            return
        
        _, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Distribución de estados
        if 'estado_cita' in data.columns:
            status_counts = data['estado_cita'].value_counts()
            axes[0, 0].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title("Distribución de Estados de Citas")
        
        # Tiempo hasta la cita
        if 'days_to_appointment' in data.columns:
            axes[0, 1].hist(data['days_to_appointment'], bins=20, alpha=0.7, color='skyblue')
            axes[0, 1].set_title("Distribución de Días hasta la Cita")
            axes[0, 1].set_xlabel("Días")
            axes[0, 1].set_ylabel("Frecuencia")
        
        # Tasa de asistencia por día de la semana
        if 'fecha_cita' in data.columns:
            data_copy = data.copy()
            data_copy['day_of_week'] = data_copy['fecha_cita'].dt.dayofweek
            attendance_by_day = data_copy.groupby('day_of_week')['estado_cita'].apply(
                lambda x: (x == 'completada').mean() * 100
            )
            day_names = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
            axes[1, 0].bar(range(len(attendance_by_day)), attendance_by_day.values, color='lightgreen')
            axes[1, 0].set_title("Tasa de Asistencia por Día")
            axes[1, 0].set_xticks(range(len(day_names)))
            axes[1, 0].set_xticklabels(day_names)
            axes[1, 0].set_ylabel("Tasa de Asistencia (%)")
        
        # Tasa de asistencia por hora
        if 'fecha_cita' in data.columns:
            data_copy['hour'] = data_copy['fecha_cita'].dt.hour
            attendance_by_hour = data_copy.groupby('hour')['estado_cita'].apply(
                lambda x: (x == 'completada').mean() * 100
            )
            axes[1, 1].bar(attendance_by_hour.index, attendance_by_hour.values, color='lightcoral')
            axes[1, 1].set_title("Tasa de Asistencia por Hora")
            axes[1, 1].set_xlabel("Hora")
            axes[1, 1].set_ylabel("Tasa de Asistencia (%)")
        
        plt.tight_layout()
        plt.show()
    
    def plot_task_completion_analysis(self, data: pd.DataFrame) -> None:
        """Grafica análisis de completado de tareas"""
        if data.empty:
            print("No hay datos de tareas para visualizar")
            return
        
        _, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Tasa de completado
        if 'task_completed' in data.columns:
            completion_rate = data['task_completed'].mean() * 100
            axes[0, 0].pie([completion_rate, 100-completion_rate], 
                          labels=['Completadas', 'Pendientes'], 
                          autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
            axes[0, 0].set_title("Tasa de Completado de Tareas")
        
        # Tiempo de completado
        if 'time_to_complete_hours' in data.columns:
            completed_tasks = data[data['task_completed'] == True]
            if not completed_tasks.empty:
                axes[0, 1].hist(completed_tasks['time_to_complete_hours'], bins=20, alpha=0.7, color='skyblue')
                axes[0, 1].set_title("Distribución de Tiempo de Completado")
                axes[0, 1].set_xlabel("Horas")
                axes[0, 1].set_ylabel("Frecuencia")
        
        # Completado por día de la semana
        if 'fecha_cita' in data.columns:
            data_copy = data.copy()
            data_copy['day_of_week'] = data_copy['fecha_cita'].dt.dayofweek
            completion_by_day = data_copy.groupby('day_of_week')['task_completed'].mean() * 100
            day_names = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
            axes[1, 0].bar(range(len(completion_by_day)), completion_by_day.values, color='lightblue')
            axes[1, 0].set_title("Tasa de Completado por Día")
            axes[1, 0].set_xticks(range(len(day_names)))
            axes[1, 0].set_xticklabels(day_names)
            axes[1, 0].set_ylabel("Tasa de Completado (%)")
        
        # Longitud de tareas vs completado
        if 'task_description' in data.columns and 'task_completed' in data.columns:
            task_lengths = data['task_description'].str.len()
            axes[1, 1].scatter(task_lengths, data['task_completed'], alpha=0.6, color='purple')
            axes[1, 1].set_title("Longitud de Tarea vs Completado")
            axes[1, 1].set_xlabel("Longitud de Tarea")
            axes[1, 1].set_ylabel("Completado (True/False)")
        
        plt.tight_layout()
        plt.show()
    
    def plot_student_risk_dashboard(self, risk_scores: pd.Series) -> None:
        """Grafica dashboard de riesgo estudiantil"""
        score_riesgo = "Score de Riesgo"
        if risk_scores.empty:
            print("No hay scores de riesgo para visualizar")
            return
        
        _, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Distribución de scores de riesgo
        axes[0, 0].hist(risk_scores.values, bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[0, 0].set_title("Distribución de Scores de Riesgo")
        axes[0, 0].set_xlabel(score_riesgo)
        axes[0, 0].set_ylabel("Frecuencia")
        
        # Top 10 estudiantes en riesgo
        top_risk = risk_scores.nlargest(10)
        axes[0, 1].barh(range(len(top_risk)), top_risk.values, color='darkred')
        axes[0, 1].set_yticks(range(len(top_risk)))
        axes[0, 1].set_yticklabels([f"Estudiante {i}" for i in top_risk.index])
        axes[0, 1].set_title("Top 10 Estudiantes en Riesgo")
        axes[0, 1].set_xlabel(score_riesgo)
        
        # Box plot de scores
        axes[1, 0].boxplot(risk_scores.values, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
        axes[1, 0].set_title("Box Plot de Scores de Riesgo")
        axes[1, 0].set_ylabel(score_riesgo)
        
        # Categorización de riesgo
        risk_categories = pd.cut(risk_scores, bins=3, labels=['Bajo', 'Medio', 'Alto'])
        category_counts = risk_categories.value_counts()
        colors = ['lightgreen', 'orange', 'red']
        axes[1, 1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', colors=colors)
        axes[1, 1].set_title("Distribución por Categoría de Riesgo")
        
        plt.tight_layout()
        plt.show()


