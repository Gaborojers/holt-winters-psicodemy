import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class TimeSeriesMetrics:
    """Métricas específicas para series temporales"""
    
    @staticmethod
    def calculate_forecast_accuracy(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        """Calcula métricas de precisión para pronósticos"""
        if len(actual) != len(predicted):
            raise ValueError("Las series deben tener la misma longitud")
        
        # Métricas básicas
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predicted))
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # SMAPE (Symmetric Mean Absolute Percentage Error)
        smape = 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))
        
        # R² score
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Theil's U statistic
        theil_u = np.sqrt(np.mean((predicted - actual) ** 2)) / (np.sqrt(np.mean(actual ** 2)) + np.sqrt(np.mean(predicted ** 2)))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'smape': smape,
            'r2': r2,
            'theil_u': theil_u
        }
    
    @staticmethod
    def calculate_directional_accuracy(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        """Calcula precisión direccional (si la predicción va en la dirección correcta)"""
        if len(actual) < 2 or len(predicted) < 2:
            return {'directional_accuracy': 0.0}
        
        # Calcular cambios direccionales
        actual_changes = np.diff(actual)
        predicted_changes = np.diff(predicted)
        
        # Dirección correcta
        correct_direction = np.sign(actual_changes) == np.sign(predicted_changes)
        directional_accuracy = np.mean(correct_direction) * 100
        
        return {'directional_accuracy': directional_accuracy}
    
    @staticmethod
    def calculate_volatility_metrics(series: pd.Series) -> Dict[str, float]:
        """Calcula métricas de volatilidad"""
        returns = series.pct_change().dropna()
        
        volatility = np.std(returns) * np.sqrt(252)  # Anualizada
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'volatility': volatility,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

class EducationalMetrics:
    """Métricas específicas para análisis educativo"""
    
    @staticmethod
    def calculate_engagement_metrics(data: pd.DataFrame) -> Dict[str, float]:
        """Calcula métricas de engagement estudiantil"""
        metrics = {}
        
        # Tasa de participación en mensajes
        if 'message_content' in data.columns:
            total_messages = len(data)
            messages_with_content = data['message_content'].str.len().sum()
            metrics['avg_message_length'] = messages_with_content / total_messages if total_messages > 0 else 0
        
        # Tasa de preocupaciones académicas
        if 'concern' in data.columns:
            concern_rate = data['concern'].mean() * 100
            metrics['concern_rate'] = concern_rate
        
        # Tasa de comportamiento constructivo
        if 'academic_constructive' in data.columns:
            constructive_rate = data['academic_constructive'].mean() * 100
            metrics['constructive_rate'] = constructive_rate
        
        # Tasa de bullying
        if 'bullying' in data.columns:
            bullying_rate = data['bullying'].mean() * 100
            metrics['bullying_rate'] = bullying_rate
        
        return metrics
    
    @staticmethod
    def calculate_appointment_metrics(data: pd.DataFrame) -> Dict[str, float]:
        """Calcula métricas de citas"""
        metrics = {}
        
        if data.empty:
            return metrics
        
        # Tasa de asistencia
        if 'estado_cita' in data.columns:
            total_appointments = len(data)
            completed = len(data[data['estado_cita'] == 'completado'])
            cancelled = len(data[data['estado_cita'] == 'cancelado'])
            no_show = len(data[data['estado_cita'] == 'no_asistio'])
            
            metrics['attendance_rate'] = (completed / total_appointments) * 100 if total_appointments > 0 else 0
            metrics['cancellation_rate'] = (cancelled / total_appointments) * 100 if total_appointments > 0 else 0
            metrics['no_show_rate'] = (no_show / total_appointments) * 100 if total_appointments > 0 else 0
        
        # Tiempo promedio hasta la cita
        if 'days_to_appointment' in data.columns:
            metrics['avg_days_to_appointment'] = data['days_to_appointment'].mean()
        
        return metrics
    
    @staticmethod
    def calculate_task_completion_metrics(data: pd.DataFrame) -> Dict[str, float]:
        """Calcula métricas de completado de tareas"""
        metrics = {}
        
        if data.empty:
            return metrics
        
        # Tasa de completado
        if 'task_completed' in data.columns:
            total_tasks = len(data)
            completed_tasks = data['task_completed'].sum()
            metrics['completion_rate'] = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        # Tiempo promedio de completado
        if 'time_to_complete_hours' in data.columns:
            completed_data = data[data['task_completed'] == 1]
            if not completed_data.empty:
                metrics['avg_completion_time_hours'] = completed_data['time_to_complete_hours'].mean()
        
        return metrics
    
    @staticmethod
    def calculate_student_risk_score(data: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """Calcula score de riesgo estudiantil"""
        if data.empty:
            return pd.Series()
        
        # Pesos por defecto
        default_weights = {
            'concern_rate': 0.3,
            'bullying_rate': 0.2,
            'low_engagement': 0.2,
            'missed_appointments': 0.15,
            'incomplete_tasks': 0.15
        }
        
        weights = weights or default_weights
        
        # Calcular métricas por estudiante
        student_metrics = data.groupby('sender_id').agg({
            'concern': 'mean',
            'bullying': 'mean',
            'message_length': 'mean',
            'task_completed': 'mean' if 'task_completed' in data.columns else lambda x: 1.0
        }).reset_index()
        
        # Normalizar métricas
        scaler = StandardScaler()
        normalized_metrics = scaler.fit_transform(student_metrics[['concern', 'bullying', 'message_length', 'task_completed']])
        
        # Calcular score de riesgo
        risk_score = (
            weights['concern_rate'] * normalized_metrics[:, 0] +
            weights['bullying_rate'] * normalized_metrics[:, 1] +
            weights['low_engagement'] * (-normalized_metrics[:, 2]) +  # Menor longitud = mayor riesgo
            weights['incomplete_tasks'] * (-normalized_metrics[:, 3])  # Menor completado = mayor riesgo
        )
        
        return pd.Series(risk_score, index=student_metrics['sender_id'])

class ModelComparison:
    """Herramientas para comparar modelos"""
    
    @staticmethod
    def compare_models(models_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Compara múltiples modelos"""
        comparison_data = []
        
        for model_name, metrics in models_results.items():
            row = {'model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    @staticmethod
    def plot_model_comparison(comparison_df: pd.DataFrame, metrics: List[str]) -> None:
        """Grafica comparación de modelos"""
        _, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                comparison_df.plot(x='model', y=metric, kind='bar', ax=axes[i])
                axes[i].set_title(f'{metric.upper()} por Modelo')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def get_best_model(comparison_df: pd.DataFrame, metric: str = 'rmse', lower_is_better: bool = True) -> str:
        """Obtiene el mejor modelo según una métrica"""
        if metric not in comparison_df.columns:
            raise ValueError(f"Métrica '{metric}' no encontrada en los datos")
        
        if lower_is_better:
            best_idx = comparison_df[metric].idxmin()
        else:
            best_idx = comparison_df[metric].idxmax()
        
        return comparison_df.loc[best_idx, 'model']

class AnomalyDetection:
    """Detección de anomalías en series temporales"""
    
    @staticmethod
    def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detecta outliers usando Z-score"""
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    @staticmethod
    def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
        """Detecta outliers usando IQR"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    @staticmethod
    def detect_change_points(series: pd.Series, window: int = 10) -> List[int]:
        """Detecta puntos de cambio en la serie"""
        if len(series) < 2 * window:
            return []
        
        change_points = []
        for i in range(window, len(series) - window):
            before_mean = series.iloc[i-window:i].mean()
            after_mean = series.iloc[i:i+window].mean()
            
            # Detectar cambio significativo
            if abs(after_mean - before_mean) > 2 * series.std():
                change_points.append(i)
        
        return change_points
