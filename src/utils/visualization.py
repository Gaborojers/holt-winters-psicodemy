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

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        def plot_if_varies(ax, y, color, title, ylabel):
            if np.allclose(y, 0) or np.all(y == y[0]):
                ax.text(0.5, 0.5, 'Sin variación', ha='center', va='center', fontsize=12, color='gray', transform=ax.transAxes)
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
            else:
                ax.plot(daily_data.index, y, color, linewidth=2)
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)

        # Preocupaciones
        plot_if_varies(axes[0, 0], daily_data['concern'] * 100, 'r-', "Tasa de Preocupaciones (%)", "% Preocupación")
        # Bullying
        plot_if_varies(axes[0, 1], daily_data['bullying'] * 100, 'orange', "Tasa de Bullying (%)", "% Bullying")
        # Comportamiento constructivo
        plot_if_varies(axes[1, 0], daily_data['academic_constructive'] * 100, 'g-', "Tasa de Comportamiento Constructivo (%)", "% Constructivo")
        # Longitud de mensajes
        axes[1, 1].plot(daily_data.index, daily_data['message_length'], 'b-', linewidth=2)
        axes[1, 1].set_title("Longitud Promedio de Mensajes")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylabel("Longitud")

        # Rotar fechas en todos los subplots
        for ax in axes.flat:
            plt.setp(ax.get_xticklabels(), rotation=45)

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
        """Grafica análisis de citas con visualizaciones individuales"""
        if data.empty:
            print("No hay datos de citas para visualizar")
            return

        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Variables para el resumen estadístico
        valid_days = pd.Series(dtype=float)
        unique_values = 0

        # 1. DISTRIBUCIÓN DE ESTADOS DE CITAS
        if 'estado_cita' in data.columns:
            print("📊 Generando distribución de estados de citas...")
            status_counts = data['estado_cita'].value_counts()

            plt.figure(figsize=(10, 8))
            colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow', 'lightgray']
            _, _, autotexts = plt.pie(status_counts.values, 
                                              labels=status_counts.index, 
                                              autopct='%1.1f%%',
                                              colors=colors[:len(status_counts)],
                                              startangle=90)

            # Mejorar la apariencia del texto
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')

            plt.title("Distribución de Estados de Citas", fontweight='bold', fontsize=14)
            plt.axis('equal')
            plt.tight_layout()
            plt.show()

        # 3. TASA DE ASISTENCIA POR DÍA DE LA SEMANA
        if 'fecha_cita' in data.columns:
            print("📅 Generando tasa de asistencia por día...")
            data_copy = data.copy()
            data_copy['day_of_week'] = data_copy['fecha_cita'].dt.dayofweek
            attendance_by_day = data_copy.groupby('day_of_week')['estado_cita'].apply(
                lambda x: (x == 'completada').mean() * 100
            )

            # Ordenar por días de la semana (Lunes a Domingo)
            day_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
            attendance_by_day = attendance_by_day.reindex(range(7))

            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(attendance_by_day)), attendance_by_day.values, 
                          color=['lightgreen' if x > 70 else 'orange' if x > 50 else 'red' for x in attendance_by_day.values])
            plt.title("Tasa de Asistencia por Día de la Semana", fontweight='bold', fontsize=14)
            plt.xlabel("Día de la Semana", fontsize=12)
            plt.ylabel("Tasa de Asistencia (%)", fontsize=12)
            plt.xticks(range(len(day_names)), day_names, rotation=45)
            plt.grid(True, alpha=0.3)

            # Agregar valores en las barras
            for bar, value in zip(bars, attendance_by_day.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                         f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

            # Agregar líneas horizontales para umbrales
            mean_attendance = attendance_by_day.mean()
            plt.axhline(y=mean_attendance, color='red', linestyle='--', linewidth=2, label=f'Promedio: {mean_attendance:.1f}%')
            plt.axhline(y=70, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Umbral Alto (70%)')
            plt.axhline(y=50, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Umbral Medio (50%)')
            plt.legend()
            plt.tight_layout()
            plt.show()

        # 3. TASA DE ASISTENCIA POR HORA
        if 'fecha_cita' in data.columns:
            print("⏰ Generando tasa de asistencia por hora...")
            data_copy['hour'] = data_copy['fecha_cita'].dt.hour
            attendance_by_hour = data_copy.groupby('hour')['estado_cita'].apply(
                lambda x: (x == 'completada').mean() * 100
            )

            plt.figure(figsize=(12, 6))
            bars = plt.bar(attendance_by_hour.index, attendance_by_hour.values, 
                          color=['lightgreen' if x > 70 else 'orange' if x > 50 else 'red' for x in attendance_by_hour.values])
            plt.title("Tasa de Asistencia por Hora del Día", fontweight='bold', fontsize=14)
            plt.xlabel("Hora del Día", fontsize=12)
            plt.ylabel("Tasa de Asistencia (%)", fontsize=12)
            plt.xticks(range(0, 24, 2))
            plt.grid(True, alpha=0.3)

            # Agregar valores en las barras
            for bar, value in zip(bars, attendance_by_hour.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                         f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

            # Agregar líneas horizontales para umbrales
            mean_attendance = attendance_by_hour.mean()
            plt.axhline(y=mean_attendance, color='red', linestyle='--', linewidth=2, label=f'Promedio: {mean_attendance:.1f}%')
            plt.axhline(y=70, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Umbral Alto (70%)')
            plt.axhline(y=50, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Umbral Medio (50%)')
            plt.legend()
            plt.tight_layout()
            plt.show()

        # RESUMEN ESTADÍSTICO
        print("\n📋 RESUMEN DE ANÁLISIS DE CITAS:")
        print("=" * 50)

        if 'estado_cita' in data.columns:
            print(f"\n📊 ESTADOS DE CITAS:")
            for status, count in status_counts.items():
                percentage = (count / len(data)) * 100
                print(f"   {status}: {count} citas ({percentage:.1f}%)")


        if 'fecha_cita' in data.columns:
            print(f"\n📅 DÍA CON MAYOR ASISTENCIA: {day_names[attendance_by_day.idxmax()]} ({attendance_by_day.max():.1f}%)")
            print(f"⏰ HORA CON MAYOR ASISTENCIA: {attendance_by_hour.idxmax()}:00 hrs ({attendance_by_hour.max():.1f}%)")
        
    def plot_task_completion_analysis(self, data: pd.DataFrame) -> None:
        """Grafica análisis de completado de tareas con visualizaciones individuales"""
        if data.empty:
            print("No hay datos de tareas para visualizar")
            return

        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. TASA DE COMPLETADO DE TAREAS
        if 'task_completed' in data.columns:
            print("📊 Generando tasa de completado de tareas...")
            completion_rate = data['task_completed'].mean() * 100

            plt.figure(figsize=(10, 8))
            wedges, texts, autotexts = plt.pie([completion_rate, 100-completion_rate], 
                                              labels=['Completadas', 'Pendientes'], 
                                              autopct='%1.1f%%', 
                                              colors=['lightgreen', 'lightcoral'],
                                              startangle=90)

            # Mejorar la apariencia del texto
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')

            plt.title("Tasa de Completado de Tareas", fontweight='bold', fontsize=14)
            plt.axis('equal')
            plt.tight_layout()
            plt.show()

        # 2. ANÁLISIS DE TIEMPO DE COMPLETADO (REFACTORIZADO)
        if 'time_to_complete_hours' in data.columns:
            print("⏰ Generando análisis de tiempo de completado...")
            completed_tasks = data[data['task_completed'] == True]

            if not completed_tasks.empty:
                valid_times = completed_tasks['time_to_complete_hours'].dropna()

                if len(valid_times) > 0:
                    # Análisis de la distribución del tiempo
                    unique_times = valid_times.nunique()
                    time_range = valid_times.max() - valid_times.min()

                    print(f"   - Tareas completadas: {len(valid_times)}")
                    print(f"   - Tiempo promedio: {valid_times.mean():.1f} horas")
                    print(f"   - Tiempo mediano: {valid_times.median():.1f} horas")
                    print(f"   - Rango: {valid_times.min():.1f} - {valid_times.max():.1f} horas")

                    # Si hay suficiente variación, usar histograma
                    if unique_times > 5 and time_range > 1:
                        plt.figure(figsize=(12, 6))
                        plt.hist(valid_times, bins=min(20, unique_times), alpha=0.7, color='skyblue', edgecolor='black')
                        plt.title("Distribución de Tiempo de Completado de Tareas", fontweight='bold', fontsize=14)
                        plt.xlabel("Tiempo de Completado (horas)", fontsize=12)
                        plt.ylabel("Frecuencia", fontsize=12)
                        plt.grid(True, alpha=0.3)

                        # Agregar líneas horizontales para umbrales
                        mean_time = valid_times.mean()
                        median_time = valid_times.median()
                        plt.axhline(y=plt.gca().get_ylim()[1] * 0.8, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_time:.1f} horas')
                        plt.axhline(y=plt.gca().get_ylim()[1] * 0.6, color='green', linestyle='--', linewidth=2, label=f'Mediana: {median_time:.1f} horas')
                        plt.legend()
                        plt.tight_layout()
                        plt.show()
                    else:
                        # Usar boxplot para distribuciones estrechas
                        plt.figure(figsize=(10, 6))
                        plt.boxplot(valid_times, vert=False, patch_artist=True, 
                                   boxprops=dict(facecolor='skyblue', alpha=0.7))
                        plt.title("Distribución de Tiempo de Completado (Boxplot)", fontweight='bold', fontsize=14)
                        plt.xlabel("Tiempo de Completado (horas)", fontsize=12)
                        plt.grid(True, alpha=0.3)

                        # Agregar estadísticas
                        stats_text = f"Media: {valid_times.mean():.1f} horas\nMediana: {valid_times.median():.1f} horas\nQ1: {valid_times.quantile(0.25):.1f} horas\nQ3: {valid_times.quantile(0.75):.1f} horas"
                        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        plt.tight_layout()
                        plt.show()

                    # Análisis adicional: Tiempo por estado de cita
                    if 'estado_cita' in data.columns:
                        print("   - Generando análisis por estado de cita...")

                        time_by_status = completed_tasks.groupby('estado_cita')['time_to_complete_hours'].agg(['mean', 'count']).round(2)

                        plt.figure(figsize=(10, 6))
                        bars = plt.bar(range(len(time_by_status)), time_by_status['mean'].values,
                                      color=['lightgreen', 'lightblue', 'lightcoral', 'lightyellow'][:len(time_by_status)])
                        plt.title("Tiempo Promedio de Completado por Estado de Cita", fontweight='bold', fontsize=14)
                        plt.xlabel("Estado de la Cita", fontsize=12)
                        plt.ylabel("Tiempo Promedio (horas)", fontsize=12)
                        plt.xticks(range(len(time_by_status)), time_by_status.index, rotation=45)
                        plt.grid(True, alpha=0.3)

                        # Agregar valores en las barras
                        for bar, value, count in zip(bars, time_by_status['mean'].values, time_by_status['count'].values):
                            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                                     f'{value:.1f}\n({count})', ha='center', va='bottom', fontweight='bold')

                        plt.tight_layout()
                        plt.show()
                else:
                    print("⚠️  No hay datos válidos de tiempo de completado")
            else:
                print("⚠️  No hay tareas completadas para analizar")

        # 3. COMPLETADO POR DÍA DE LA SEMANA
        if 'fecha_cita' in data.columns:
            print(" Generando tasa de completado por día...")
            data_copy = data.copy()
            data_copy['day_of_week'] = data_copy['fecha_cita'].dt.dayofweek
            completion_by_day = data_copy.groupby('day_of_week')['task_completed'].mean() * 100

            # Ordenar por días de la semana (Lunes a Domingo)
            day_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
            completion_by_day = completion_by_day.reindex(range(7))

            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(completion_by_day)), completion_by_day.values, 
                          color=['lightgreen' if x > 70 else 'orange' if x > 50 else 'red' for x in completion_by_day.values])
            plt.title("Tasa de Completado de Tareas por Día de la Semana", fontweight='bold', fontsize=14)
            plt.xlabel("Día de la Semana", fontsize=12)
            plt.ylabel("Tasa de Completado (%)", fontsize=12)
            plt.xticks(range(len(day_names)), day_names, rotation=45)
            plt.grid(True, alpha=0.3)

            # Agregar valores en las barras
            for bar, value in zip(bars, completion_by_day.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                         f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

            # Agregar líneas horizontales para umbrales
            mean_completion = completion_by_day.mean()
            plt.axhline(y=mean_completion, color='red', linestyle='--', linewidth=2, label=f'Promedio: {mean_completion:.1f}%')
            plt.axhline(y=70, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Umbral Alto (70%)')
            plt.axhline(y=50, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Umbral Medio (50%)')
            plt.legend()
            plt.tight_layout()
            plt.show()

        # 3. ANÁLISIS DE PRODUCTIVIDAD POR TUTOR
        if 'id_tutor' in data.columns and 'task_completed' in data.columns:
            print("👨‍�� Generando análisis de productividad por tutor...")

            tutor_productivity = data.groupby('id_tutor').agg({
                'task_completed': ['mean', 'count'],
                'task_description': 'count'
            }).round(3)

            tutor_productivity.columns = ['completion_rate', 'total_tasks', 'total_assignments']
            tutor_productivity = tutor_productivity.sort_values('completion_rate', ascending=False)

            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(tutor_productivity)), tutor_productivity['completion_rate'].values * 100,
                          color=['lightgreen' if x > 0.7 else 'orange' if x > 0.5 else 'red' for x in tutor_productivity['completion_rate'].values])
            plt.title("Tasa de Completado de Tareas por Tutor", fontweight='bold', fontsize=14)
            plt.xlabel("Tutor", fontsize=12)
            plt.ylabel("Tasa de Completado (%)", fontsize=12)
            plt.xticks(range(len(tutor_productivity)), tutor_productivity.index, rotation=45)
            plt.grid(True, alpha=0.3)

            # Agregar valores en las barras
            for bar, value, count in zip(bars, tutor_productivity['completion_rate'].values, tutor_productivity['total_tasks'].values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                         f'{value*100:.1f}%\n({count})', ha='center', va='bottom', fontweight='bold')

            # Agregar líneas horizontales para umbrales
            mean_productivity = tutor_productivity['completion_rate'].mean() * 100
            plt.axhline(y=mean_productivity, color='red', linestyle='--', linewidth=2, label=f'Promedio: {mean_productivity:.1f}%')
            plt.axhline(y=70, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Umbral Alto (70%)')
            plt.axhline(y=50, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Umbral Medio (50%)')
            plt.legend()
            plt.tight_layout()
            plt.show()

        # RESUMEN ESTADÍSTICO
        print("\n📋 RESUMEN DE ANÁLISIS DE TAREAS:")
        print("=" * 50)

        if 'task_completed' in data.columns:
            total_tasks = len(data)
            completed_tasks = data['task_completed'].sum()
            print(f"\n📊 ESTADO GENERAL DE TAREAS:")
            print(f"   Total de tareas: {total_tasks}")
            print(f"   Tareas completadas: {completed_tasks}")
            print(f"   Tasa de completado: {completion_rate:.1f}%")

        if 'time_to_complete_hours' in data.columns and len(valid_times) > 0:
            print(f"\n⏰ ANÁLISIS DE TIEMPO:")
            print(f"   Tiempo promedio: {valid_times.mean():.1f} horas")
            print(f"   Tiempo mediano: {valid_times.median():.1f} horas")
            print(f"   Rango: {valid_times.min():.1f} - {valid_times.max():.1f} horas")

        if 'fecha_cita' in data.columns:
            print(f"\n DÍA CON MAYOR COMPLETADO: {day_names[completion_by_day.idxmax()]} ({completion_by_day.max():.1f}%)")
    
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


