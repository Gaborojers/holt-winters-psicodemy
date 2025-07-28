import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
    
    def clean_ai_analysis_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        # Extraer métricas del campo 'analysis' si existe
        if 'analysis' in df_clean.columns:
            df_clean['bullying'] = df_clean['analysis'].apply(lambda x: x.get('bullying') if isinstance(x, dict) else np.nan)
            df_clean['concern'] = df_clean['analysis'].apply(lambda x: x.get('concern') if isinstance(x, dict) else np.nan)
            df_clean['academic_constructive'] = df_clean['analysis'].apply(lambda x: x.get('academic_constructive') if isinstance(x, dict) else np.nan)
        # Agregar features temporales
        if 'created_at' in df_clean.columns:
            df_clean['hour'] = df_clean['created_at'].dt.hour
            df_clean['day_of_week'] = df_clean['created_at'].dt.dayofweek
            df_clean['month'] = df_clean['created_at'].dt.month
            df_clean['is_weekend'] = df_clean['day_of_week'] >= 5
        return df_clean
    
    def clean_appointments_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y prepara datos de citas"""
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # Codificar estados de citas
        status_mapping = {
            'pendiente': 0,
            'confirmada': 1,
            'completada': 2,
            'cancelada': 3,
            'no_asistio': 4
        }
        
        if 'status' in df_clean.columns:
            df_clean['status_code'] = df_clean['status'].map(status_mapping)        
        
        # Features temporales
        if 'created_at' in df_clean.columns:
            df_clean['days_to_appointment'] = (
                df_clean['fecha_cita'] - df_clean['created_at']
            ).dt.days
        
        # Codificar variables categóricas
        categorical_columns = ['student_id', 'tutor_id', 'subject']
        for col in categorical_columns:
            if col in df_clean.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df_clean[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                    df_clean[col].fillna('unknown')
                )
        
        return df_clean
    
    def clean_tasks_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y prepara datos de tareas"""
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # Convertir booleanos
        if 'task_completed' in df_clean.columns:
            df_clean['task_completed'] = df_clean['task_completed'].astype(int)
        
        # Calcular tiempo de completado
        if 'appointment_created_at' in df_clean.columns and 'appointment_updated_at' in df_clean.columns:
            df_clean['time_to_complete_hours'] = (
                df_clean['appointment_updated_at'] - df_clean['appointment_created_at']
            ).dt.total_seconds() / 3600
        
        # Features de tareas
        if 'description' in df_clean.columns:
            df_clean['task_length'] = df_clean['description'].str.len()
        
        # Codificar variables categóricas
        categorical_columns = ['id_alumno', 'id_tutor', 'task_description']
        for col in categorical_columns:
            if col in df_clean.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df_clean[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                    df_clean[col].fillna('unknown')
                )
        
        return df_clean
    
    def create_time_series_features(self, df: pd.DataFrame, date_column: str = 'fecha_cita') -> pd.DataFrame:
        if df.empty or date_column not in df.columns:
            return df
        
        df_ts = df.copy()
        
        # Asegurar que la columna de fecha es datetime
        df_ts[date_column] = pd.to_datetime(df_ts[date_column])
        
        # Features temporales básicas
        df_ts['year'] = df_ts[date_column].dt.year
        df_ts['month'] = df_ts[date_column].dt.month
        df_ts['day'] = df_ts[date_column].dt.day
        df_ts['hour'] = df_ts[date_column].dt.hour
        df_ts['day_of_week'] = df_ts[date_column].dt.dayofweek
        df_ts['day_of_year'] = df_ts[date_column].dt.dayofyear
        df_ts['week_of_year'] = df_ts[date_column].dt.isocalendar().week
        
        # Features cíclicas
        df_ts['month_sin'] = np.sin(2 * np.pi * df_ts['month'] / 12)
        df_ts['month_cos'] = np.cos(2 * np.pi * df_ts['month'] / 12)
        df_ts['day_of_week_sin'] = np.sin(2 * np.pi * df_ts['day_of_week'] / 7)
        df_ts['day_of_week_cos'] = np.cos(2 * np.pi * df_ts['day_of_week'] / 7)
        df_ts['hour_sin'] = np.sin(2 * np.pi * df_ts['hour'] / 24)
        df_ts['hour_cos'] = np.cos(2 * np.pi * df_ts['hour'] / 24)
        
        # Indicadores de tiempo
        df_ts['is_weekend'] = df_ts['day_of_week'].isin([5, 6]).astype(int)
        df_ts['is_business_hour'] = df_ts['hour'].between(9, 17).astype(int)
        df_ts['is_morning'] = df_ts['hour'].between(6, 12).astype(int)
        df_ts['is_afternoon'] = df_ts['hour'].between(12, 18).astype(int)
        df_ts['is_evening'] = df_ts['hour'].between(18, 22).astype(int)
        df_ts['is_night'] = ((df_ts['hour'] >= 22) | (df_ts['hour'] < 6)).astype(int)
        
        return df_ts
    
    def aggregate_by_time_period(
        self, 
        df: pd.DataFrame, 
        date_column: str = 'fecha_cita',
        period: str = 'D',
        agg_functions: Dict[str, List[str]] = None
    ) -> pd.DataFrame:
        if df.empty or date_column not in df.columns:
            return df
        
        # Funciones de agregación por defecto
        if agg_functions is None:
            agg_functions = {
                'count': ['count'],
                'bullying': ['sum', 'mean'],
                'concern': ['sum', 'mean'],
                'academic_constructive': ['sum', 'mean'],
                'message_length': ['mean', 'std'],
                'task_completed': ['sum', 'mean'],
                'status_code': ['mean']
            }
        
        # Filtrar columnas que existen en el DataFrame
        available_columns = [col for col in agg_functions.keys() if col in df.columns]
        agg_dict = {}
        
        for col in available_columns:
            for func in agg_functions[col]:
                if func == 'count':
                    agg_dict[col] = 'count'
                else:
                    agg_dict[f'{col}_{func}'] = func
        
        # Agregar por período
        df_agg = df.set_index(date_column).resample(period).agg(agg_dict)
        
        return df_agg
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'forward') -> pd.DataFrame:
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        if strategy == 'forward':
            df_clean = df_clean.fillna(method='ffill')
        elif strategy == 'backward':
            df_clean = df_clean.fillna(method='bfill')
        elif strategy == 'interpolate':
            df_clean = df_clean.interpolate(method='time')
        elif strategy == 'mean':
            df_clean = df_clean.fillna(df_clean.mean())
        elif strategy == 'zero':
            df_clean = df_clean.fillna(0)
        
        return df_clean
    
    def normalize_features(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        if df.empty:
            return df
        
        df_norm = df.copy()
        
        if columns is None:
            # Seleccionar columnas numéricas automáticamente
            numeric_columns = df_norm.select_dtypes(include=[np.number]).columns
            columns = [col for col in numeric_columns if col not in ['year', 'month', 'day', 'hour', 'day_of_week']]
        
        if columns:
            df_norm[columns] = self.scaler.fit_transform(df_norm[columns])
        
        return df_norm
