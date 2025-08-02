import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class AnomalyDetector:
    """Clase para detección de anomalías en datos educativos."""
    
    def __init__(self):
        self.pre_anomalies = {}
        self.post_anomalies = {}
    
    def detect_preprocessing_anomalies(self, df: pd.DataFrame, dataset_type: str) -> Dict:
        """Detecta anomalías ANTES del preprocesamiento usando técnicas avanzadas."""
        anomalies = {
            'missing_values': {},
            'duplicates': {},
            'outliers': {},
            'advanced_anomalies': {},
            'domain_specific': {},
            'summary': {}
        }
        
        # Valores faltantes
        missing_data = df.isnull().sum()
        anomalies['missing_values'] = {
            'counts': missing_data.to_dict(),
            'total_missing': missing_data.sum(),
            'total_percentage': (missing_data.sum() / (len(df) * len(df.columns))) * 100
        }
        
        # Duplicados (optimizado para datasets grandes)
        try:
            # Excluir columnas conocidas que contienen diccionarios
            exclude_columns = ['analysis', 'checklist']
            columns_for_duplicates = [col for col in df.columns if col not in exclude_columns]
            
            # Para datasets grandes, usar solo las primeras columnas numéricas y de texto
            if len(df) > 1000:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                text_cols = df.select_dtypes(include=['object']).columns.tolist()
                # Excluir columnas que podrían contener diccionarios
                text_cols = [col for col in text_cols if col not in exclude_columns]
                
                # Usar máximo 5 columnas para evitar problemas de memoria
                columns_for_duplicates = (numeric_cols[:3] + text_cols[:2])[:5]
            
            if columns_for_duplicates:
                duplicates = df[columns_for_duplicates].duplicated().sum()
            else:
                duplicates = 0
        except Exception as e:
            # Si hay cualquier error, establecer duplicados en 0
            duplicates = 0
        anomalies['duplicates'] = {
            'count': duplicates,
            'percentage': (duplicates / len(df)) * 100
        }
        
        # Outliers por columnas numéricas (método IQR tradicional)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outliers_info = {}
        
        for col in numeric_columns:
            if col in df.columns and df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100
                }
        
        anomalies['outliers'] = outliers_info
        
        # Anomalías avanzadas usando Isolation Forest y LOF
        if len(numeric_columns) > 0:
            advanced_anomalies = self._detect_advanced_anomalies(df, numeric_columns)
            anomalies['advanced_anomalies'] = advanced_anomalies
        
        # Análisis específico por dominio de datos educativos
        domain_anomalies = self._detect_domain_specific_anomalies(df, dataset_type)
        anomalies['domain_specific'] = domain_anomalies
        
        # Resumen
        anomalies['summary'] = {
            'total_records': len(df)
        }
        
        # Calcular score de calidad después de crear el resumen
        anomalies['summary']['data_quality_score'] = self._calculate_quality_score(anomalies)
        
        self.pre_anomalies[dataset_type] = anomalies
        return anomalies
    
    def detect_preprocessing_anomalies_chunked(self, df: pd.DataFrame, dataset_type: str, 
                                             chunk_size: int = 1000) -> Dict:
        """
        Detecta anomalías procesando el dataset por chunks para optimizar memoria.
        Ideal para datasets grandes (>10,000 registros).
        """
        if len(df) <= chunk_size:
            # Para datasets pequeños, usar el método original
            return self.detect_preprocessing_anomalies(df, dataset_type)
        
        print(f"🔄 Procesando dataset de {len(df)} registros en chunks de {chunk_size}...")
        
        # Inicializar contadores globales
        total_missing = 0
        total_duplicates = 0
        all_outliers = {}
        all_advanced_anomalies = []
        all_domain_anomalies = []
        
        # Procesar por chunks
        num_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx].copy()
            
            print(f"   Procesando chunk {i+1}/{num_chunks} (registros {start_idx+1}-{end_idx})")
            
            # Analizar chunk
            chunk_anomalies = self.detect_preprocessing_anomalies(chunk, dataset_type)
            
            # Acumular resultados
            total_missing += chunk_anomalies['missing_values']['total_missing']
            total_duplicates += chunk_anomalies['duplicates']['count']
            
            # Acumular outliers por columna
            for col, info in chunk_anomalies['outliers'].items():
                if col not in all_outliers:
                    all_outliers[col] = {'count': 0, 'percentage': 0}
                all_outliers[col]['count'] += info['count']
            
            # Acumular anomalías avanzadas
            if 'advanced_anomalies' in chunk_anomalies:
                all_advanced_anomalies.append(chunk_anomalies['advanced_anomalies'])
            
            # Acumular anomalías específicas del dominio
            if 'domain_specific' in chunk_anomalies:
                all_domain_anomalies.append(chunk_anomalies['domain_specific'])
        
        # Consolidar resultados
        consolidated_anomalies = {
            'missing_values': {
                'counts': df.isnull().sum().to_dict(),
                'total_missing': total_missing,
                'total_percentage': (total_missing / (len(df) * len(df.columns))) * 100
            },
            'duplicates': {
                'count': total_duplicates,
                'percentage': (total_duplicates / len(df)) * 100
            },
            'outliers': all_outliers,
            'advanced_anomalies': self._consolidate_advanced_anomalies(all_advanced_anomalies, len(df)),
            'domain_specific': self._consolidate_domain_anomalies(all_domain_anomalies, len(df)),
            'summary': {
                'total_records': len(df),
                'chunks_processed': num_chunks,
                'chunk_size': chunk_size
            }
        }
        
        # Calcular score de calidad después de crear el resumen completo
        consolidated_anomalies['summary']['data_quality_score'] = self._calculate_quality_score(consolidated_anomalies)
        
        self.pre_anomalies[f"{dataset_type}_chunked"] = consolidated_anomalies
        return consolidated_anomalies
    
    def _detect_advanced_anomalies(self, df: pd.DataFrame, numeric_columns: List[str]) -> Dict:
        """Detecta anomalías usando técnicas avanzadas de machine learning."""
        if len(numeric_columns) < 2:
            return {'error': 'Se requieren al menos 2 columnas numéricas'}
        
        # Preparar datos
        X = df[numeric_columns].fillna(df[numeric_columns].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {}
        
        # Isolation Forest
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_predictions = iso_forest.fit_predict(X_scaled)
            iso_anomalies = iso_predictions == -1
            
            results['isolation_forest'] = {
                'anomalies_detected': iso_anomalies.sum(),
                'anomaly_percentage': (iso_anomalies.sum() / len(df)) * 100,
                'anomaly_indices': df[iso_anomalies].index.tolist()
            }
        except Exception as e:
            results['isolation_forest'] = {'error': str(e)}
        
        # Local Outlier Factor
        try:
            lof = LocalOutlierFactor(contamination=0.1, n_neighbors=min(20, len(df)//4))
            lof_predictions = lof.fit_predict(X_scaled)
            lof_anomalies = lof_predictions == -1
            
            results['local_outlier_factor'] = {
                'anomalies_detected': lof_anomalies.sum(),
                'anomaly_percentage': (lof_anomalies.sum() / len(df)) * 100,
                'anomaly_indices': df[lof_anomalies].index.tolist()
            }
        except Exception as e:
            results['local_outlier_factor'] = {'error': str(e)}
        
        return results
    
    def _detect_domain_specific_anomalies(self, df: pd.DataFrame, dataset_type: str) -> Dict:
        """Detecta anomalías específicas del dominio educativo."""
        domain_anomalies = {}
        
        if dataset_type == 'ai_data':
            domain_anomalies = self._analyze_ai_data_anomalies(df)
        elif dataset_type == 'appointments_data':
            domain_anomalies = self._analyze_appointments_anomalies(df)
        elif dataset_type == 'tasks_data':
            domain_anomalies = self._analyze_tasks_anomalies(df)
        
        return domain_anomalies
    
    def _analyze_ai_data_anomalies(self, df: pd.DataFrame) -> Dict:
        """Analiza anomalías específicas en datos de IA considerando el preprocesamiento."""
        anomalies = {}
        
        # 1. Análisis de scores de IA (antes del preprocesamiento)
        if 'analysis' in df.columns:
            try:
                # Extraer scores de bullying, concern y academic
                bullying_scores = []
                concern_scores = []
                academic_scores = []
                
                for analysis in df['analysis']:
                    if isinstance(analysis, dict):
                        bullying_scores.append(analysis.get('bullying', 0))
                        concern_scores.append(analysis.get('concern', 0))
                        academic_scores.append(analysis.get('academic_constructive', 0))
                
                if bullying_scores:
                    # Detectar scores extremos
                    bullying_array = np.array(bullying_scores)
                    concern_array = np.array(concern_scores)
                    academic_array = np.array(academic_scores)
                    
                    # Scores fuera del rango [0, 1]
                    invalid_bullying = np.sum((bullying_array < 0) | (bullying_array > 1))
                    invalid_concern = np.sum((concern_array < 0) | (concern_array > 1))
                    invalid_academic = np.sum((academic_array < 0) | (academic_array > 1))
                    
                    anomalies['invalid_scores'] = {
                        'bullying_out_of_range': invalid_bullying,
                        'concern_out_of_range': invalid_concern,
                        'academic_out_of_range': invalid_academic
                    }
                    
                    # Detectar patrones sospechosos (todos los scores iguales)
                    same_scores = np.sum(
                        (bullying_array == concern_array) & 
                        (concern_array == academic_array)
                    )
                    anomalies['suspicious_patterns'] = {
                        'identical_scores': same_scores
                    }
                    
                    # Detectar scores constantes (posible error en el modelo de IA)
                    constant_bullying = np.all(bullying_array == bullying_array[0]) if len(bullying_array) > 1 else False
                    constant_concern = np.all(concern_array == concern_array[0]) if len(concern_array) > 1 else False
                    constant_academic = np.all(academic_array == academic_array[0]) if len(academic_array) > 1 else False
                    
                    anomalies['constant_scores'] = {
                        'bullying_constant': constant_bullying,
                        'concern_constant': constant_concern,
                        'academic_constant': constant_academic
                    }
            except Exception as e:
                anomalies['error'] = f"Error analizando datos de IA: {str(e)}"
        
        # 2. Análisis de longitud de mensajes
        if 'message_length' in df.columns:
            message_lengths = df['message_length'].dropna()
            if len(message_lengths) > 0:
                # Mensajes muy cortos o muy largos
                short_messages = (message_lengths < 5).sum()
                long_messages = (message_lengths > 1000).sum()
                
                anomalies['message_length'] = {
                    'very_short_messages': short_messages,
                    'very_long_messages': long_messages
                }
        
        # 3. Análisis de features temporales (después del preprocesamiento)
        temporal_features = ['hour', 'day_of_week', 'month', 'is_weekend']
        temporal_anomalies = {}
        
        for feature in temporal_features:
            if feature in df.columns:
                if feature == 'hour':
                    # Horas fuera del rango [0, 23]
                    invalid_hours = ((df[feature] < 0) | (df[feature] > 23)).sum()
                    temporal_anomalies['invalid_hours'] = invalid_hours
                
                elif feature == 'day_of_week':
                    # Días fuera del rango [0, 6]
                    invalid_days = ((df[feature] < 0) | (df[feature] > 6)).sum()
                    temporal_anomalies['invalid_days'] = invalid_days
                
                elif feature == 'month':
                    # Meses fuera del rango [1, 12]
                    invalid_months = ((df[feature] < 1) | (df[feature] > 12)).sum()
                    temporal_anomalies['invalid_months'] = invalid_months
        
        if temporal_anomalies:
            anomalies['temporal_features'] = temporal_anomalies
        
        # 4. Análisis de estructura de datos
        if 'analysis' in df.columns:
            # Verificar que todos los análisis tengan la estructura esperada
            expected_keys = ['bullying', 'concern', 'academic_constructive']
            malformed_analysis = 0
            
            for analysis in df['analysis']:
                if not isinstance(analysis, dict) or not all(key in analysis for key in expected_keys):
                    malformed_analysis += 1
            
            anomalies['data_structure'] = {
                'malformed_analysis_count': malformed_analysis
            }
        
        return anomalies
    
    def _analyze_appointments_anomalies(self, df: pd.DataFrame) -> Dict:
        """Analiza anomalías específicas en datos de citas considerando el preprocesamiento."""
        anomalies = {}
        
        # 1. Estados de cita inválidos (antes del preprocesamiento)
        if 'estado_cita' in df.columns:
            valid_states = ['pendiente', 'confirmada', 'completada', 'cancelada', 'no_asistio']
            invalid_states = ~df['estado_cita'].isin(valid_states)
            anomalies['invalid_appointment_states'] = {
                'count': invalid_states.sum(),
                'invalid_values': df[invalid_states]['estado_cita'].unique().tolist()
            }
        
        # 2. Análisis de status_code (después del preprocesamiento)
        if 'status_code' in df.columns:
            # Verificar que status_code esté en el rango [0, 4]
            invalid_status_codes = ((df['status_code'] < 0) | (df['status_code'] > 4)).sum()
            anomalies['invalid_status_codes'] = {
                'count': invalid_status_codes,
                'percentage': (invalid_status_codes / len(df)) * 100
            }
        
        # 3. Fechas de cita en el futuro muy lejano o pasado muy lejano
        if 'fecha_cita' in df.columns:
            try:
                df['fecha_cita'] = pd.to_datetime(df['fecha_cita'])
                current_date = pd.Timestamp.now()
                
                # Citas en el futuro muy lejano (> 1 año)
                future_citas = df['fecha_cita'] > (current_date + pd.DateOffset(years=1))
                
                # Citas en el pasado muy lejano (> 10 años)
                past_citas = df['fecha_cita'] < (current_date - pd.DateOffset(years=10))
                
                anomalies['date_anomalies'] = {
                    'far_future_appointments': future_citas.sum(),
                    'far_past_appointments': past_citas.sum()
                }
            except Exception as e:
                anomalies['date_error'] = str(e)
        
        # 4. Análisis de days_to_appointment (después del preprocesamiento)
        if 'days_to_appointment' in df.columns:
            days_to_appointment = df['days_to_appointment'].dropna()
            if len(days_to_appointment) > 0:
                # Días negativos (citas creadas después de la fecha de cita)
                negative_days = (days_to_appointment < 0).sum()
                
                # Días muy lejanos (> 365 días)
                very_far_days = (days_to_appointment > 365).sum()
                
                anomalies['appointment_timing'] = {
                    'negative_days': negative_days,
                    'very_far_appointments': very_far_days
                }
        
        # 5. Análisis de variables codificadas (después del preprocesamiento)
        encoded_columns = [col for col in df.columns if col.endswith('_encoded')]
        encoding_anomalies = {}
        
        for col in encoded_columns:
            if col in df.columns:
                # Verificar valores negativos en columnas codificadas
                negative_values = (df[col] < 0).sum()
                encoding_anomalies[f'{col}_negative'] = negative_values
        
        if encoding_anomalies:
            anomalies['encoding_issues'] = encoding_anomalies
        
        # 6. Análisis de checklist (estructura de datos)
        if 'checklist' in df.columns:
            malformed_checklists = 0
            empty_checklists = 0
            
            for checklist in df['checklist']:
                if not isinstance(checklist, list):
                    malformed_checklists += 1
                elif len(checklist) == 0:
                    empty_checklists += 1
                else:
                    # Verificar estructura de cada tarea en el checklist
                    for task in checklist:
                        if not isinstance(task, dict) or 'description' not in task or 'completed' not in task:
                            malformed_checklists += 1
                            break
            
            anomalies['checklist_structure'] = {
                'malformed_checklists': malformed_checklists,
                'empty_checklists': empty_checklists
            }
        
        return anomalies
    
    def _analyze_tasks_anomalies(self, df: pd.DataFrame) -> Dict:
        """Analiza anomalías específicas en datos de tareas considerando el preprocesamiento."""
        anomalies = {}
        
        # 1. Tareas sin descripción (antes del preprocesamiento)
        if 'task_description' in df.columns:
            empty_descriptions = df['task_description'].isna() | (df['task_description'] == '')
            anomalies['empty_task_descriptions'] = {
                'count': empty_descriptions.sum(),
                'percentage': (empty_descriptions.sum() / len(df)) * 100
            }
        
        # 2. Análisis de task_completed (después del preprocesamiento - convertido a int)
        if 'task_completed' in df.columns:
            # Verificar que task_completed esté en [0, 1] después de la conversión
            invalid_completion = ((df['task_completed'] < 0) | (df['task_completed'] > 1)).sum()
            anomalies['invalid_task_completion'] = {
                'count': invalid_completion,
                'percentage': (invalid_completion / len(df)) * 100
            }
        
        # 3. Tareas completadas pero con estado de cita cancelada
        if 'task_completed' in df.columns and 'estado_cita' in df.columns:
            completed_tasks_cancelled = (
                (df['task_completed'] == 1) & 
                (df['estado_cita'] == 'cancelada')
            )
            anomalies['completed_tasks_cancelled_appointments'] = {
                'count': completed_tasks_cancelled.sum()
            }
        
        # 4. Análisis de time_to_complete_hours (después del preprocesamiento)
        if 'time_to_complete_hours' in df.columns:
            time_to_complete = df['time_to_complete_hours'].dropna()
            if len(time_to_complete) > 0:
                # Tiempos negativos
                negative_times = (time_to_complete < 0).sum()
                
                # Tiempos muy largos (> 30 días)
                very_long_times = (time_to_complete > 720).sum()
                
                # Tiempos sospechosamente cortos (< 1 minuto para tareas completadas)
                suspicious_short_times = (
                    (time_to_complete < 0.0167) & 
                    (df['task_completed'] == 1)
                ).sum()
                
                anomalies['completion_timing'] = {
                    'negative_times': negative_times,
                    'very_long_times': very_long_times,
                    'suspicious_short_times': suspicious_short_times
                }
        
        # 5. Análisis de task_length (después del preprocesamiento)
        if 'task_length' in df.columns:
            task_lengths = df['task_length'].dropna()
            if len(task_lengths) > 0:
                # Descripciones muy cortas
                very_short_descriptions = (task_lengths < 3).sum()
                
                # Descripciones muy largas
                very_long_descriptions = (task_lengths > 500).sum()
                
                anomalies['task_description_length'] = {
                    'very_short_descriptions': very_short_descriptions,
                    'very_long_descriptions': very_long_descriptions
                }
        
        # 6. Análisis de variables codificadas (después del preprocesamiento)
        encoded_columns = [col for col in df.columns if col.endswith('_encoded')]
        encoding_anomalies = {}
        
        for col in encoded_columns:
            if col in df.columns:
                # Verificar valores negativos en columnas codificadas
                negative_values = (df[col] < 0).sum()
                encoding_anomalies[f'{col}_negative'] = negative_values
        
        if encoding_anomalies:
            anomalies['encoding_issues'] = encoding_anomalies
        
        # 7. Análisis de consistencia entre appointment_id y datos de cita
        if 'appointment_id' in df.columns and 'fecha_cita' in df.columns:
            # Verificar que las fechas de tareas coincidan con las fechas de citas
            # (esto requeriría datos de citas para comparación completa)
            pass
        
        return anomalies
    
    def detect_postprocessing_anomalies(self, df: pd.DataFrame, dataset_type: str, 
                                      method: str = 'isolation_forest') -> Dict:
        """Detecta anomalías DESPUÉS del preprocesamiento."""
        if df.empty:
            return {'error': 'DataFrame vacío'}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            return {'error': 'No hay columnas numéricas para análisis'}
        
        X = df[numeric_columns].fillna(0)
        
        # Aplicar algoritmo de detección
        if method == 'isolation_forest':
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(X)
            anomaly_scores = model.predict(X)
            anomalies_detected = anomaly_scores == -1
        
        # Estadísticas
        anomaly_stats = {
            'total_anomalies': anomalies_detected.sum(),
            'anomaly_percentage': (anomalies_detected.sum() / len(df)) * 100,
            'method_used': method,
            'columns_analyzed': list(numeric_columns)
        }
        
        self.post_anomalies[dataset_type] = {
            'stats': anomaly_stats,
            'anomaly_indices': df[anomalies_detected].index.tolist()
        }
        
        return self.post_anomalies[dataset_type]
    
    def _consolidate_advanced_anomalies(self, chunk_anomalies: List[Dict], total_records: int) -> Dict:
        """Consolida anomalías avanzadas de múltiples chunks."""
        if not chunk_anomalies:
            return {}
        
        consolidated = {}
        
        # Consolidar Isolation Forest
        iso_anomalies = []
        for chunk in chunk_anomalies:
            if 'isolation_forest' in chunk and 'anomalies_detected' in chunk['isolation_forest']:
                iso_anomalies.append(chunk['isolation_forest']['anomalies_detected'])
        
        if iso_anomalies:
            total_iso_anomalies = sum(iso_anomalies)
            consolidated['isolation_forest'] = {
                'anomalies_detected': total_iso_anomalies,
                'anomaly_percentage': (total_iso_anomalies / total_records) * 100
            }
        
        # Consolidar Local Outlier Factor
        lof_anomalies = []
        for chunk in chunk_anomalies:
            if 'local_outlier_factor' in chunk and 'anomalies_detected' in chunk['local_outlier_factor']:
                lof_anomalies.append(chunk['local_outlier_factor']['anomalies_detected'])
        
        if lof_anomalies:
            total_lof_anomalies = sum(lof_anomalies)
            consolidated['local_outlier_factor'] = {
                'anomalies_detected': total_lof_anomalies,
                'anomaly_percentage': (total_lof_anomalies / total_records) * 100
            }
        
        return consolidated
    
    def _consolidate_domain_anomalies(self, chunk_anomalies: List[Dict], total_records: int) -> Dict:
        """Consolida anomalías específicas del dominio de múltiples chunks."""
        if not chunk_anomalies:
            return {}
        
        consolidated = {}
        
        # Consolidar anomalías de IA
        if any('invalid_scores' in chunk for chunk in chunk_anomalies):
            total_invalid = {
                'bullying_out_of_range': 0,
                'concern_out_of_range': 0,
                'academic_out_of_range': 0
            }
            
            for chunk in chunk_anomalies:
                if 'invalid_scores' in chunk:
                    for key in total_invalid:
                        total_invalid[key] += chunk['invalid_scores'].get(key, 0)
            
            consolidated['invalid_scores'] = total_invalid
        
        # Consolidar patrones sospechosos
        total_identical_scores = 0
        for chunk in chunk_anomalies:
            if 'suspicious_patterns' in chunk:
                total_identical_scores += chunk['suspicious_patterns'].get('identical_scores', 0)
        
        if total_identical_scores > 0:
            consolidated['suspicious_patterns'] = {'identical_scores': total_identical_scores}
        
        # Consolidar scores constantes
        constant_scores = {
            'bullying_constant': False,
            'concern_constant': False,
            'academic_constant': False
        }
        
        for chunk in chunk_anomalies:
            if 'constant_scores' in chunk:
                for key in constant_scores:
                    constant_scores[key] = constant_scores[key] or chunk['constant_scores'].get(key, False)
        
        if any(constant_scores.values()):
            consolidated['constant_scores'] = constant_scores
        
        return consolidated
    
    def _calculate_quality_score(self, anomalies: Dict) -> float:
        """Calcula score de calidad de datos (0-100) considerando todas las anomalías detectadas"""
        score = 100.0
        
        # Obtener total de registros de forma segura
        total_records = anomalies.get('summary', {}).get('total_records', 1)
        if total_records == 0:
            total_records = 1  # Evitar división por cero
        
        # Penalización por valores faltantes (peso: 0.5)
        missing_percentage = anomalies.get('missing_values', {}).get('total_percentage', 0)
        score -= missing_percentage * 0.5
        
        # Penalización por duplicados (peso: 0.3)
        duplicate_percentage = anomalies.get('duplicates', {}).get('percentage', 0)
        score -= duplicate_percentage * 0.3
        
        # Penalización por outliers tradicionales (peso: 0.2)
        outliers = anomalies.get('outliers', {})
        outlier_percentage = sum(
            info.get('percentage', 0) for info in outliers.values()
        )
        score -= outlier_percentage * 0.2
        
        # Penalización por anomalías avanzadas (peso: 0.4)
        advanced_anomalies = anomalies.get('advanced_anomalies', {})
        
        # Isolation Forest
        if 'isolation_forest' in advanced_anomalies:
            iso_percentage = advanced_anomalies['isolation_forest'].get('anomaly_percentage', 0)
            score -= iso_percentage * 0.2
        
        # Local Outlier Factor
        if 'local_outlier_factor' in advanced_anomalies:
            lof_percentage = advanced_anomalies['local_outlier_factor'].get('anomaly_percentage', 0)
            score -= lof_percentage * 0.2
        
        # Penalización por anomalías específicas del dominio (peso: 0.6)
        domain_anomalies = anomalies.get('domain_specific', {})
        
        # Anomalías en datos de IA
        if 'invalid_scores' in domain_anomalies:
            total_invalid = sum(domain_anomalies['invalid_scores'].values())
            score -= (total_invalid / total_records) * 100 * 0.3
        
        if 'suspicious_patterns' in domain_anomalies:
            identical_scores = domain_anomalies['suspicious_patterns'].get('identical_scores', 0)
            score -= (identical_scores / total_records) * 100 * 0.2
        
        if 'constant_scores' in domain_anomalies:
            constant_count = sum(domain_anomalies['constant_scores'].values())
            if constant_count > 0:
                score -= 10.0  # Penalización fija por scores constantes
        
        # Anomalías en datos de citas
        if 'invalid_appointment_states' in domain_anomalies:
            invalid_states = domain_anomalies['invalid_appointment_states'].get('count', 0)
            score -= (invalid_states / total_records) * 100 * 0.2
        
        if 'appointment_timing' in domain_anomalies:
            negative_days = domain_anomalies['appointment_timing'].get('negative_days', 0)
            score -= (negative_days / total_records) * 100 * 0.1
        
        # Anomalías en datos de tareas
        if 'empty_task_descriptions' in domain_anomalies:
            empty_descriptions = domain_anomalies['empty_task_descriptions'].get('percentage', 0)
            score -= empty_descriptions * 0.2
        
        if 'completion_timing' in domain_anomalies:
            suspicious_times = domain_anomalies['completion_timing'].get('suspicious_short_times', 0)
            score -= (suspicious_times / total_records) * 100 * 0.1
        
        return max(0, score)