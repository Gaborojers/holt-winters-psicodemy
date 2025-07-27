import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from pymongo.collection import Collection

class DataLoader:
    """Cargador de datos desde MongoDB"""
    
    def __init__(self, collection: Collection):
        self.collection = collection
    
    def load_ai_analysis_data(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sender_id: Optional[str] = None,
        recipient_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Carga datos de análisis de IA"""
        query = {}
        
        # Filtros de fecha
        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter['$gte'] = start_date
            if end_date:
                date_filter['$lte'] = end_date
            query['created_at'] = date_filter
        
        # Filtros de usuario
        if sender_id:
            query['sender_id'] = sender_id
        if recipient_id:
            query['recipient_id'] = recipient_id
        
        cursor = self.collection.find(query)
        data = list(cursor)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convertir ObjectId a string
        if '_id' in df.columns:
            df['_id'] = df['_id'].astype(str)
        
        # Convertir fechas
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        if 'updated_at' in df.columns:
            df['updated_at'] = pd.to_datetime(df['updated_at'])
        
        return df
    
    def load_appointments_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        id_alumno: Optional[str] = None,
        id_tutor: Optional[str] = None,
        estado_cita: Optional[str] = None
    ) -> pd.DataFrame:
        query = {}
        
        # Filtros de fecha
        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter['$gte'] = start_date
            if end_date:
                date_filter['$lte'] = end_date
            query['fecha_cita'] = date_filter
        
        # Filtros específicos según el modelo de API
        if id_alumno:
            query['id_alumno'] = id_alumno
        if id_tutor:
            query['id_tutor'] = id_tutor
        if estado_cita:
            query['estado_cita'] = estado_cita
        
        cursor = self.collection.find(query)
        data = list(cursor)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convertir ObjectId a string
        if '_id' in df.columns:
            df['_id'] = df['_id'].astype(str)
        
        # Convertir fechas según el modelo de API
        date_columns = ['created_at', 'updated_at', 'fecha_cita', 'deleted_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    
    def load_tasks_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        id_alumno: Optional[str] = None,
        id_tutor: Optional[str] = None,
        completed: Optional[bool] = None
    ) -> pd.DataFrame:
        query = {}
        
        # Filtros de fecha
        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter['$gte'] = start_date
            if end_date:
                date_filter['$lte'] = end_date
            query['fecha_cita'] = date_filter
        
        # Filtros específicos según el modelo de API
        if id_alumno:
            query['id_alumno'] = id_alumno
        if id_tutor:
            query['id_tutor'] = id_tutor
        
        # Solo citas que tengan checklist
        query['checklist'] = {'$exists': True, '$ne': []}
        
        cursor = self.collection.find(query)
        appointments_data = list(cursor)
        
        if not appointments_data:
            return pd.DataFrame()
        
        # Extraer tareas del checklist de cada cita
        tasks_data = []
        for appointment in appointments_data:
            appointment_id = str(appointment.get('_id', ''))
            id_tutor = appointment.get('id_tutor', '')
            id_alumno = appointment.get('id_alumno', '')
            fecha_cita = appointment.get('fecha_cita')
            estado_cita = appointment.get('estado_cita', '')
            created_at = appointment.get('created_at')
            updated_at = appointment.get('updated_at')
            reason = appointment.get('reason', '')
            
            checklist = appointment.get('checklist', [])
            
            for task in checklist:
                task_data = {
                    'appointment_id': appointment_id,
                    'id_tutor': id_tutor,
                    'id_alumno': id_alumno,
                    'fecha_cita': fecha_cita,
                    'estado_cita': estado_cita,
                    'appointment_created_at': created_at,
                    'appointment_updated_at': updated_at,
                    'reason': reason,
                    'task_description': task.get('description', ''),
                    'task_completed': task.get('completed', False)
                }
                tasks_data.append(task_data)
        
        if not tasks_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(tasks_data)
        
        # Aplicar filtro de completado si se especifica
        if completed is not None:
            df = df[df['task_completed'] == completed]
        
        # Convertir fechas
        date_columns = ['fecha_cita', 'appointment_created_at', 'appointment_updated_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    
    def get_time_series_data(
        self,
        group_by: str = 'date',
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        pipeline = []
        
        # Aplicar filtros
        if filters:
            pipeline.append({'$match': filters})
        
        # Agrupar por fecha
        if group_by == 'date':
            pipeline.extend([
                {
                    '$group': {
                        '_id': {
                            'date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$created_at'}}
                        },
                        'count': {'$sum': 1}
                    }
                },
                {'$sort': {'_id.date': 1}}
            ])
        elif group_by == 'hour':
            pipeline.extend([
                {
                    '$group': {
                        '_id': {
                            'date': {'$dateToString': {'format': '%Y-%m-%d %H:00:00', 'date': '$created_at'}}
                        },
                        'count': {'$sum': 1}
                    }
                },
                {'$sort': {'_id.date': 1}}
            ])
        
        cursor = self.collection.aggregate(pipeline)
        data = list(cursor)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['_id'].apply(lambda x: x['date']))
        df = df.drop('_id', axis=1).set_index('date')
        
        return df
