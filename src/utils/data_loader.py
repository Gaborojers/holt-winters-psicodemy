import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pymongo.collection import Collection
from pymongo.cursor import Cursor

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
        student_id: Optional[str] = None,
        tutor_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> pd.DataFrame:
        """Carga datos de citas"""
        query = {}
        
        # Filtros de fecha
        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter['$gte'] = start_date
            if end_date:
                date_filter['$lte'] = end_date
            query['created_at'] = date_filter
        
        # Filtros específicos
        if student_id:
            query['student_id'] = student_id
        if tutor_id:
            query['tutor_id'] = tutor_id
        if status:
            query['status'] = status
        
        cursor = self.collection.find(query)
        data = list(cursor)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convertir ObjectId a string
        if '_id' in df.columns:
            df['_id'] = df['_id'].astype(str)
        
        # Convertir fechas
        date_columns = ['created_at', 'updated_at', 'appointment_date', 'confirmed_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    
    def load_tasks_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        student_id: Optional[str] = None,
        tutor_id: Optional[str] = None,
        completed: Optional[bool] = None
    ) -> pd.DataFrame:
        """Carga datos de tareas"""
        query = {}
        
        # Filtros de fecha
        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter['$gte'] = start_date
            if end_date:
                date_filter['$lte'] = end_date
            query['created_at'] = date_filter
        
        # Filtros específicos
        if student_id:
            query['student_id'] = student_id
        if tutor_id:
            query['tutor_id'] = tutor_id
        if completed is not None:
            query['completed'] = completed
        
        cursor = self.collection.find(query)
        data = list(cursor)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convertir ObjectId a string
        if '_id' in df.columns:
            df['_id'] = df['_id'].astype(str)
        
        # Convertir fechas
        date_columns = ['created_at', 'updated_at', 'due_date', 'completed_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    
    def get_time_series_data(
        self,
        group_by: str = 'date',
        metric: str = 'count',
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Genera series temporales de los datos"""
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
