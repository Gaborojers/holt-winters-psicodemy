import os
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from dotenv import load_dotenv

load_dotenv()

class DatabaseConfig:
    """Configuración de conexión a MongoDB"""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.getenv(
            'MONGODB_URI', 
            'mongodb://admin:secret123@3.223.236.109:27017'
        )
        self.client: Optional[MongoClient] = None
        self.database: Optional[Database] = None
        
    def connect(self, database_name: str = 'tutoring_platform') -> Database:
        """Establece conexión a MongoDB"""
        try:
            self.client = MongoClient(self.connection_string)
            self.database = self.client[database_name]
            # Verificar conexión
            self.client.admin.command('ping')
            print(f"✅ Conexión exitosa a MongoDB: {database_name}")
            return self.database
        except Exception as e:
            print(f"❌ Error conectando a MongoDB: {e}")
            raise
    
    def get_collection(self, collection_name: str) -> Collection:
        """Obtiene una colección específica"""
        if not self.database:
            raise ValueError("Debe conectarse primero a la base de datos")
        return self.database[collection_name]
    
    def close(self):
        """Cierra la conexión a MongoDB"""
        if self.client:
            self.client.close()
            print("🔌 Conexión a MongoDB cerrada")

# Instancia global de configuración
db_config = DatabaseConfig()
