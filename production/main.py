import pandas as pd
import numpy as np
import os

# Importar modelos y métricas
from src.core.base_model import ARIMAModel
from src.core.metrics import TimeSeriesMetrics

# 1. Cargar los datos limpios
ruta_data = os.path.join(os.path.dirname(__file__), '..', 'notebooks')  # Ajusta si tus CSV están en otra carpeta

appointments = pd.read_csv(os.path.join(ruta_data, 'appointments_clean.csv'), parse_dates=['fecha_cita', 'created_at', 'updated_at'])
ai_clean = pd.read_csv(os.path.join(ruta_data, 'ai_clean.csv'), parse_dates=['created_at'])
tasks_clean = pd.read_csv(os.path.join(ruta_data, 'tasks_clean.csv'), parse_dates=['fecha_cita'])

# 2. Crear la serie temporal: cantidad de citas por día
appointments['fecha_cita'] = pd.to_datetime(appointments['fecha_cita'])
appointments_daily = appointments.groupby(appointments['fecha_cita'].dt.date).size()
appointments_daily.index = pd.to_datetime(appointments_daily.index)
appointments_daily = appointments_daily.sort_index()

# 3. Separar train/test (por ejemplo, 80% train, 20% test)
split_idx = int(len(appointments_daily) * 0.8)
train, test = appointments_daily.iloc[:split_idx], appointments_daily.iloc[split_idx:]

# 4. Entrenar modelo ARIMA
model = ARIMAModel(order=(1,1,1))
model.fit(train)

# 5. Predecir sobre el test
pred = model.predict(len(test))
pred.index = test.index  # Alinear índices

# 6. Calcular métricas de forecasting
metrics = TimeSeriesMetrics.calculate_forecast_accuracy(test, pred)

print("=== Forecasting de citas por día ===")
print("Métricas:")
for k, v in metrics.items():
    print(f"  {k}: {v:.3f}")

# 7. (Opcional) Graficar resultados si lo deseas
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test', color='orange')
    plt.plot(pred.index, pred, label='Predicción', color='green', linestyle='--')
    plt.legend()
    plt.title('Pronóstico de citas por día')
    plt.show()
except ImportError:
    print("matplotlib no está instalado. Instala con: pip install matplotlib")