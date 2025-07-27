import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class BaseTimeSeriesModel(ABC):
    """Clase base para modelos de series temporales"""
    
    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.is_fitted = False
        self.model = None
        self.fitted_values = None
        self.residuals = None
        self.metrics = {}
    
    @abstractmethod
    def fit(self, data: pd.Series) -> 'BaseTimeSeriesModel':
        """Ajusta el modelo a los datos"""
        pass
    
    @abstractmethod
    def predict(self, steps: int) -> pd.Series:
        """Realiza predicciones"""
        pass
    
    @abstractmethod
    def forecast(self, data: pd.Series, steps: int) -> pd.Series:
        """Realiza pronósticos"""
        pass
    
    def evaluate(self, actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        """Evalúa el modelo con métricas estándar"""
        if len(actual) != len(predicted):
            raise ValueError("Las series actual y predicha deben tener la misma longitud")
        
        # Calcular métricas
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # R² score
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
        
        return self.metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtiene información del modelo"""
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'metrics': self.metrics,
            'model_params': getattr(self.model, 'params', None) if self.model else None
        }

class HoltWintersModel(BaseTimeSeriesModel):
    """Modelo Holt-Winters para series temporales"""
    
    def __init__(self, seasonal_periods: int = 7, trend: str = 'add', seasonal: str = 'add'):
        super().__init__("Holt-Winters")
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.alpha = None
        self.beta = None
        self.gamma = None
    
    def fit(self, data: pd.Series) -> 'HoltWintersModel':
        """Ajusta el modelo Holt-Winters"""
        try:
            from holtwinters import ExponentialSmoothing
            
            # Crear y ajustar el modelo
            self.model = ExponentialSmoothing(
                data,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods
            ).fit()
            
            # Obtener parámetros
            self.alpha = self.model.params['smoothing_level']
            self.beta = self.model.params.get('smoothing_slope', None)
            self.gamma = self.model.params.get('smoothing_seasonal', None)
            
            # Obtener valores ajustados
            self.fitted_values = self.model.fittedvalues
            self.residuals = data - self.fitted_values
            
            self.is_fitted = True
            
        except ImportError:
            raise ImportError("La librería holtwinters no está instalada. Instálala con: pip install holtwinters")
        
        return self
    
    def predict(self, steps: int) -> pd.Series:
        """Realiza predicciones"""
        if not self.is_fitted:
            raise ValueError("El modelo debe estar ajustado antes de hacer predicciones")
        
        forecast = self.model.forecast(steps)
        return forecast
    
    def forecast(self, data: pd.Series, steps: int) -> pd.Series:
        """Realiza pronósticos ajustando el modelo a nuevos datos"""
        self.fit(data)
        return self.predict(steps)

class ARIMAModel(BaseTimeSeriesModel):
    """Modelo ARIMA para series temporales"""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), seasonal_order: Tuple[int, int, int, int] = None):
        super().__init__("ARIMA")
        self.order = order
        self.seasonal_order = seasonal_order
        self.aic = None
        self.bic = None
    
    def fit(self, data: pd.Series) -> 'ARIMAModel':
        """Ajusta el modelo ARIMA"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            # Crear y ajustar el modelo
            if self.seasonal_order:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                self.model = SARIMAX(data, order=self.order, seasonal_order=self.seasonal_order)
            else:
                self.model = ARIMA(data, order=self.order)
            
            self.model = self.model.fit()
            
            # Obtener métricas de información
            self.aic = self.model.aic
            self.bic = self.model.bic
            
            # Obtener valores ajustados
            self.fitted_values = self.model.fittedvalues
            self.residuals = self.model.resid
            
            self.is_fitted = True
            
        except ImportError:
            raise ImportError("statsmodels no está instalado. Instálalo con: pip install statsmodels")
        
        return self
    
    def predict(self, steps: int) -> pd.Series:
        """Realiza predicciones"""
        if not self.is_fitted:
            raise ValueError("El modelo debe estar ajustado antes de hacer predicciones")
        
        forecast = self.model.forecast(steps)
        return forecast
    
    def forecast(self, data: pd.Series, steps: int) -> pd.Series:
        """Realiza pronósticos ajustando el modelo a nuevos datos"""
        self.fit(data)
        return self.predict(steps)
    
    def auto_arima(self, data: pd.Series, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> 'ARIMAModel':
        """Encuentra automáticamente los mejores parámetros ARIMA"""
        try:
            from pmdarima import auto_arima
            
            # Encontrar mejores parámetros
            best_model = auto_arima(
                data,
                start_p=0, start_q=0,
                max_p=max_p, max_d=max_d, max_q=max_q,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False
            )
            
            # Actualizar parámetros
            self.order = best_model.order
            self.seasonal_order = best_model.seasonal_order
            
            # Ajustar el modelo con los mejores parámetros
            return self.fit(data)
            
        except ImportError:
            raise ImportError("pmdarima no está instalado. Instálalo con: pip install pmdarima")

class EnsembleModel(BaseTimeSeriesModel):
    """Modelo ensemble que combina múltiples modelos"""
    
    def __init__(self, models: List[BaseTimeSeriesModel], weights: Optional[List[float]] = None):
        super().__init__("Ensemble")
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("El número de pesos debe coincidir con el número de modelos")
    
    def fit(self, data: pd.Series) -> 'EnsembleModel':
        """Ajusta todos los modelos del ensemble"""
        for model in self.models:
            model.fit(data)
        
        # Calcular predicciones del ensemble
        predictions = []
        for model in self.models:
            predictions.append(model.fitted_values)
        
        # Combinar predicciones con pesos
        self.fitted_values = pd.Series(0, index=data.index)
        for i, pred in enumerate(predictions):
            self.fitted_values += self.weights[i] * pred
        
        self.residuals = data - self.fitted_values
        self.is_fitted = True
        
        return self
    
    def predict(self, steps: int) -> pd.Series:
        """Realiza predicciones combinando todos los modelos"""
        if not self.is_fitted:
            raise ValueError("El modelo debe estar ajustado antes de hacer predicciones")
        
        predictions = []
        for model in self.models:
            predictions.append(model.predict(steps))
        
        # Combinar predicciones con pesos
        ensemble_pred = pd.Series(0, index=predictions[0].index)
        for i, pred in enumerate(predictions):
            ensemble_pred += self.weights[i] * pred
        
        return ensemble_pred
    
    def forecast(self, data: pd.Series, steps: int) -> pd.Series:
        """Realiza pronósticos ajustando todos los modelos"""
        self.fit(data)
        return self.predict(steps)
