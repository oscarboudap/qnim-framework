"""
Infrastructure: Sklearn Preprocessing Adapter
==============================================

Adaptador que implementa IPreprocessingPort usando Sklearn.

Pipeline:
    StandardScaler → PCA (12 components) → MinMaxScaler ([-π, π] range)
    
Mantiene estado entre fit_transform y transform para preprocessing
consistente de datos de entrenamiento vs. test.
"""

import numpy as np
from pathlib import Path
from typing import Tuple
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from src.application.ports import IPreprocessingPort
from src.infrastructure.exceptions import PreprocessingException


class SklearnPreprocessor(IPreprocessingPort):
    """
    Preprocesar datos usando sklearn Pipeline.
    
    Implements:
        - fit_transform(): Ajusta + transforma datos
        - transform(): Transforma usando modelo ajustado
        - save(): Almacena pipeline en disco
        - load(): Carga pipeline previo
    
    Pipeline (estateless pero guarda modelo):
        1. StandardScaler: media=0, std=1
        2. PCA: reducir a 12 componentes principales
        3. MinMaxScaler: escalar a rango [-π, π]
    """
    
    # Hiperparámetros centralizados (según ModelTrainingUseCase)
    N_COMPONENTS = 12          # Componentes PCA
    MIN_VALUE = -np.pi         # Rango mínimo
    MAX_VALUE = np.pi          # Rango máximo
    
    def __init__(self):
        """Inicializa pipeline vacío (sin ajuste aún)."""
        self.pipeline: Pipeline = None
        self.is_fitted: bool = False
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Ajusta el pipeline en X y lo aplica.
        
        Args:
            X: Features [n_samples, n_features]
        
        Returns:
            X transformado [n_samples, N_COMPONENTS]
        
        Raises:
            PreprocessingException: Si X inválida o falla el pipeline
        """
        try:
            # Validación
            if not isinstance(X, np.ndarray):
                raise TypeError("X debe ser np.ndarray")
            if len(X.shape) != 2:
                raise ValueError("X debe ser 2D [n_samples, n_features]")
            if X.shape[0] == 0:
                raise ValueError("X debe tener al menos 1 muestra")
            if X.shape[1] < self.N_COMPONENTS:
                raise ValueError(
                    f"X tiene {X.shape[1]} features, "
                    f"pero PCA requiere >= {self.N_COMPONENTS}"
                )
            
            # Construir pipeline si no existe
            if self.pipeline is None:
                self.pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=self.N_COMPONENTS)),
                    ('minmax', MinMaxScaler(feature_range=(self.MIN_VALUE, self.MAX_VALUE)))
                ])
            
            # Ajustar y transformar
            X_transformed = self.pipeline.fit_transform(X)
            self.is_fitted = True
            
            print(f"✅ Fit+transform completado: {X.shape} → {X_transformed.shape}")
            return X_transformed
        
        except Exception as e:
            raise PreprocessingException(
                f"Error en fit_transform: {str(e)}"
            )
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica pipeline ajustado a X (sin reajustar).
        
        Args:
            X: Features [n_samples, n_features]
        
        Returns:
            X transformado [n_samples, N_COMPONENTS]
        
        Raises:
            PreprocessingException: Si pipeline no ajustado o falla transformación
        """
        try:
            if not self.is_fitted or self.pipeline is None:
                raise RuntimeError(
                    "Pipeline no ha sido ajustado. "
                    "Llama fit_transform primero."
                )
            
            # Validación
            if not isinstance(X, np.ndarray):
                raise TypeError("X debe ser np.ndarray")
            if len(X.shape) != 2:
                raise ValueError("X debe ser 2D [n_samples, n_features]")
            if X.shape[0] == 0:
                raise ValueError("X debe tener al menos 1 muestra")
            
            X_transformed = self.pipeline.transform(X)
            print(f"✅ Transform completado: {X.shape} → {X_transformed.shape}")
            return X_transformed
        
        except Exception as e:
            raise PreprocessingException(
                f"Error en transform: {str(e)}"
            )
    
    def save(self, path: str) -> None:
        """
        Guarda el pipeline ajustado en disco.
        
        Args:
            path: Ruta de salida (formato .pkl)
        
        Raises:
            PreprocessingException: Si falla la escritura
        """
        try:
            if not self.is_fitted or self.pipeline is None:
                raise RuntimeError("No hay pipeline para guardar. Ejecuta fit_transform primero.")
            
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.pipeline, output_path)
            print(f"💾 Pipeline guardado en: {output_path}")
        
        except Exception as e:
            raise PreprocessingException(
                f"Error al guardar pipeline: {str(e)}"
            )
    
    def load(self, path: str) -> None:
        """
        Carga un pipeline previamente guardado.
        
        Args:
            path: Ruta del archivo .pkl
        
        Raises:
            PreprocessingException: Si archivo no existe o falla la carga
        """
        try:
            input_path = Path(path)
            if not input_path.exists():
                raise FileNotFoundError(f"Pipeline no encontrado: {path}")
            
            self.pipeline = joblib.load(input_path)
            self.is_fitted = True
            print(f"📦 Pipeline cargado desde: {input_path}")
        
        except Exception as e:
            raise PreprocessingException(
                f"Error al cargar pipeline: {str(e)}"
            )
