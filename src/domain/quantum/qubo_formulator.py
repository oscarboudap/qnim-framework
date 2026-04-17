# src/domain/quantum/qubo_formulator.py
from typing import List, Dict
import numpy as np
from src.domain.quantum.entities import QUBOProblem

class TemplateMatchingQUBO:
    """
    Generador de la matriz QUBO para optimización de parámetros.
    Convierte la búsqueda de 'la plantilla que mejor encaja con la señal' 
    en un paisaje de energía donde el valle más profundo es la respuesta correcta.
    """
    
    @classmethod
    def build_formulation(cls, target_signal: np.ndarray, templates: List[dict]) -> QUBOProblem:
        """
        Construye el Hamiltoniano H = H_cost + H_penalty
        donde H_cost minimiza el error cuadrático medio (MSE) entre la señal y las plantillas,
        y H_penalty asegura que solo se seleccione UNA plantilla (one-hot encoding).
        """
        num_templates = len(templates)
        linear = {}
        quadratic = {}
        
        penalty_weight = 1000.0 # Penalización fuerte para forzar una única elección
        
        # 1. H_cost: Errores individuales (Términos lineales)
        for i, tmpl in enumerate(templates):
            mse = np.mean((target_signal - tmpl['strain'])**2)
            # Añadimos el peso del MSE y la penalización lineal del one-hot (-2P)
            linear[i] = mse - penalty_weight
            
        # 2. H_penalty: Términos cruzados (Términos cuadráticos)
        # Asegura que q_i * q_j = 0 si i != j (solo una plantilla activa)
        for i in range(num_templates):
            for j in range(i + 1, num_templates):
                quadratic[(i, j)] = 2.0 * penalty_weight
                
        return QUBOProblem(linear_terms=linear, quadratic_terms=quadratic, offset=penalty_weight)