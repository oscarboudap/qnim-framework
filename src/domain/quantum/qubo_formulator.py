"""Servicios de dominio para formulación de problemas QUBO.

Nível Postdoctoral: Convierte problemas de optimización clásicos
(template matching, variable selection) en problemas QUBO que pueden
resolverse con recocido cuántico o QAOA.
"""

from typing import List
import numpy as np

from .value_objects import QUBOProblem, TemplateSignal


class TemplateMatchingQUBO:
    """
    Servicio de dominio stateless que genera formulaciones QUBO para
    template matching en detección de ondas gravitacionales.
    
    Objetivo: Encontrar la plantilla que mejor encaja con la señal observada.
    Formulación: H = H_cost + H_penalty
      H_cost = -Σ_i (1 - MSE_i) × x_i  (minimizar MSE)
      H_penalty = P × Σ_{i<j} x_i × x_j  (una única plantilla: one-hot)
    """
    
    @staticmethod
    def build_formulation(target_signal: np.ndarray,
                        templates: List[TemplateSignal],
                        penalty_weight: float = None) -> QUBOProblem:
        """
        Construye el Hamiltoniano QUBO para template matching.
        
        Objetivo: Minimizar MSE entre target_signal y template[i] seleccionado,
        sujeto a restricción one-hot (solo una plantilla activa).
        
        Args:
            target_signal: Strain observada (array 1D)
            templates: Lista de TemplateSignal (value objects)
            penalty_weight: Peso de penalización para one-hot.
                           Si None, se calcula: 10 × max(MSE)
                           
        Returns:
            QUBOProblem con linear_terms, quadratic_terms, offset
            
        Raises:
            ValueError: Si target_signal o templates están vacíos
        """
        if len(target_signal) == 0:
            raise ValueError("target_signal no puede estar vacía")
        
        if len(templates) == 0:
            raise ValueError("Debe haber al menos una plantilla")
        
        if len(templates) > 1000:
            raise ValueError("Máximo 1000 plantillas permitidas")
        
        num_templates = len(templates)
        linear_terms = {}
        quadratic_terms = {}
        
        # Calcular MSE individuales
        mse_values = []
        for tmpl in templates:
            if len(tmpl.strain_data) != len(target_signal):
                raise ValueError(
                    f"Longitud de plantilla ({len(tmpl.strain_data)}) "
                    f"!= longitud de señal ({len(target_signal)})"
                )
            mse = np.mean((target_signal - tmpl.strain_data) ** 2)
            mse_values.append(mse)
        
        # Auto-calcular penalty_weight si no se proporciona
        if penalty_weight is None:
            max_mse = max(mse_values) if mse_values else 1.0
            penalty_weight = 10.0 * max(max_mse, 1e-6)  # Evitar cero
        
        if penalty_weight <= 0:
            raise ValueError("penalty_weight debe ser positivo")
        
        # 1. H_cost: Términos lineales (MSE de cada plantilla)
        # Queremos minimizar MSE, así que hamiltonian = MSE
        for i, mse in enumerate(mse_values):
            linear_terms[i] = mse
        
        # 2. H_penalty: Términos cuadráticos (one-hot constraint)
        # Σ_{i<j} x_i × x_j penalizado para que no puedan ambos estar activos
        for i in range(num_templates):
            for j in range(i + 1, num_templates):
                quadratic_terms[(i, j)] = penalty_weight
        
        # 3. Offset: corrección para energía del estado one-hot all-zeros
        #    (necesario para que el punto (0,0,...,0) no sea óptimo)
        offset = penalty_weight * num_templates
        
        return QUBOProblem(
            linear_terms=linear_terms,
            quadratic_terms=quadratic_terms,
            offset=offset
        )
    
    @staticmethod
    def compute_penalty_weight(mse_values: List[float]) -> float:
        """
        Calcula un peso de penalización apropiado basado en los MSEs.
        
        Heurística: penalty = 10 × max(MSE) para garantizar que
        la restricción one-hot sea más costosa que cualquier diferencia de MSE.
        
        Args:
            mse_values: Lista de valores MSE (uno por plantilla)
            
        Returns:
            Peso de penalización (float positivo)
        """
        if not mse_values:
            return 1.0
        
        max_mse = max(mse_values)
        return 10.0 * max(max_mse, 1e-6)