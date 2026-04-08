import numpy as np

class HubbleMetrology:
    """
    Valida la constante de Hubble (H0) contra los modelos Planck (67.4) y SH0ES (73.0).
    """
    def __init__(self):
        self.h0_planck = 67.4
        self.h0_shoes = 73.0

    def evaluate_tension(self, h0_inferred: float, sigma_h0: float):
        """
        Determina con cuántas sigmas de confianza tu software apoya un modelo u otro.
        """
        tension_planck = abs(h0_inferred - self.h0_planck) / sigma_h0
        tension_shoes = abs(h0_inferred - self.h0_shoes) / sigma_h0
        
        return {
            "p_value_planck": np.exp(-0.5 * tension_planck**2),
            "p_value_shoes": np.exp(-0.5 * tension_shoes**2),
            "preferred_model": "Planck" if tension_planck < tension_shoes else "SH0ES"
        }