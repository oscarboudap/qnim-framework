import numpy as np
from scipy.integrate import quad

class HubbleSolverService:
    """
    Resuelve la constante de Hubble (H0) integrando la métrica FLRW.
    Aplica el formalismo de Sirenas Estándar para resolver la Tensión de Hubble.
    """
    def __init__(self, omega_m: float = 0.31, omega_lambda: float = 0.69):
        self.omega_m = omega_m
        self.omega_lambda = omega_lambda
        self.c = 299792.458 # km/s

    def _integrand(self, z):
        # Función de expansión E(z) para un universo plano Lambda-CDM
        return 1.0 / np.sqrt(self.omega_m * (1 + z)**3 + self.omega_lambda)

    def infer_h0(self, d_l_mpc: float, z_inferred: float) -> float:
        """
        Calcula H0 a partir de la Distancia de Luminosidad (dL) y el Redshift (z).
        dL = (c*(1+z)/H0) * integral(0, z, dz'/E(z'))
        """
        if z_inferred <= 0: return 0.0
        
        integral, _ = quad(self._integrand, 0, z_inferred)
        # H0 = (c * (1+z) * integral) / dL
        h0 = (self.c * (1 + z_inferred) * integral) / d_l_mpc
        return h0

    def calculate_fisher_uncertainty(self, snr: float, d_l: float) -> float:
        """
        Matriz de Información de Fisher simplificada para dL.
        La incertidumbre en la distancia escala como 1/SNR.
        """
        delta_dl = d_l / snr
        return delta_dl