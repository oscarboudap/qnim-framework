# src/domain/astrophysics/converters.py
import numpy as np
from .value_objects import SolarMass

class AstrophysicsMath:
    """Servicio de Dominio que encapsula las matemáticas exactas de la astrofísica de agujeros negros."""
    
    G_CONST = 6.67430e-11 # m^3 kg^-1 s^-2
    C_CONST = 299792458   # m/s
    MSUN_KG = 1.98847e30  # kg

    @staticmethod
    def extract_individual_masses(m_chirp_det: float, eta: float = 0.245) -> tuple[SolarMass, SolarMass]:
        """Convierte la Masa de Chirp detectada en masas individuales (m1, m2)."""
        m_total = m_chirp_det / (eta**(3/5))
        diff = np.sqrt(max(0, 1 - 4 * eta))
        
        m1 = (m_total / 2) * (1 + diff)
        m2 = (m_total / 2) * (1 - diff)
        
        return SolarMass(m1), SolarMass(m2)

    @staticmethod
    def calculate_kerr_expected_quadrupole(mass_sun: float, spin_param: float) -> float:
        """
        Calcula el momento cuadrupolar exacto en Relatividad General (Kerr).
        Fórmula del Teorema de No-Cabello: Q = -J^2 / M = - a^2 * M^3 (en unidades naturales)
        """
        # Simplificación teórica para el dominio
        return - (spin_param**2) * (mass_sun**3)

    @staticmethod
    def calculate_echo_delay_time(mass_sun: float) -> float:
        """
        Cálculo del retraso temporal de los ecos en Gravedad Cuántica / Fuzzballs.
        Delta_t ~ 2 * r_* * ln(M / l_planck)
        """
        # Aproximación del dominio para un Schwarzschild BH en ms
        mass_sec = mass_sun * 4.925490947e-6
        return 2 * mass_sec * np.log(1e40) * 1000 # Retorno en milisegundos