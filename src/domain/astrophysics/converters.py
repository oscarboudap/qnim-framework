"""
CONVERTIDORES Y ADAPTADORES DE DOMINIO.

Punto de entrada para conversiones entre dominios y cálculos específicos.

NOTA: La mayoría de lógica se ha movido a domain_services.py (limpieza DDD).
Este archivo mantiene compatibilidad hacia atrás delegando.
"""

import numpy as np
from .domain_services import AstrophysicalCalculus


class AstrophysicsMath:
    """
    [LEGACY] Interfaz de compatibilidad hacia atrás.
    
    ⚠️ DEPRECADA: Usar directamente AstrophysicalCalculus en new code.
    
    Se mantiene por compatibilidad con código existente en SSTG.
    Todas las llamadas delegan a domain_services.py.
    """
    
    # Constantes (legacy access)
    G = 6.67430e-11
    C = 299792458
    MSUN_KG = 1.98847e30
    L_PLANCK = 1.616255e-35

    @staticmethod
    def get_chirp_mass(m1: float, m2: float) -> float:
        """DEPRECADA: Usar AstrophysicalCalculus.chirp_mass()"""
        return AstrophysicalCalculus.chirp_mass(m1, m2)

    @staticmethod
    def calculate_echo_delay(m_total_sun: float) -> float:
        """
        DEPRECADA: Usar AstrophysicalCalculus.echo_delay_quantum_gravity()
        
        Calcula el tiempo de delay logarítmico para ecos de Gravedad Cuántica.
        $$\Delta t \sim 2 r_* \ln(M / \ell_{Planck})$$
        """
        return AstrophysicalCalculus.echo_delay_quantum_gravity(m_total_sun)

    @staticmethod
    def calculate_f_isco(m_total_sun: float) -> float:
        """
        DEPRECADA: Usar AstrophysicalCalculus.innermost_stable_circular_orbit_frequency()
        
        Frecuencia de la Innermost Stable Circular Orbit.
        """
        return AstrophysicalCalculus.innermost_stable_circular_orbit_frequency(m_total_sun)