import numpy as np
from src.domain.astrophysics.value_objects import SolarMass

class KerrVacuumProvider:
    """
    CAPA 2: GEOMETRÍA INTRÍNSECA (Relatividad General Pura).
    Genera la forma de onda fenomenológica IMR (Inspiral-Merger-Ringdown)
    en un vacío perfecto, asumiendo el Teorema de No-Cabello.
    """
    G = 6.67430e-11
    C = 299792458.0
    M_SUN = 1.98847e30

    def generate_base_strain(self, m1: SolarMass, m2: SolarMass, distance_mpc: float, fs: int = 4096, duration: float = 4.0) -> np.ndarray:
        t = np.linspace(0, duration, int(fs * duration))
        
        m_chirp_kg = ((m1.value * m2.value)**0.6 / (m1.value + m2.value)**0.2) * self.M_SUN
        
        # Tiempo adimensional hasta el merger (tau)
        tau = (self.C**3 * (duration - t)) / (5 * self.G * m_chirp_kg)
        tau = np.maximum(tau, 1e-10)
        
        # Frecuencia orbital y fase (Post-Newtoniana básica para el dominio)
        omega = (1/8) * (self.C**3 / (self.G * m_chirp_kg)) * tau**(-3/8)
        phase = -2 * tau**(5/8)
        
        # Decaimiento estricto 1/dL de la Relatividad General
        d_meters = distance_mpc * 3.0856e22
        amplitude = (self.G * m_chirp_kg / (self.C**2 * d_meters)) * (tau**(-1/4))
        
        strain = amplitude * np.cos(phase)
        return np.nan_to_num(strain, nan=0.0, posinf=0.0, neginf=0.0)