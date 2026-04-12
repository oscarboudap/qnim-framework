from cmath import phase

import numpy as np

class UniversalMetricProvider:
    """
    Motor de Resolución de Formas de Onda Parametrizadas de QNIM.
    Integra Relatividad General, Materia Oscura, Multiverso y Gravedad Modificada.
    """

    def __init__(self, fs=4096, duration=4):
        self.fs = fs
        self.duration = duration
        self.C = 299792458.0
        self.G = 6.67430e-11
        self.M_SUN = 1.98847e30

    def solve_spacetime(self, params):
        t = np.linspace(0, self.duration, int(self.fs * self.duration))
        
        # 1. Parámetros de Masa y Geometría base
        m_chirp = ((params['m1'] * params['m2'])**0.6 / (params['m1'] + params['m2'])**0.2) * self.M_SUN
        
        # 2. DIALES DE FÍSICA DE FRONTERA
        dev_1 = params.get('eta1', 0.0)    # Desviación métrica general
        rho_dm = params.get('rho_dark_matter', 0.0) # Materia Oscura (Fricción)
        m_g = params.get('graviton_mass', 0.0)      # Gravedad Modificada (Dispersión)
        n_extra = params.get('extra_dims', 0.0)     # Multiverso (Fuga energética)

        # 3. Evolución de la Frecuencia Modificada
        # tau es el tiempo adimensional hasta el merger
        tau = (self.C**3 * (self.duration - t)) / (5 * self.G * m_chirp)
        tau = np.maximum(tau, 1e-10)
        # Ajuste por Materia Oscura (Dark Dress): acelera el tiempo de colapso
        tau_eff = tau * (1 - rho_dm * 1e-15) 
        
        # Frecuencia orbital omega(t) con corrección de gravitón masivo (dispersión)
        omega_base = (1/8) * (self.C**3 / (self.G * m_chirp)) * tau_eff**(-3/8)
        dispersion_mod = 1 + (m_g**2 / (omega_base + 1e-5)**2)
        omega_final = omega_base * dispersion_mod * (1 + dev_1 * tau_eff**(1/4))

        # 4. Integración de Fase
        phase = -2 * tau_eff**(5/8) * (1 + dev_1 * tau_eff**(1/4))

        # 5. Generación del Strain h(t) con Fuga Multiversal
        # En teorías de branas, la amplitud decae más rápido si hay dimensiones extra
        d_meters = params['distance'] * 1e6 * 3.0856e16
        amplitude_base = (self.G * m_chirp / (self.C**2 * d_meters))
        
        # Aplicamos decaimiento de dimensiones extra (Multiverso)
        amplitude = amplitude_base * (tau_eff**(-1/4)) / (1 + n_extra)
        
        # Al final de solve_spacetime:
        strain = amplitude * np.cos(phase)

        # Limpieza radical de valores no numéricos
        strain = np.ascontiguousarray(strain, dtype=np.float64)
        strain = np.nan_to_num(strain, nan=0.0, posinf=0.0, neginf=0.0)
        return t, np.nan_to_num(strain)