import numpy as np

class RelativityProvider:
    """Resuelve formas de onda basadas en Relatividad General (Kerr)."""
    
    def generate_strain(self, params, fs=4096, duration=2):
        t = np.linspace(0, duration, int(fs * duration))
        # Aproximación de Chirp simple para validación inicial
        # M_chirp = ((m1*m2)^3/5) / (m1+m2)^1/5
        m_chirp = ((params['m1']*params['m2'])**0.6) / (params['m1']+params['m2'])**0.2
        
        # Evolución de frecuencia de potencia 3/8 (fase de inspiral)
        f = 50 * (1 - t/duration)**(-3/8) 
        strain = np.sin(2 * np.pi * f * t) / params['distance']
        return t, strain