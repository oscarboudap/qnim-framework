import numpy as np

class BeyondGRInjector:
    """
    CAPA 5: FÍSICA MÁS ALLÁ DEL GR (Propagación y Gravedad Modificada).
    Inyecta firmas de dispersión y fuga de energía durante el viaje cosmológico de la onda.
    """
    
    @staticmethod
    def inject_kaluza_klein_leakage(strain: np.ndarray, extra_dimensions: int, distance_mpc: float) -> np.ndarray:
        """
        Teorías ADD/Randall-Sundrum: Parte de la energía gravitacional escapa al 'bulk' 
        (dimensiones extra), causando una atenuación de amplitud anómala h ~ d_L^{-(1+n/2)}.
        """
        if extra_dimensions == 0: return strain
        # Factor de amortiguación anómalo por dimensiones extra
        damping_factor = 1.0 / (1.0 + (distance_mpc / 1000) * (extra_dimensions / 2.0))
        return strain * damping_factor

    @staticmethod
    def inject_massive_graviton_dispersion(strain: np.ndarray, graviton_mass_ev: float, fs: int) -> np.ndarray:
        """
        Teoría de dRGT (Gravedad Masiva): El gravitón tiene masa, produciendo 
        una dispersión donde las bajas frecuencias viajan más lento.
        """
        if graviton_mass_ev <= 0.0: return strain
        # Transformada de Fourier para aplicar retraso dependiente de la frecuencia
        freqs = np.fft.rfftfreq(len(strain), d=1/fs)
        spectrum = np.fft.rfft(strain)
        
        # Desfase de fase proporcional a m_g^2 / f (simplificación fenomenológica)
        phase_shift = np.exp(-1j * (graviton_mass_ev**2) / (freqs + 1e-5))
        spectrum_shifted = spectrum * phase_shift
        
        return np.fft.irfft(spectrum_shifted, n=len(strain))