import numpy as np

class HorizonTopologyInjector:
    """
    CAPA 6: ESTRUCTURA DEL HORIZONTE Y TOPOLOGÍA.
    Inyecta ecos, discretización cuántica y violaciones del teorema de no-cabello.
    """
    
    @staticmethod
    def inject_fuzzball_echoes(strain: np.ndarray, total_mass_sun: float, fs: int) -> np.ndarray:
        """
        Teoría de Cuerdas (Mathur): El BH no tiene horizonte, sino una superficie de 'cuerdas'.
        La onda rebota generando ecos separados por Delta_t ~ 2*r_star*ln(M/l_planck).
        """
        echo_signal = np.zeros_like(strain)
        # Cálculo del delay basado en la masa (simplificado a ~10-50ms por masa estelar)
        delay_samples = int(fs * (total_mass_sun * 0.001)) 
        reflectivity = 0.3 # La bola de cuerdas absorbe el 70%, rebota el 30%
        
        # Inyectamos 3 ecos que decaen exponencialmente
        for i in range(1, 4):
            shift = delay_samples * i
            if shift < len(strain):
                echo_signal[shift:] += strain[:-shift] * (reflectivity ** i)
                
        return strain + echo_signal

    @staticmethod
    def inject_lqg_area_quantization(strain: np.ndarray) -> np.ndarray:
        """
        Loop Quantum Gravity (LQG): El espacio-tiempo tiene estructura granular.
        El área del horizonte está cuantizada (Espectro de Barbero-Immirzi), 
        produciendo saltos discretos en la métrica.
        """
        # La amplitud h(t) solo puede tomar valores en escalones cuánticos de Planck
        quantum_step_size = np.max(np.abs(strain)) * 0.05 
        return np.round(strain / quantum_step_size) * quantum_step_size