import numpy as np

class QuantumEffectInjector:
    """
    Inyector de firmas de Gravedad Cuántica.
    Aplica efectos basados en la etiqueta sorteada estadísticamente.
    """

    def apply_theory_drift(self, strain, params):
        """
        Punto de entrada único. Elige la física a aplicar según params['theory'].
        """
        theory = params.get('theory', 'RG')

        # Si la estadística decidió que este evento es QUANTUM
        if theory == "QUANTUM":
            # 1. Aplicamos discretización (LQG)
            strain = self.apply_lqg_discretization(strain)
            # 2. Aplicamos ecos de Shapiro (Cuerdas/Fuzzballs)
            strain = self.apply_string_echoes(strain, params)
            
        return strain

    @staticmethod
    def apply_lqg_discretization(strain, scale=1e-21): 
        """
        Introduce una discretización en la fase de la onda.
        Simula que el espacio-tiempo tiene 'píxeles' (Nivel LQG).
        """
        # Ajustamos la escala para que sea perceptible en el strain (h ~ 10^-21)
        phase_shift = np.round(strain / scale) * scale
        return strain + (phase_shift * 0.05)

    @staticmethod
    def apply_string_echoes(strain, params, n_echoes=3):
        """
        Añade ecos de Shapiro tras el merger. 
        Firma característica de los Fuzzballs (Teoría de Cuerdas).
        """
        echo_signal = np.zeros_like(strain)
        # 4096 es la frecuencia de muestreo (fs) definida en el provider
        delay = int(0.05 * 4096) 
        for i in range(1, n_echoes + 1):
            amplitude = 0.3**i 
            shift = i * delay
            if shift < len(strain):
                echo_signal[shift:] += strain[:len(strain)-shift] * amplitude
        
        return strain + echo_signal