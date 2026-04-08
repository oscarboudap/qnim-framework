import numpy as np
import dataclasses

class AnomalyGeneratorService:
    """
    Generador de anomalías basado en la ruptura del Teorema de No-Cabello.
    Simula desviaciones del momento cuadrupolar (M2) y ecos de Gravedad Cuántica.
    """
    def apply_theory_drift(self, signal, theory="LQG", epsilon=0.15):
        """
        Inyecta desviaciones basadas en la jerarquía de teorías (Punto 2.2).
        Fuzzballs: Modifica el momento cuadrupolar M2 y genera ecos de superficie.
        LQG: Simula la dispersión por granularidad del espacio-tiempo.
        """
        strain = np.copy(signal.strain)
        peak_idx = np.argmax(np.abs(strain))
        # Vector de tiempo relativo al pico (Merger)
        t_post = np.arange(len(strain) - peak_idx) / signal.sample_rate

        if theory == "LQG":
            # 1. DISCRETIZACIÓN DEL ESPACIO-TIEMPO (Punto 2.2.2.1)
            # Efecto: Modos cuasinormales (QNM) con corrimiento de fase por granularidad.
            # Simulamos una dispersión sutil que altera la frecuencia del ringdown.
            lqg_mod = epsilon * np.sin(2.0 * np.pi * 60 * t_post) * np.exp(-t_post / 0.04)
            strain[peak_idx:] += lqg_mod
            
        elif theory == "STRING_FUZZBALL":
            # 2. VIOLACIÓN DEL MOMENTO CUADRUPOLAR M2 (Punto 2.2.2.2)
            # Según RG: M2 = -a^2 * M^3. Aquí inyectamos el 'cabello' o métrica sucia.
            # Representa la vibración de la superficie física del Fuzzball.
            quadrupole_drift = epsilon * np.cos(2.0 * np.pi * 30 * t_post) * np.exp(-t_post / 0.06)
            
            # 3. ECOS DE SUPERFICE (Shapiro Echos)
            # Característica clave de cuerdas: el horizonte no es un vacío, hay reflexión.
            # Inyectamos un eco amortiguado con un retraso delta_t logarítmico.
            echo_delay = int(0.025 * signal.sample_rate) # ~25ms de retraso
            if peak_idx + echo_delay + 100 < len(strain):
                echo = strain[peak_idx:peak_idx+100] * (epsilon * 0.4)
                strain[peak_idx+echo_delay : peak_idx+echo_delay+100] += echo
            
            strain[peak_idx:] += quadrupole_drift
            
        return dataclasses.replace(signal, strain=strain)

    def get_quadrupole_deviation(self, prob_inference: float) -> float:
        """
        Transforma la confianza del VQC en un valor de 'Cabello' (delta Q).
        Formalismo: M2 = M2_kerr * (1 + delta_q)
        A mayor rotación en el Espacio de Hilbert, mayor evidencia de métrica no-Kerr.
        """
        # Normalizamos la desviación respecto al centroide de ruido (0.5)
        # Un delta_q > 0.05 suele considerarse evidencia de física exótica.
        delta_q = abs(prob_inference - 0.5) * 2.2 
        return delta_q