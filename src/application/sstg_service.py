import numpy as np
import random
from src.domain.astrophysics.sstg.engine import StochasticGravityEngine
from src.domain.astrophysics.sstg.providers.universal_provider import UniversalMetricProvider
from src.domain.astrophysics.sstg.injectors.quantum_effects import QuantumEffectInjector

class SSTGService:
    """Orquestador SSTG: Generador de Datasets de Entrenamiento QML."""

    def __init__(self):
        self.engine = StochasticGravityEngine()
        self.provider = UniversalMetricProvider()
        self.quantum_injector = QuantumEffectInjector()

    def generate_training_sample(self):
        """Genera una muestra individual con realismo estadístico y ruido."""
        
        # 1. Parámetros ponderados estadísticamente
        params = self.engine.sample_parameters()
        category = params['theory']

        # 2. Diales de 'Física de Frontera' (Activación basada en la categoría sorteada)
        self._apply_theory_dials(params, category)

        # 3. Generación de la onda pura (Aproximación PN)
        t, strain_pure = self.provider.solve_spacetime(params)

        # 4. Inyección de Ruido Realista (Ponderación de SNR)
        # La potencia del ruido aumenta con la distancia (SNR baja es más probable)
        snr_target = 2500 / params['distance'] # Estimación de SNR observacional
        noise_std = np.std(strain_pure) / snr_target
        noise = np.random.normal(0, noise_std, len(strain_pure))
        
        final_strain = strain_pure + noise

        # 5. Post-procesamiento según anomalías
        if category == "QUANTUM":
            final_strain = self.quantum_injector.apply_theory_drift(final_strain, params)
        elif category == "MULTIVERSE" and params.get('is_wormhole'):
            final_strain = self._apply_wormhole_effects(final_strain)

        return {
            "strain": final_strain,
            "metadata": params,
            "label": category,
            "snr_real": snr_target
        }

    def _apply_theory_dials(self, params, category):
        """Configura los parámetros de desviación según la probabilidad sorteada."""
        if category == "DARK_MATTER":
            params['rho_dark_matter'] = random.uniform(1e11, 1e14)
        elif category == "MULTIVERSE":
            params['extra_dims'] = random.uniform(0.01, 0.08)
            params['is_wormhole'] = random.random() > 0.7
        elif category == "MODIFIED_GRAVITY":
            params['graviton_mass'] = 1e-23
            params['eta1'] = random.uniform(0.05, 0.2)
        elif category == "QUANTUM":
            params['eta1'] = random.uniform(0.1, 0.4)

    def _apply_wormhole_effects(self, strain):
        """Efecto topológico: Inversión y eco de salida."""
        return -0.9 * np.roll(strain, 150) + strain