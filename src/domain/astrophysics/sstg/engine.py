import numpy as np
from src.domain.astrophysics.value_objects import TheoryFamily, SolarMass
from src.domain.astrophysics.sstg.providers.kerr_provider import KerrVacuumProvider
from src.domain.astrophysics.sstg.injectors.layer5_beyond_gr import BeyondGRInjector
from src.domain.astrophysics.sstg.injectors.layer6_horizon_topology import HorizonTopologyInjector

class QuantumUniverseEngine:
    """
    Orquestador del Dominio SSTG (Synthetic Signal Time Generation).
    Ensambla las formas de onda capa por capa basándose en el framework teórico QNIM.
    """
    
    def __init__(self, sample_rate: int = 4096):
        self.fs = sample_rate
        self.kerr_provider = KerrVacuumProvider()
        self.beyond_gr = BeyondGRInjector()
        self.horizon = HorizonTopologyInjector()

    def synthesize_event(self, m1: float, m2: float, distance: float, target_theory: TheoryFamily) -> np.ndarray:
        """Sintetiza una onda gravitacional aplicando física desde la Capa 2 hasta la 7."""
        
        # 1. CAPA 2 (Base): Creamos la plantilla perfecta de Relatividad General
        strain = self.kerr_provider.generate_base_strain(
            m1=SolarMass(m1), 
            m2=SolarMass(m2), 
            distance_mpc=distance, 
            fs=self.fs
        )
        
        total_mass = m1 + m2

        # 2. CAPAS PROFUNDAS (Inyectamos anomalías según la teoría seleccionada)
        if target_theory == TheoryFamily.GENERAL_RELATIVITY:
            # Vacío perfecto, no se añade "pelo"
            pass 
            
        elif target_theory == TheoryFamily.LOOP_QUANTUM_GRAVITY:
            # LQG afecta a la estructura discreta del espacio
            strain = self.horizon.inject_lqg_area_quantization(strain)
            
        elif target_theory == TheoryFamily.STRING_FUZZBALL:
            # Fuzzballs generan reflexiones (ecos) near-horizon
            strain = self.horizon.inject_fuzzball_echoes(strain, total_mass, self.fs)
            
        elif target_theory == TheoryFamily.SCALAR_TENSOR:
            # Horndeski/DHOST pueden simularse con un gravitón masivo o dispersión
            strain = self.beyond_gr.inject_massive_graviton_dispersion(strain, graviton_mass_ev=1e-22, fs=self.fs)
            
        # 3. Limpieza de artefactos numéricos
        strain = np.ascontiguousarray(strain, dtype=np.float64)
        return np.nan_to_num(strain, nan=0.0)