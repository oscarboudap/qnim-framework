"""
Application Service: Synthetic Data Generation Use Case
========================================================

Orquesta generación de dataset sintético para entrenamiento/validación.

NOTA ARQUITECTÓNICA: SSTG (síntesis de datos) es concern de infraestructura,
no domain. Esta clase lo refleja delegando al puerto ISyntheticDataGeneratorPort.

Responsibilities:
- Definiir parámetros de síntesis
- Coordinar generación (via puerto)
- Coordinar persistencia (via puerto)
- Retornar metadata del dataset

No contiene: Código de síntesis, I/O directo.
"""

from dataclasses import dataclass
from typing import List

from src.domain.astrophysics.value_objects import TheoryFamily
from src.application.ports import (
    ISyntheticDataGeneratorPort,
    IStoragePort,
)
from src.application.dto import SyntheticDatasetInfo


@dataclass(frozen=True)
class SynthesisParameters:
    """Parámetros de síntesis de dataset."""
    total_events: int = 200
    rg_fraction: float = 0.5  # Proporción Relatividad General
    golden_events_fraction: float = 0.3  # Alto SNR
    min_mass_solar: float = 3.0
    max_mass_solar: float = 100.0
    distance_mpc: float = 400.0  # Fixed distancia comóvil


class SyntheticDataGenerationUseCase:
    """
    Caso de Uso: Generar dataset masivo de ondas gravitacionales.
    
    Orquesta síntesis (infrastructure) + persistencia (infrastructure).
    
    ACCIÓN: Antes de producción, mover ISyntheticDataGeneratorPort
    a src/infrastructure/sstg/ (no está en domain, es data synthesis).
    """
    
    def __init__(self,
                 data_generator: ISyntheticDataGeneratorPort,
                 storage: IStoragePort):
        """
        Args:
            data_generator: Puerto para síntesis (infrastructure)
            storage: Puerto para persistencia (infrastructure)
        """
        self.generator = data_generator
        self.storage = storage
    
    def execute(self, params: SynthesisParameters) -> SyntheticDatasetInfo:
        """
        Ejecuta síntesis balanceada 50% GR / 50% LQG.
        
        Args:
            params: Configuración de síntesis
            
        Returns:
            SyntheticDatasetInfo: Metadata del dataset creado
        """
        print(f"🎲 Iniciando Síntesis de {params.total_events} eventos...")
        
        samples = []
        num_rg = int(params.total_events * params.rg_fraction)
        num_lqg = params.total_events - num_rg
        num_golden = int(params.total_events * params.golden_events_fraction)
        
        golden_count = 0
        
        for i in range(params.total_events):
            # 1. Elegir teoría (50/50 balance)
            is_gr = i < num_rg
            target_theory = TheoryFamily.GENERAL_RELATIVITY if is_gr else TheoryFamily.LOOP_QUANTUM_GRAVITY
            
            # 2. Golden events: distancia corta (alto SNR)
            distance = 100.0 if golden_count < num_golden else params.distance_mpc
            if golden_count < num_golden:
                golden_count += 1
            
            # 3. Parámetros de masa (variación)
            m1 = params.min_mass_solar + (i % 10) * (params.max_mass_solar - params.min_mass_solar) / 10
            m2 = m1 * 0.85  # Ratio típico
            
            # 4. Síntesis via infraestructura (no aquí en application)
            strain = self.generator.synthesize_event(
                m1_solar_masses=m1,
                m2_solar_masses=m2,
                distance_mpc=distance,
                theory_family=target_theory.value
            )
            
            # 5. Empaquetar
            samples.append({
                "strain": strain,
                "label": target_theory.value,
                "metadata": {
                    "m1": m1,
                    "m2": m2,
                    "distance": distance,
                    "is_golden": golden_count <= num_golden
                }
            })
        
        # 6. Persistir via infraestructura
        output_path = self.storage.save_batch(samples)
        
        print(f"✅ Dataset guardado en: {output_path}")
        
        return SyntheticDatasetInfo(
            num_events_rg=num_rg,
            num_events_lqg=num_lqg,
            total_events=params.total_events,
            output_directory=output_path,
            golden_events_count=num_golden
        )