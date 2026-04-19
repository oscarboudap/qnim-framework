"""
Infrastructure: SSTG (Synthetic Signal Template Generator) Adapter
==================================================================

Adaptador que implementa ISyntheticDataGeneratorPort.

Delega a QuantumUniverseEngine desde domain/astrophysics/sstg para
generar señales sintéticas de ondas gravitacionales.

Future: Planificado mover SSTG completamente a infrastructure como
adaptador que genera datos sintéticos para experimentos.
"""

import numpy as np
from typing import Optional

from src.application.ports import ISyntheticDataGeneratorPort
from src.infrastructure.exceptions import ReportingException


class SSTGAdapter(ISyntheticDataGeneratorPort):
    """
    Generador de datos sintéticos usando SSTG.
    
    Implements:
        - synthesize_event(): Genera strain data sintético
    
    Encapsulation:
        Delega a domain/astrophysics/sstg.QuantumUniverseEngine.
        Application no conoce detalles de cómo se generan señales.
    
    Future Movement Plan:
        1. Esta clase importa desde domain (transitorio)
        2. Event: "Move SSTG to infrastructure" (PR separado)
        3. Entonces: infrastructure/sstg_engine.py (copia de domain)
        4. Actualizar imports aquí
        5. Deprecate domain/astrophysics/sstg
    """
    
    def __init__(self):
        """
        Inicializa el adaptador SSTG.
        
        Nota: Importa lazy (en método) para evitar circular imports.
        """
        self._engine = None
    
    def _get_engine(self):
        """Lazy-load del motor SSTG (evita circular imports)."""
        if self._engine is None:
            try:
                # Importar al momento del uso
                from src.domain.astrophysics.sstg import QuantumUniverseEngine
                self._engine = QuantumUniverseEngine()
            except ImportError as e:
                raise ReportingException(
                    f"Error importando QuantumUniverseEngine: {str(e)}"
                )
        return self._engine
    
    def synthesize_event(self,
                        mass1_solar_masses: float,
                        mass2_solar_masses: float,
                        distance_mpc: float,
                        theory_family: str,
                        sampling_rate_hz: int = 4096,
                        duration_seconds: float = 8.0) -> np.ndarray:
        """
        Sintetiza un evento de onda gravitacional.
        
        Args:
            mass1_solar_masses: Masa del cuerpo 1 (M☉)
            mass2_solar_masses: Masa del cuerpo 2 (M☉)
            distance_mpc: Distancia del observador (Mpc)
            theory_family: Familia teórica ("GR", "scalar-tensor", "modified-gravity", etc.)
            sampling_rate_hz: Frecuencia de muestreo (default: 4096 como LIGO)
            duration_seconds: Duración total de la ventana (default: 8s)
        
        Returns:
            np.ndarray: Strain data simulado [n_samples,]
        
        Raises:
            ReportingException: Si parámetros inválidos o falla la síntesis
        """
        try:
            # Validación de parámetros
            if mass1_solar_masses <= 0 or mass2_solar_masses <= 0:
                raise ValueError("Masas deben ser > 0")
            if distance_mpc <= 0:
                raise ValueError("Distancia debe ser > 0")
            if sampling_rate_hz <= 0:
                raise ValueError("Sampling rate debe ser > 0")
            if duration_seconds <= 0:
                raise ValueError("Duración debe ser > 0")
            
            # Obtener motor
            engine = self._get_engine()
            
            # Generar evento
            print(
                f"🔬 Sintetizando evento: "
                f"m1={mass1_solar_masses}M☉, "
                f"m2={mass2_solar_masses}M☉, "
                f"d={distance_mpc}Mpc, "
                f"teoría={theory_family}"
            )
            
            # Llamar al motor (implementación específica según SSTG)
            strain = engine.generate_signal(
                mass1=mass1_solar_masses,
                mass2=mass2_solar_masses,
                distance=distance_mpc,
                theory=theory_family,
                sampling_rate=sampling_rate_hz,
                duration=duration_seconds
            )
            
            print(f"✅ Evento sintético generado: {strain.shape}")
            return strain
        
        except Exception as e:
            raise ReportingException(
                f"Error sintetizando evento: {str(e)}"
            )
