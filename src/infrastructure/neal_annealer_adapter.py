from __future__ import annotations

from dataclasses import dataclass

import neal
import numpy as np

from src.domain.quantum.interfaces import IQuantumAnnealer
from src.domain.quantum.entities import AnnealingResult


@dataclass
class _PhysicalMatchResult:
    """Parámetros físicos extraídos por template matching QUBO."""
    m1_msun: float
    m2_msun: float
    chi_eff: float


class NealSimulatedAnnealerAdapter(IQuantumAnnealer):
    """
    Adaptador de Infraestructura para el Recocido Cuántico.
    Usa el simulador local de D-Wave (Neal). 
    Diseño Hot-Swap: Para usar la QPU real, solo habría que cambiar el 'SimulatedAnnealingSampler' 
    por 'DWaveSampler(token=...)' en este único archivo.
    """
    
    def __init__(self):
        """Inicializa el sampler de Neal"""
        self.sampler = neal.SimulatedAnnealingSampler()
    
    def sample_qubo(self, Q: dict, num_reads: int = 100) -> AnnealingResult:
        # Ejecutamos la búsqueda del estado fundamental (Ground State)
        response = self.sampler.sample_qubo(Q, num_reads=num_reads)
        
        # Extraemos el mejor resultado
        best_sample = response.first.sample
        lowest_energy = response.first.energy
        occurrences = response.first.num_occurrences
        
        # Si el estado fundamental se ha encontrado muchas veces, tenemos alta confianza
        is_confident = occurrences >= (num_reads * 0.1)
        
        return AnnealingResult(
            best_state=best_sample,
            lowest_energy=lowest_energy,
            num_occurrences=occurrences,
            is_ground_state_confident=is_confident
        )
    
    def get_embedding_time(self, num_qubits: int) -> float:
        """
        Tiempo estimado para embedding en simulador (siempre rápido).
        
        Args:
            num_qubits: Número de qubits lógicos
            
        Returns:
            Tiempo estimado en microsegundos
        """
        # Simulador local: tiempo negligible, solo modelamos overhead
        return 10.0 + (num_qubits * 0.5)  # ~10-100 microsegundos
    
    def get_native_graph_topology(self) -> dict:
        """
        Retorna topología nativa (simulador no tiene restricciones).
        
        Returns:
            Dict completamente conectado para 8 qubits (como simplificación)
        """
        # Simulador: asumimos conectividad completa
        num_qubits = 8
        topology = {}
        for i in range(num_qubits):
            topology[i] = [j for j in range(num_qubits) if i != j]
        return topology

    def extract_physical_parameters(
        self,
        dataset,
        n_templates: int = 100,
        regularization: float = 0.01,
    ) -> _PhysicalMatchResult:
        """
        Template matching QUBO: extrae parámetros físicos de la señal GW.

        Formula el problema de selección de plantilla como QUBO one-hot:
          H = Σ_i MSE_i · x_i  +  P · Σ_{i<j} x_i · x_j
        donde P = 10 · max(MSE) + regularización garantiza que la
        restricción one-hot sea dominante (solo una plantilla activa).

        El recocido simulado (Neal) encuentra el índice i* que minimiza
        el MSE entre la señal observada y el banco de plantillas GR.

        Parámetros físicos de la cuadrícula (espacio GR estándar):
          m1 ∈ [20, 50] M_☉,  m2 ∈ [15, 45] M_☉,  χ_eff ∈ [-0.15, 0.15]
        Chirp mass: M_c = (m1·m2)^(3/5) / (m1+m2)^(1/5)
        Señal FFT simplificada: amplitud × cos(2π·M_c·f + χ_eff·π)

        Args:
            dataset: Objeto con atributo X_train (N × n_features) — 
                     primeras componentes FFT de la strain SSTG.
            n_templates: Número de plantillas en la cuadrícula paramétrica.
            regularization: Desplazamiento aditivo al peso de penalización 
                            (evita degeneraciones numéricas).

        Returns:
            _PhysicalMatchResult con m1_msun, m2_msun, chi_eff del mejor template.
        """
        X = np.asarray(dataset.X_train, dtype=float)
        n_features = X.shape[1]

        # Señal media observada (promedio sobre todos los eventos de entrenamiento)
        observed = X.mean(axis=0)  # shape (n_features,)
        obs_norm = np.linalg.norm(observed)

        # ── Cuadrícula de templates GR ───────────────────────────────────────
        # Se usan raíces cúbicas del número de templates para la cuadrícula 3-D
        side = max(2, int(np.ceil(n_templates ** (1.0 / 3))))
        m1_vals = np.linspace(20.0, 50.0, side)
        m2_vals = np.linspace(15.0, 45.0, side)
        chi_vals = np.linspace(-0.15, 0.15, side)

        templates: list[dict] = []
        freqs = np.arange(n_features, dtype=float) + 1.0  # frecuencias relativas

        for m1 in m1_vals:
            for m2 in m2_vals:
                for chi in chi_vals:
                    if len(templates) >= n_templates:
                        break
                    # Chirp mass (unidades solares)
                    M_c = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
                    # Forma de onda GR simplificada: amplitud decreciente × portadora
                    amplitude = np.exp(-0.1 * freqs / n_features)
                    phase = 2.0 * np.pi * (M_c / 30.0) * np.log1p(freqs) + chi * np.pi
                    strain = amplitude * np.cos(phase)
                    # Escalar a la norma del observable para comparabilidad
                    s_norm = np.linalg.norm(strain)
                    if s_norm > 0 and obs_norm > 0:
                        strain = strain * (obs_norm / s_norm)
                    templates.append({"m1": m1, "m2": m2, "chi_eff": chi, "strain": strain})
                if len(templates) >= n_templates:
                    break
            if len(templates) >= n_templates:
                break

        templates = templates[:n_templates]
        n_t = len(templates)

        # ── Construcción del QUBO ────────────────────────────────────────────
        mse_values = [
            float(np.mean((observed - t["strain"]) ** 2)) for t in templates
        ]
        max_mse = max(mse_values) if mse_values else 1.0
        penalty = 10.0 * max_mse + regularization

        Q: dict[tuple, float] = {}
        # Términos lineales (diagonal): coste de seleccionar plantilla i
        for i, mse in enumerate(mse_values):
            Q[(i, i)] = mse
        # Términos cuadráticos: penalización one-hot (como mucho una plantilla activa)
        for i in range(n_t):
            for j in range(i + 1, n_t):
                Q[(i, j)] = penalty

        # ── Annealing con Neal (D-Wave simulado) ────────────────────────────
        response = self.sampler.sample_qubo(Q, num_reads=200)
        best_sample = response.first.sample

        # Plantilla activa con menor MSE (desempate si hay varias activas)
        active = [i for i, v in best_sample.items() if v == 1]
        best_idx = (
            min(active, key=lambda i: mse_values[i])
            if active
            else int(np.argmin(mse_values))
        )

        best = templates[best_idx]
        return _PhysicalMatchResult(
            m1_msun=float(best["m1"]),
            m2_msun=float(best["m2"]),
            chi_eff=float(best["chi_eff"]),
        )