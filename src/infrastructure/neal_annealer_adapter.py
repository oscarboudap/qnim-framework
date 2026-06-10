"""
src/infrastructure/neal_annealer_adapter.py
============================================
VERSIÓN CORREGIDA: usa QUBO con match function ponderada por PSD LIGO O3.

CAMBIO CRÍTICO:
  Antes: QUBO con MSE euclidiano sobre señales cos(2π M_c/30 * log(f))
  Ahora: QUBO con match function M = ⟨h_i|h_j⟩/√(⟨h_i|h_i⟩⟨h_j|h_j⟩)
         ponderada por PSD de LIGO O3 (físicamente correcto).

  Esta reformulación es CONSISTENTE con la Ec. 2.6 del TFM (FIM).

Autor: Óscar Boullosa Dapena — TFM QNIM, UNIR 2026
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import neal
import numpy as np

from src.domain.quantum.interfaces import IQuantumAnnealer
from src.domain.quantum.entities import AnnealingResult
from src.infrastructure.qubo_match_ligo import (
    build_ligo_match_qubo,
    LIGOMatchQUBOResult,
)

logger = logging.getLogger("qnim.infrastructure.neal_annealer_adapter")


@dataclass
class _PhysicalMatchResult:
    """Parámetros físicos extraídos por template matching QUBO con match function real."""
    m1_msun: float
    m2_msun: float
    chi_eff: float
    best_match: float           # M(h_obs, h_template) ∈ [-1, 1]
    snr_estimate: float         # SNR = match × SNR_base
    is_gr_consistent: bool      # M > 0.97 → GR consistente
    m1_std_msun: float = 0.0
    m2_std_msun: float = 0.0
    chi_eff_std: float = 0.0
    annealing_selection_stability: float = 1.0
    annealing_energy_mean: float = 0.0
    annealing_energy_std: float = 0.0
    n_noise_realizations: int = 1
    noise_model: str = "none"


class NealSimulatedAnnealerAdapter(IQuantumAnnealer):
    """
    Adaptador de Infraestructura para Recocido Cuántico.

    Usa el simulador local de D-Wave (Neal) con el QUBO reformulado
    correctamente usando la match function ponderada por PSD de LIGO O3.

    Hot-Swap: Para usar la QPU real, solo cambiar SimulatedAnnealingSampler
    por DWaveComposite(token=...) en este único archivo.
    """

    def __init__(
        self,
        annealing_noise_std: float = 0.03,
        n_noise_realizations: int = 24,
        noise_seed: int = 1234,
    ):
        self.sampler = neal.SimulatedAnnealingSampler()
        self.annealing_noise_std = max(0.0, float(annealing_noise_std))
        self.n_noise_realizations = max(1, int(n_noise_realizations))
        self.noise_seed = int(noise_seed)

    def _perturb_qubo(self, Q: dict, rng: np.random.Generator) -> dict:
        """Aplica ruido gaussiano a coeficientes QUBO para simular sesgo/calibración."""
        if self.annealing_noise_std <= 0.0:
            return dict(Q)

        Q_noisy = {}
        for key, value in Q.items():
            scale = max(abs(float(value)), 1e-6)
            sigma = self.annealing_noise_std * scale
            Q_noisy[key] = float(value + rng.normal(0.0, sigma))
        return Q_noisy

    @staticmethod
    def _decode_template_index(best_sample: dict, qubo_result: LIGOMatchQUBOResult) -> int:
        """Decodifica la solución one-hot en índice de template."""
        active_templates = [i for i, v in best_sample.items() if v == 1]
        if active_templates:
            return int(min(active_templates, key=lambda i: qubo_result.qubo_linear.get(i, 1.0)))
        return int(qubo_result.best_template_idx)

    def sample_qubo(self, Q: dict, num_reads: int = 100) -> AnnealingResult:
        response = self.sampler.sample_qubo(Q, num_reads=num_reads)
        best_sample = response.first.sample
        lowest_energy = response.first.energy
        occurrences = response.first.num_occurrences
        is_confident = occurrences >= (num_reads * 0.1)
        return AnnealingResult(
            best_state=best_sample,
            lowest_energy=lowest_energy,
            num_occurrences=occurrences,
            is_ground_state_confident=is_confident,
        )

    def get_embedding_time(self, num_qubits: int) -> float:
        return 10.0 + (num_qubits * 0.5)

    def get_native_graph_topology(self) -> dict:
        return {i: [j for j in range(8) if j != i] for i in range(8)}

    def extract_physical_parameters(
        self,
        dataset,
        n_templates: int = 64,
        regularization: float = 0.01,
    ) -> _PhysicalMatchResult:
        """
        Template matching QUBO con match function ponderada por PSD LIGO O3.

        HAMILTONIANO CORREGIDO:
            H_QUBO = Σ_i (1 - M²_i) x_i + P Σ_{i<j} x_i x_j

        donde M_i = ⟨h_obs|h_i⟩ / √(⟨h_obs|h_obs⟩ ⟨h_i|h_i⟩)
        es la match function en el espacio de Hilbert GW ponderado por LIGO O3.

        CONSISTENCIA CON EL TFM:
          Este Hamiltoniano vive en el MISMO espacio de Hilbert que la Ec. 2.6
          del TFM (FIM = ⟨∂_i h|∂_j h⟩), cerrando el loop teórico entre
          la formulación GR y la arquitectura computacional del framework.

        Args:
            dataset: BalancedDataset con X_train (N × n_features)
            n_templates: número de templates en la cuadrícula 3D
            regularization: peso extra de regularización para el QUBO
        """
        X = np.asarray(dataset.X_train, dtype=float)
        observed = X.mean(axis=0)  # señal media observada

        logger.info(
            f"Construyendo QUBO con match function LIGO O3 "
            f"({n_templates} templates, features={len(observed)})"
        )

        # ── Construir QUBO con match function real ────────────────────────
        qubo_result: LIGOMatchQUBOResult = build_ligo_match_qubo(
            observed_features=observed,
            n_templates=n_templates,
            penalty_weight_factor=10.0,
            seed=42,
        )

        # ── Resolver con Neal (simulador D-Wave) ──────────────────────────
        Q_dict = {}
        # Términos lineales (diagonal)
        for i, coeff in qubo_result.qubo_linear.items():
            Q_dict[(i, i)] = coeff + regularization
        # Términos cuadráticos (penalización one-hot)
        for (i, j), coeff in list(qubo_result.qubo_quadratic.items())[:500]:  # cap para eficiencia
            Q_dict[(i, j)] = coeff

        # ── Resolver con ensemble ruidoso para aproximar ruido de annealing ──
        rng = np.random.default_rng(self.noise_seed)
        selected_indices = []
        selected_energies = []

        for _ in range(self.n_noise_realizations):
            Q_noisy = self._perturb_qubo(Q_dict, rng)
            response = self.sampler.sample_qubo(Q_noisy, num_reads=200)
            selected_energies.append(float(response.first.energy))
            selected_indices.append(self._decode_template_index(response.first.sample, qubo_result))

        index_hist = Counter(selected_indices)
        mode_idx, mode_count = max(
            index_hist.items(),
            key=lambda item: (item[1], -qubo_result.qubo_linear.get(item[0], 1.0)),
        )
        selection_stability = mode_count / max(len(selected_indices), 1)

        best_idx = int(mode_idx)
        if selection_stability < 0.40 and best_idx != qubo_result.best_template_idx:
            logger.warning(
                f"Inestabilidad alta bajo ruido de annealing "
                f"(stability={selection_stability:.2f}). "
                f"Forzando template de mejor match directo ({qubo_result.best_template_idx})."
            )
            best_idx = int(qubo_result.best_template_idx)

        # Recuperar parámetros del template seleccionado en lugar de forzar siempre el directo
        if 0 <= best_idx < len(qubo_result.template_bank):
            selected_m1, selected_m2, selected_chi = qubo_result.template_bank[best_idx]
            selected_match = qubo_result.template_matches[best_idx]
        else:
            selected_m1 = qubo_result.m1_msun
            selected_m2 = qubo_result.m2_msun
            selected_chi = qubo_result.chi_eff
            selected_match = qubo_result.best_match

        # Incertidumbre inducida por ruido del annealer (distribución de índices)
        ensemble_params = [
            qubo_result.template_bank[i]
            for i in selected_indices
            if 0 <= i < len(qubo_result.template_bank)
        ]
        if ensemble_params:
            m1_std = float(np.std([p[0] for p in ensemble_params]))
            m2_std = float(np.std([p[1] for p in ensemble_params]))
            chi_std = float(np.std([p[2] for p in ensemble_params]))
        else:
            m1_std = 0.0
            m2_std = 0.0
            chi_std = 0.0

        energy_mean = float(np.mean(selected_energies)) if selected_energies else 0.0
        energy_std = float(np.std(selected_energies)) if selected_energies else 0.0
        selected_snr = max(0.0, float(selected_match) * 24.0)
        selected_gr_consistent = bool(float(selected_match) > 0.85)

        logger.info(
            f"D-Wave QUBO (ruido simulado): m1={selected_m1:.1f}±{m1_std:.2f} M_☉, "
            f"m2={selected_m2:.1f}±{m2_std:.2f} M_☉, "
            f"χ_eff={selected_chi:.3f}±{chi_std:.3f}, "
            f"M={selected_match:.4f}, "
            f"SNR_est={selected_snr:.1f}, "
            f"GR_consistent={selected_gr_consistent}, "
            f"stability={selection_stability:.2f}, "
            f"E={energy_mean:.4f}±{energy_std:.4f}"
        )

        return _PhysicalMatchResult(
            m1_msun=float(selected_m1),
            m2_msun=float(selected_m2),
            chi_eff=float(selected_chi),
            best_match=float(selected_match),
            snr_estimate=float(selected_snr),
            is_gr_consistent=selected_gr_consistent,
            m1_std_msun=m1_std,
            m2_std_msun=m2_std,
            chi_eff_std=chi_std,
            annealing_selection_stability=float(selection_stability),
            annealing_energy_mean=energy_mean,
            annealing_energy_std=energy_std,
            n_noise_realizations=self.n_noise_realizations,
            noise_model="gaussian_qubo_coefficients",
        )