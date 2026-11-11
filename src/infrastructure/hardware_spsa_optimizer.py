"""Budgeted SPSA adaptation for a caller-provided real-hardware evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class HardwareEvaluation:
    loss: float
    job_ids: tuple[str, ...]
    shots: int
    pubs: int
    qpu_seconds: Optional[float]


@dataclass(frozen=True)
class HardwareSPSAConfig:
    steps: int = 20
    learning_rate: float = 0.01
    perturbation: float = 0.05
    alpha: float = 0.602
    gamma: float = 0.101
    seed: int = 42
    reference_every: int = 5
    reference_shots: int = 128

    def __post_init__(self):
        if self.steps < 1:
            raise ValueError("steps debe ser positivo")
        if self.learning_rate <= 0 or self.perturbation <= 0:
            raise ValueError("learning_rate y perturbation deben ser positivos")
        if self.alpha < 0 or self.gamma < 0:
            raise ValueError("alpha y gamma no pueden ser negativos")
        if self.reference_every < 0:
            raise ValueError("reference_every no puede ser negativo")
        if self.reference_shots < 1:
            raise ValueError("reference_shots debe ser positivo")


@dataclass
class HardwareSPSAResult:
    parameters: np.ndarray
    loss_history: list[float] = field(default_factory=list)
    reference_history: list[dict] = field(default_factory=list)
    steps_completed: int = 0
    jobs: int = 0
    pubs: int = 0
    shots: int = 0
    qpu_seconds: Optional[float] = 0.0
    job_ids: list[str] = field(default_factory=list)


class HardwareSPSAExecutionError(RuntimeError):
    def __init__(self, message: str, parameters: np.ndarray, audit: HardwareSPSAResult):
        super().__init__(message)
        self.parameters = parameters.copy()
        self.audit = audit


HardwareEvaluator = Callable[[np.ndarray, np.ndarray, int], HardwareEvaluation]


class HardwareSPSAOptimizer:
    """Two-loss SPSA steps with fixed minibatches and auditable execution use."""

    def __init__(self, config: Optional[HardwareSPSAConfig] = None):
        self.config = config or HardwareSPSAConfig()

    @staticmethod
    def _validate_evaluation(result: HardwareEvaluation) -> HardwareEvaluation:
        if not isinstance(result, HardwareEvaluation):
            raise TypeError("evaluator debe devolver HardwareEvaluation")
        if not np.isfinite(result.loss):
            raise ValueError("La pérdida evaluada debe ser finita")
        if result.shots < 0 or result.pubs < 0:
            raise ValueError("El uso reportado no puede ser negativo")
        if result.qpu_seconds is not None:
            if result.qpu_seconds < 0 or not np.isfinite(result.qpu_seconds):
                raise ValueError("qpu_seconds debe ser finito y no negativo")
        return result

    @staticmethod
    def _record(result: HardwareSPSAResult, evaluation: HardwareEvaluation) -> None:
        result.jobs += len(evaluation.job_ids)
        result.pubs += evaluation.pubs
        result.shots += evaluation.shots
        if result.qpu_seconds is None or evaluation.qpu_seconds is None:
            result.qpu_seconds = None
        else:
            result.qpu_seconds += evaluation.qpu_seconds
        result.job_ids.extend(evaluation.job_ids)

    def optimize(
        self,
        initial_parameters: np.ndarray,
        n_samples: int,
        batch_size: int,
        shots: int,
        evaluator: HardwareEvaluator,
        reference_evaluator: Optional[HardwareEvaluator] = None,
        reference_indices: Optional[np.ndarray] = None,
    ) -> HardwareSPSAResult:
        parameters = np.asarray(initial_parameters, dtype=float).copy()
        if parameters.ndim != 1 or parameters.size == 0 or not np.isfinite(parameters).all():
            raise ValueError("initial_parameters debe ser un vector 1D finito y no vacío")
        if n_samples < 1 or not 1 <= batch_size <= n_samples:
            raise ValueError("Se requiere 1 <= batch_size <= n_samples")
        if shots < 1:
            raise ValueError("shots debe ser positivo")
        if self.config.reference_every and reference_evaluator is not None:
            if reference_indices is None or np.asarray(reference_indices).size == 0:
                raise ValueError("reference_indices es obligatorio para evaluación de referencia")
        elif reference_evaluator is None and reference_indices is not None:
            raise ValueError("reference_indices requiere reference_evaluator")

        rng = np.random.default_rng(self.config.seed)
        audit = HardwareSPSAResult(parameters=parameters.copy())

        def evaluate(callable_: HardwareEvaluator, values, indices, shots_for_eval):
            try:
                evaluation = self._validate_evaluation(
                    callable_(values.copy(), indices.copy(), shots_for_eval)
                )
            except Exception as exc:
                partial = getattr(exc, "hardware_execution", None)
                if partial is not None:
                    audit.jobs += len(partial.get("job_ids", ()))
                    audit.pubs += partial.get("pubs", 0)
                    audit.shots += partial.get("shots", 0)
                    if audit.qpu_seconds is not None:
                        partial_seconds = partial.get("qpu_seconds")
                        audit.qpu_seconds = (
                            None
                            if partial_seconds is None
                            else audit.qpu_seconds + partial_seconds
                        )
                raise HardwareSPSAExecutionError(
                    f"Fallo de evaluación hardware: {exc}", parameters, audit
                ) from exc
            self._record(audit, evaluation)
            return evaluation

        for step in range(self.config.steps):
            indices = rng.choice(n_samples, size=batch_size, replace=False)
            delta = rng.choice(np.array([-1.0, 1.0]), size=parameters.size)
            c_k = self.config.perturbation / ((step + 1) ** self.config.gamma)
            a_k = self.config.learning_rate / ((step + 1) ** self.config.alpha)

            loss_plus = evaluate(evaluator, parameters + c_k * delta, indices, shots).loss
            loss_minus = evaluate(evaluator, parameters - c_k * delta, indices, shots).loss
            gradient = ((loss_plus - loss_minus) / (2.0 * c_k)) * delta
            proposed = parameters - a_k * gradient

            reference = None
            should_reference = (
                reference_evaluator is not None
                and self.config.reference_every > 0
                and (step + 1) % self.config.reference_every == 0
            )
            if should_reference:
                fixed_reference_indices = np.asarray(reference_indices, dtype=int)
                if np.any(fixed_reference_indices < 0) or np.any(fixed_reference_indices >= n_samples):
                    raise ValueError("reference_indices contiene índices fuera del dataset")
                reference_eval = evaluate(
                    reference_evaluator,
                    proposed,
                    fixed_reference_indices,
                    self.config.reference_shots,
                )
                reference = {"step": step + 1, "loss": reference_eval.loss}
                audit.reference_history.append(reference)

            parameters = proposed
            audit.parameters = parameters.copy()
            audit.loss_history.append(float((loss_plus + loss_minus) / 2.0))
            audit.steps_completed = step + 1

        return audit
