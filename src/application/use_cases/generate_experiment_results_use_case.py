"""
generate_experiment_results_use_case.py  (v6 — añade use_ligo_pca)
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentConfig:
    mode: str = "sim"
    max_iterations: int   = 60
    batch_size: int       = 64
    patience: int         = 20
    learning_rate: float  = 0.03
    ansatz_reps: int      = 2
    shots: int            = 1024
    reference_shots: int  = 4096
    readout_hidden_size: int  = 16
    readout_refit_every: int  = 10
    max_classes: Optional[int] = None
    # Generadores
    use_physics_generator: bool = False
    use_ligo_pca: bool          = False   # [v6] pipeline exacto del paper
    physics_approximant: str    = "IMRPhenomD"
    n_per_class: int            = 80
    n_val_per_class: int        = 20
    n_feynman_params: int = 4
    use_zne: bool         = False

    def __post_init__(self):
        if self.mode not in ("sim", "ibm"):
            raise ValueError(f"mode debe ser 'sim' o 'ibm'")
        if self.readout_hidden_size < 4:
            raise ValueError("readout_hidden_size debe ser >= 4")
        if self.use_ligo_pca and self.use_physics_generator:
            raise ValueError("--ligo-pca y --physics-generator son mutuamente excluyentes")


@dataclass
class ExperimentResult:
    accuracy_sim: float
    accuracy_ibm: Optional[float]
    final_loss: float
    n_epochs: int
    loss_history: list = field(default_factory=list)
    confusion_matrix: Optional[list] = None
    metadata: dict = field(default_factory=dict)


class GenerateExperimentResultsUseCase:

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _build_dataset(self):
        cfg = self.config

        if cfg.use_ligo_pca:
            from src.infrastructure.ligo_pca_adapter import LIGOPCAAdapter
            adapter = LIGOPCAAdapter(
                approximant=cfg.physics_approximant,
                max_classes=cfg.max_classes,
            )
            return adapter.generate_balanced_dataset(
                n_per_class=cfg.n_per_class,
                n_val_per_class=cfg.n_val_per_class,
                seed=42,
            )

        elif cfg.use_physics_generator:
            from src.infrastructure.physics_sstg_adapter import PhysicsSSTGAdapter
            adapter = PhysicsSSTGAdapter(
                approximant=cfg.physics_approximant,
                max_classes=cfg.max_classes,
            )
            return adapter.generate_balanced_dataset(
                n_per_class=cfg.n_per_class,
                n_val_per_class=cfg.n_val_per_class,
                seed=42,
            )

        else:
            from src.infrastructure.sstg_adapter import SSTGAdapter
            adapter = SSTGAdapter()
            return adapter.generate_balanced_dataset(
                n_per_class=cfg.n_per_class,
                n_val_per_class=cfg.n_val_per_class,
                target_snr_range=(8, 30),
                seed=42,
                max_classes=cfg.max_classes,
            )

    def _build_vqc_trainer(self):
        from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer
        cfg = self.config
        return QiskitVQCTrainer(
            mode=cfg.mode,
            ansatz_reps=cfg.ansatz_reps,
            patience=cfg.patience,
            learning_rate=cfg.learning_rate,
            readout_hidden_size=cfg.readout_hidden_size,
            readout_refit_every=cfg.readout_refit_every,
        )

    def execute(self) -> ExperimentResult:
        cfg     = self.config
        dataset = self._build_dataset()
        trainer = self._build_vqc_trainer()
        result  = trainer.train_and_evaluate(
            dataset, n_qubits=12, shots=cfg.shots,
            max_iterations=cfg.max_iterations,
            use_zne=cfg.use_zne,
            n_feynman_params=cfg.n_feynman_params,
        )

        accuracy_ibm = None
        if cfg.mode == "ibm":
            token = os.environ.get("IBM_QUANTUM_TOKEN", "")
            if token:
                try:
                    final_w = getattr(result, 'final_weights', None)
                    if final_w is not None:
                        _, accuracy_ibm = trainer._validate_on_ibm(
                            weights=final_w,
                            dataset=dataset,
                            n_qubits=12,
                            use_zne=cfg.use_zne,
                        )
                    else:
                        print('  ⚠️  Sin pesos finales para validar en IBM.')
                except Exception as exc:
                    print(f"  ⚠️  Validación IBM falló: {exc}")
            else:
                print("  ⚠️  IBM_QUANTUM_TOKEN no exportado.")
                print("       Ejecuta: export IBM_QUANTUM_TOKEN='tu_token'")

        return ExperimentResult(
            accuracy_sim=result.accuracy_sim,
            accuracy_ibm=accuracy_ibm,
            final_loss=result.loss_history[-1] if result.loss_history else float("nan"),
            n_epochs=result.n_epochs,
            loss_history=result.loss_history,
            confusion_matrix=result.confusion_matrix,
            metadata={"use_ligo_pca": cfg.use_ligo_pca,
                      "approximant": cfg.physics_approximant},
        )