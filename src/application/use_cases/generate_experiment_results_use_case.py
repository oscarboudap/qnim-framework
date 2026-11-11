"""
generate_experiment_results_use_case.py  (v6 — añade use_ligo_pca)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from src.ibm_credentials import get_ibm_quantum_token


@dataclass
class ExperimentConfig:
    mode: str = "sim"
    max_iterations: int = 60
    batch_size: int = 64
    patience: int = 20
    learning_rate: float = 0.03
    n_qubits: int = 12
    pca_components: int = 12
    n_pca_fit: int = 64
    feature_map_reps: int = 2
    ansatz_reps: int = 2
    entanglement: str = "linear"
    shots: int = 1024
    reference_shots: int = 4096
    readout_hidden_size: int = 16
    readout_refit_every: int = 10
    max_classes: Optional[int] = None
    use_physics_generator: bool = False
    use_ligo_pca: bool = False
    physics_approximant: str = "IMRPhenomD"
    n_per_class: int = 80
    n_val_per_class: int = 20
    n_test_per_class: int = 100
    n_feynman_params: int = 4
    use_zne: bool = False
    train_on_hardware: bool = False
    backend_name: str = "ibm_fez"
    hardware_shots: int = 128
    hardware_batch_size: int = 4
    hardware_finetune_steps: int = 20
    n_hw_validation: int = 1300
    shots_validation: int = 2048

    def __post_init__(self):
        if self.mode not in ("sim", "ibm"):
            raise ValueError("mode debe ser 'sim' o 'ibm'")
        if self.readout_hidden_size < 4:
            raise ValueError("readout_hidden_size debe ser >= 4")
        if not 2 <= self.n_qubits <= 32:
            raise ValueError("n_qubits debe estar entre 2 y 32")
        if not 1 <= self.pca_components <= self.n_qubits:
            raise ValueError("pca_components debe estar entre 1 y n_qubits")
        if self.n_pca_fit < self.pca_components:
            raise ValueError("n_pca_fit debe ser >= pca_components")
        if self.feature_map_reps < 1 or self.feature_map_reps > 3:
            raise ValueError("feature_map_reps debe estar entre 1 y 3")
        if self.ansatz_reps < 1 or self.ansatz_reps > 4:
            raise ValueError("ansatz_reps debe estar entre 1 y 4")
        if self.entanglement not in {"linear", "circular", "full"}:
            raise ValueError("entanglement debe ser 'linear', 'circular' o 'full'")
        if self.use_ligo_pca and self.use_physics_generator:
            raise ValueError("--ligo-pca y --physics-generator son mutuamente excluyentes")
        if self.train_on_hardware and self.mode != "ibm":
            raise ValueError("El entrenamiento remoto requiere mode='ibm'")
        if self.train_on_hardware and self.hardware_finetune_steps < 1:
            raise ValueError("hardware_finetune_steps debe ser positivo")


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
                n_components=cfg.pca_components,
                n_pca_fit=cfg.n_pca_fit,
                approximant=cfg.physics_approximant,
                max_classes=cfg.max_classes,
            )
            return adapter.generate_balanced_dataset(
                n_per_class=cfg.n_per_class,
                n_val_per_class=cfg.n_val_per_class,
                n_test_per_class=cfg.n_test_per_class,
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
        cfg = self.config
        token = get_ibm_quantum_token()
        if cfg.mode == "ibm" and not token:
            raise RuntimeError("IBM_QUANTUM_TOKEN no está configurado")

        from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer

        return QiskitVQCTrainer(
            mode=cfg.mode,
            token=token,
            backend_name=cfg.backend_name,
            train_on_hardware=cfg.train_on_hardware,
            hardware_shots=cfg.hardware_shots,
            hardware_batch_size=cfg.hardware_batch_size,
            hardware_finetune_steps=cfg.hardware_finetune_steps,
            n_hw_validation=cfg.n_hw_validation,
            shots_validation=cfg.shots_validation,
            feature_map_reps=cfg.feature_map_reps,
            ansatz_reps=cfg.ansatz_reps,
            entanglement=cfg.entanglement,
            patience=cfg.patience,
            learning_rate=cfg.learning_rate,
            readout_hidden_size=cfg.readout_hidden_size,
            readout_refit_every=cfg.readout_refit_every,
        )

    def execute(self) -> ExperimentResult:
        cfg     = self.config
        trainer = self._build_vqc_trainer()
        dataset = self._build_dataset()
        result  = trainer.train_and_evaluate(
            dataset, n_qubits=cfg.n_qubits, shots=cfg.shots,
            n_features=cfg.pca_components,
            max_iterations=cfg.max_iterations,
            use_real_hardware=cfg.mode == "ibm",
            hardware_finetune_steps=(
                cfg.hardware_finetune_steps if cfg.train_on_hardware else None
            ),
            use_zne=cfg.use_zne,
            n_feynman_params=cfg.n_feynman_params,
        )

        accuracy_ibm = None
        if cfg.mode == "ibm":
            accuracy_ibm = (
                result.accuracy_real_zne
                if cfg.use_zne
                else result.accuracy_real_no_zne
            )

        return ExperimentResult(
            accuracy_sim=result.accuracy_sim,
            accuracy_ibm=accuracy_ibm,
            final_loss=result.loss_history[-1] if result.loss_history else float("nan"),
            n_epochs=result.n_epochs,
            loss_history=result.loss_history,
            confusion_matrix=result.confusion_matrix,
            metadata={"use_ligo_pca": cfg.use_ligo_pca,
                      "approximant": cfg.physics_approximant,
                      "n_qubits": cfg.n_qubits,
                      "pca_components": cfg.pca_components,
                      "ibm_validation": getattr(trainer, "last_ibm_validation", None),
                      "hardware_spsa": getattr(trainer, "last_hardware_spsa", None),
                      "feature_map_reps": cfg.feature_map_reps,
                      "ansatz_reps": cfg.ansatz_reps,
                      "entanglement": cfg.entanglement},
        )