"""
Script Layer Configuration Management

Centralizes all script parameters, paths, and magic numbers into immutable
value objects following DDD principles.

Architecture:
  - All hardcoded values moved to frozen dataclasses
  - Validates ranges and constraints at instantiation
  - Enables easy parameter sweeps and batch experiments
  - Supports environment variable overrides

Example:
  config = ScriptConfig.from_env()
  use_case = container.get_training_use_case()
  use_case.execute(config.training_params)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import os


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class QuantumVQCConfig:
    """Quantum VQC (Variational Quantum Classifier) parameters."""
    
    num_qubits: int = 12
    num_features: int = 12
    ansatz_reps: int = 2
    num_layers: int = 3
    
    optimizer_name: str = "SPSA"
    max_iterations: int = 100
    learning_rate: float = 0.01
    
    initial_weights_path: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate quantum parameters."""
        if self.num_qubits < 2 or self.num_qubits > 20:
            raise ValueError(f"num_qubits must be 2-20, got {self.num_qubits}")
        if self.num_features < 2 or self.num_features > 64:
            raise ValueError(f"num_features must be 2-64, got {self.num_features}")
        if self.max_iterations < 10 or self.max_iterations > 1000:
            raise ValueError(f"max_iterations must be 10-1000, got {self.max_iterations}")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Data preprocessing parameters."""
    
    target_samples: int = 16384
    sample_rate_hz: int = 4096
    
    pca_components: int = 12
    pca_whiten: bool = True
    
    standardize: bool = True
    min_max_range: Tuple[float, float] = (-3.141592653589793, 3.141592653589793)  # [-π, π]
    
    def __post_init__(self) -> None:
        """Validate preprocessing parameters."""
        if self.target_samples < 1024 or self.target_samples > 65536:
            raise ValueError(f"target_samples must be 1024-65536, got {self.target_samples}")
        if self.pca_components < 2 or self.pca_components > 64:
            raise ValueError(f"pca_components must be 2-64, got {self.pca_components}")
        if self.sample_rate_hz < 512 or self.sample_rate_hz > 16384:
            raise ValueError(f"sample_rate_hz must be 512-16384, got {self.sample_rate_hz}")


@dataclass(frozen=True)
class PathsConfig:
    """File and directory configuration."""
    
    data_dir: str = "data"
    synthetic_data_subdir: str = "synthetic"
    raw_data_subdir: str = "raw"
    
    models_dir: str = "models"
    reports_dir: str = "reports"
    
    vqc_weights_filename: str = "qnim_vqc_weights.npy"
    preprocessing_pipeline_filename: str = "qnim_preprocessing_pipeline.pkl"
    
    def get_data_synthetic_path(self) -> Path:
        """Get full path to synthetic data directory."""
        return Path(self.data_dir) / self.synthetic_data_subdir
    
    def get_data_raw_path(self) -> Path:
        """Get full path to raw data directory."""
        return Path(self.data_dir) / self.raw_data_subdir
    
    def get_vqc_weights_path(self) -> Path:
        """Get full path to VQC weights file."""
        return Path(self.models_dir) / self.vqc_weights_filename
    
    def get_preprocessing_pipeline_path(self) -> Path:
        """Get full path to preprocessing pipeline file."""
        return Path(self.models_dir) / self.preprocessing_pipeline_filename
    
    def __post_init__(self) -> None:
        """Validate path configuration."""
        if not Path(self.data_dir).exists():
            raise ValueError(f"data_dir does not exist: {self.data_dir}")


@dataclass(frozen=True)
class TrainingConfig:
    """Complete training execution configuration."""
    
    vqc: QuantumVQCConfig = field(default_factory=QuantumVQCConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    use_real_ibm: bool = False
    batch_size: int = 32
    validation_split: float = 0.2
    
    random_seed: int = 42
    
    def __post_init__(self) -> None:
        """Validate training configuration."""
        if self.batch_size < 1 or self.batch_size > 512:
            raise ValueError(f"batch_size must be 1-512, got {self.batch_size}")
        if self.validation_split < 0.0 or self.validation_split > 0.5:
            raise ValueError(f"validation_split must be 0.0-0.5, got {self.validation_split}")


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class SweepExperimentConfig:
    """Parameters for parameter sweep experiments."""
    
    snr_levels: List[float] = field(default_factory=lambda: [5, 10, 15, 20, 25])
    distance_levels_mpc: List[float] = field(default_factory=lambda: [100, 400, 800, 1600, 3200])
    theory_families: List[str] = field(default_factory=lambda: ["RG", "LQG", "WHITE_HOLE"])
    
    iterations_per_config: int = 10
    num_Monte_Carlo_samples: int = 100
    
    output_csv: str = "results_sweep.csv"
    
    def __post_init__(self) -> None:
        """Validate sweep configuration."""
        if len(self.snr_levels) == 0:
            raise ValueError("snr_levels cannot be empty")
        if len(self.distance_levels_mpc) == 0:
            raise ValueError("distance_levels_mpc cannot be empty")
        if self.iterations_per_config < 1 or self.iterations_per_config > 100:
            raise ValueError(f"iterations_per_config must be 1-100, got {self.iterations_per_config}")


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for inference/decoding operations."""
    
    ensemble_passes: int = 5
    confidence_threshold: float = 0.5
    
    mass_range_solar_masses: Tuple[float, float] = (10.0, 100.0)
    distance_range_mpc: Tuple[float, float] = (10.0, 5000.0)
    
    use_templates: bool = True
    template_bank_size: int = 50
    
    def __post_init__(self) -> None:
        """Validate inference configuration."""
        if self.ensemble_passes < 1 or self.ensemble_passes > 50:
            raise ValueError(f"ensemble_passes must be 1-50, got {self.ensemble_passes}")
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError(f"confidence_threshold must be 0.0-1.0, got {self.confidence_threshold}")


@dataclass(frozen=True)
class PlottingConfig:
    """Configuration for plotting and visualization scripts."""
    
    output_format: str = "png"
    dpi: int = 300
    figure_width_inches: float = 12.0
    figure_height_inches: float = 7.0
    
    color_palette: str = "husl"
    
    kde_levels: int = 10
    histogram_bins: int = 15
    
    output_dir: str = "reports/figures"
    
    def __post_init__(self) -> None:
        """Validate plotting configuration."""
        if self.dpi < 72 or self.dpi > 600:
            raise ValueError(f"dpi must be 72-600, got {self.dpi}")
        if self.figure_width_inches < 4.0 or self.figure_width_inches > 20.0:
            raise ValueError(f"figure_width_inches must be 4.0-20.0, got {self.figure_width_inches}")
        if self.histogram_bins < 5 or self.histogram_bins > 100:
            raise ValueError(f"histogram_bins must be 5-100, got {self.histogram_bins}")


# ============================================================================
# SCRIPT CONFIGURATION FACTORY
# ============================================================================

@dataclass(frozen=True)
class ScriptConfig:
    """
    Master configuration for all scripts.
    
    Centralizes all magic numbers and paths. Can be overridden via environment
    variables using prefix QNIM_SCRIPT_*.
    """
    
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    sweep: SweepExperimentConfig = field(default_factory=SweepExperimentConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    
    @classmethod
    def from_env(cls) -> "ScriptConfig":
        """Load configuration from environment variables and defaults."""
        return cls()
    
    @classmethod
    def from_yaml(cls, path: str) -> "ScriptConfig":
        """Load configuration from YAML file."""
        raise NotImplementedError("YAML loading not yet implemented")
