"""
src/application/ports/results_reporter_port.py
================================================
Puerto abstracto para la capa de resultados y visualización.

CAPA: Application — define el CONTRATO, no la implementación.
REGLA: Cero imports de matplotlib, seaborn, numpy. Solo typing y ABC.

La implementación concreta vive en:
  src/infrastructure/reporting/matplotlib_results_reporter.py

Este puerto garantiza que la capa Application nunca depende de
matplotlib, seaborn ni ninguna librería de visualización concreta.
Si mañana queremos usar Plotly en vez de matplotlib, solo
cambiamos el adaptador en infrastructure/, sin tocar nada más.

Autor: Óscar Boullosa Dapena — TFM QNIM, UNIR 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  DTOs DE RESULTADOS (Data Transfer Objects)
#  Viven en Application porque son los contratos de entrada/salida
#  de los casos de uso. NO contienen lógica de negocio.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QFIAdvantageDTO:
    """
    DTO: resultado de la comparativa QFI vs CFI para un parámetro físico.

    QFI = Quantum Fisher Information (límite cuántico de Cramér-Rao).
    CFI = Classical Fisher Information (límite clásico).

    Si ratio > 1 con significancia > 2σ → ventaja cuántica formal demostrada.
    """
    parameter_name: str
    f_quantum: float
    f_classical: float
    ratio: float
    ratio_uncertainty: float       # Incertidumbre bootstrap (n=1000)
    significance_sigma: float      # (F_Q - F_C) / σ_bootstrap

    @property
    def has_quantum_advantage(self) -> bool:
        return self.ratio > 1.0 and self.significance_sigma >= 2.0

    def to_latex_row(self) -> str:
        """Fila LaTeX lista para copiar al TFM (Tabla 5.X)."""
        return (
            f"${self.parameter_name}$ & "
            f"${self.f_quantum:.1f}\\pm{self.ratio_uncertainty * self.f_quantum:.1f}$ & "
            f"${self.f_classical:.1f}$ & "
            f"${self.ratio:.2f}\\pm{self.ratio_uncertainty:.2f}$ & "
            f"${self.significance_sigma:.1f}\\sigma$ \\\\"
        )


@dataclass
class VQCTrainingResultDTO:
    """
    DTO: resultado completo del entrenamiento del VQC cuántico.

    Captura toda la información necesaria para:
    - Las figuras del TFM (convergencia, accuracy vs SNR)
    - El análisis de barren plateaus
    - La comparativa sim vs hardware real
    """
    # Historia de entrenamiento
    loss_history: list[float] = field(default_factory=list)
    accuracy_val_history: list[float] = field(default_factory=list)

    # Métricas finales
    accuracy_sim: float = 0.0          # Accuracy en Aer (simulador)
    accuracy_real_no_zne: float = 0.0  # Accuracy en IBM real sin ZNE
    accuracy_real_zne: float = 0.0     # Accuracy en IBM real con ZNE

    # Metadata del entrenamiento
    n_epochs_converged: int = 0
    converged_early: bool = False
    total_time_s: float = 0.0
    n_qubits_used: int = 0             # Qubits realmente usados (≤ max backend)
    n_circuit_evaluations: int = 0
    backend_name: str = "aer"
    optimizer_used: str = "QNSPSA-EML-Feynman"
    feature_map_used: str = "ChebyshevFeatureMap"
    speedup_vs_spsa: float = 0.0

    # Confusion matrix (como lista para serialización JSON-safe)
    confusion_matrix_normalized: Optional[list[list[float]]] = None
    class_names: list[str] = field(default_factory=list)


@dataclass
class GW150914ReanalysisDTO:
    """
    DTO: resultado del re-análisis de GW150914 con QNIM.

    Encapsula la comparativa con GWTC-1 y los factores de Bayes
    para los 10 modelos de teoría gravitacional evaluados.
    """
    # Parámetros físicos extraídos por QNIM
    m1_msun: float = 0.0
    m2_msun: float = 0.0
    chi_eff: float = 0.0
    d_l_mpc: float = 0.0
    m_final_msun: float = 0.0
    chi_final: float = 0.0

    # Incertidumbres (68% CI bootstrap)
    m1_uncertainty: float = 0.0
    m2_uncertainty: float = 0.0
    chi_eff_uncertainty: float = 0.0
    d_l_uncertainty: float = 0.0

    # Consistencia con GWTC-1
    all_params_within_90pct_ci: bool = False

    # Factores de Bayes (escala de Jeffreys)
    # |log10(B)| < 0.5: anecdótico; 0.5-1.0: moderado; >1.0: fuerte
    bayes_factors: dict[str, float] = field(default_factory=dict)

    # Inferencia cosmológica (sirena estándar)
    h0_km_s_mpc: float = 0.0
    h0_ci_upper_68: float = 0.0
    h0_ci_lower_68: float = 0.0

    @property
    def is_gr_consistent(self) -> bool:
        """GR consistente si todos los |log10(B)| < 0.5 (anecdótico)."""
        return all(abs(v) < 0.5 for k, v in self.bayes_factors.items()
                   if k != "GR_pure")


@dataclass
class FullExperimentResultDTO:
    """
    DTO maestro: agrega TODOS los resultados del experimento QNIM.

    Este es el objeto que sale del HybridInferenceOrchestrator
    y que la capa de Presentation convierte en figuras y reportes.

    El flujo correcto es:
      SSTG (domain) → dataset HDF5
        → HybridOrchestrator (application) usa:
            - D-Wave (IQuantumOptimizerPort) → parámetros físicos
            - IBM VQC (IQuantumMLTrainerPort) → clasificación teorías
        → FullExperimentResultDTO
        → IResultsReporterPort → figuras + CSV + JSON
    """
    timestamp: str = ""

    # Resultados del VQC cuántico (rama IBM)
    vqc_training: Optional[VQCTrainingResultDTO] = None

    # Ventaja cuántica formal (QFI vs CFI)
    qfi_advantages: list[QFIAdvantageDTO] = field(default_factory=list)

    # Re-análisis de GW150914
    gw150914: Optional[GW150914ReanalysisDTO] = None

    # Accuracy vs SNR sweep
    accuracy_vs_snr: dict[int, float] = field(default_factory=dict)

    # Barren plateau analysis (Var[∂L/∂θ] por n_qubits)
    barren_plateau_variance: dict[int, float] = field(default_factory=dict)

    # Big-O benchmark
    bigO_benchmark: list[dict] = field(default_factory=list)

    # Metadatos del experimento
    dataset_n_events: int = 0
    dataset_n_classes: int = 0
    dataset_snr_mean: float = 0.0
    dataset_snr_std: float = 0.0
    dataset_physically_valid: bool = False


# ─────────────────────────────────────────────────────────────────────────────
#  PUERTO ABSTRACTO: IResultsReporterPort
# ─────────────────────────────────────────────────────────────────────────────


class IResultsReporterPort(ABC):
    """
    Contrato abstracto para el sistema de reporting de resultados.

    CAPA: Application.
    IMPLEMENTADO EN: src/infrastructure/reporting/matplotlib_results_reporter.py

    El orquestador llama a este puerto sin saber si la implementación
    usa matplotlib, plotly, seaborn o cualquier otra librería.

    Métodos del contrato:
    ─────────────────────
    generate_all_figures(result, output_dir) → dict[str, str]
        Genera las 7 figuras del TFM y retorna un dict {nombre: ruta_png}

    export_json_report(result, path) → str
        Exporta el resultado completo como JSON

    export_csv_metrics(result, path) → str
        Exporta las métricas tabulares como CSV (para incluir en el TFM)

    generate_latex_tables(result) → dict[str, str]
        Genera las tablas LaTeX listas para copiar al .tex
    """

    @abstractmethod
    def generate_all_figures(
        self,
        result: FullExperimentResultDTO,
        output_dir: str = "reports/figures",
    ) -> dict[str, str]:
        """
        Genera las 7 figuras del TFM.

        Figuras producidas:
          fig1_convergence.png       — Curvas pérdida + accuracy vs época
          fig2_confusion_matrix.png  — Confusion matrix 10×10 normalizada
          fig3_qfi_cfi.png           — QFI vs CFI con barras de error
          fig4_accuracy_snr.png      — Accuracy vs SNR: QNIM vs ResNet
          fig5_barren_plateaus.png   — Var[∂L/∂θ] vs n_qubits con/sin EML
          fig6_gw150914.png          — Re-análisis GW150914 + Bayes factors
          fig7_dashboard.png         — Dashboard global de resultados IBM

        Returns:
            dict {nombre_figura: ruta_absoluta_png}
        """

    @abstractmethod
    def export_json_report(
        self,
        result: FullExperimentResultDTO,
        path: str = "reports/full_results.json",
    ) -> str:
        """
        Serializa el FullExperimentResultDTO completo como JSON.
        Retorna la ruta del fichero guardado.
        """

    @abstractmethod
    def export_csv_metrics(
        self,
        result: FullExperimentResultDTO,
        path: str = "reports/results_summary.csv",
    ) -> str:
        """
        Exporta las métricas en formato CSV tabular.
        Retorna la ruta del fichero guardado.
        """

    @abstractmethod
    def generate_latex_tables(
        self,
        result: FullExperimentResultDTO,
    ) -> dict[str, str]:
        """
        Genera tablas LaTeX listas para copiar al TFM.

        Returns:
            dict con claves:
              "qfi_table"        — Tabla QFI vs CFI (Tabla 5.X del TFM)
              "convergence_table" — Tabla épocas/loss/accuracy
              "bigO_table"       — Tabla Big-O comparativa
              "gw150914_table"   — Tabla parámetros vs GWTC-1
              "bayes_table"      — Tabla factores de Bayes
        """
