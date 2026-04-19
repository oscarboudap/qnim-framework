"""
Presentation Layer: Training Visualization
===========================================

Presentador para visualizar resultados de entrenamiento y análisis.
Encapsula toda la lógica de matplotlib/seaborn.
"""

from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.application.dto import ConfusionMatrixData
from src.presentation.configuration import VIZ_CONFIG
from src.presentation.exceptions import VisualizationException


class TrainingVisualizationPresenter:
    """
    Presentador para visualizar resultados de entrenamiento.
    
    Encapsula matplotlib/seaborn. Si queremos cambiar backend
    (a Plotly, Bokeh, etc) solo editamos aquí.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Inicializa presentador de visualización.
        
        Args:
            output_dir: Directorio de salida (default: reports/figures)
        """
        self.output_dir = Path(output_dir or VIZ_CONFIG.DEFAULT_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_loss_curve(self,
                                  loss_history: List[float],
                                  output_filename: Optional[str] = None) -> str:
        """
        Genera curva de aprendizaje (Loss Curve).
        
        Args:
            loss_history: Historial de pérdida [n_iterations]
            output_filename: Nombre archivo (default: loss_curve.png)
        
        Returns:
            Ruta completa del archivo PNG guardado
        
        Raises:
            VisualizationException: Si validación falla o matplotlib error
        """
        try:
            # Validación
            self._validate_loss_history(loss_history)
            
            # Filename
            if output_filename is None:
                output_filename = "loss_curve.png"
            output_path = self.output_dir / output_filename
            
            # Plot
            fig, ax = plt.subplots(figsize=VIZ_CONFIG.FIGURE_SIZE_NORMAL)
            ax.plot(loss_history,
                   label=VIZ_CONFIG.LOSS_CURVE_LABEL,
                   color=VIZ_CONFIG.COLOR_PRIMARY,
                   lw=VIZ_CONFIG.LOSS_CURVE_LINEWIDTH)
            
            ax.set_title('Evolución del Aprendizaje del VQC (12 Cúbits)',
                        fontsize=VIZ_CONFIG.TITLE_FONTSIZE)
            ax.set_xlabel('Iteraciones (Evaluaciones)',
                         fontsize=VIZ_CONFIG.AXIS_FONTSIZE)
            ax.set_ylabel('Función de Coste (Binary Cross-Entropy)',
                         fontsize=VIZ_CONFIG.AXIS_FONTSIZE)
            
            ax.grid(True,
                   linestyle=VIZ_CONFIG.GRID_LINESTYLE,
                   alpha=VIZ_CONFIG.GRID_ALPHA)
            ax.legend()
            
            # Save
            fig.savefig(output_path,
                       dpi=VIZ_CONFIG.DPI_EXPORT,
                       bbox_inches=VIZ_CONFIG.BBOX_INCHES)
            plt.close(fig)
            
            print(f"📊 Curva de pérdida guardada: {output_path}")
            return str(output_path)
        
        except VisualizationException:
            raise
        except Exception as e:
            raise VisualizationException(
                f"Error generando curva de pérdida: {str(e)}"
            )
    
    def plot_corner_distribution(self,
                                 mass_samples: np.ndarray,
                                 spin_samples: np.ndarray,
                                 output_filename: Optional[str] = None) -> str:
        """
        Genera Corner Plot para distribuciones de D-Wave.
        
        Args:
            mass_samples: Muestras de masa [n_samples]
            spin_samples: Muestras de spin [n_samples]
            output_filename: Nombre archivo (default: corner_plot.png)
        
        Returns:
            Ruta completa del archivo PNG guardado
        
        Raises:
            VisualizationException: Si validación falla o matplotlib error
        """
        try:
            # Validación
            self._validate_samples(mass_samples, spin_samples)
            
            # Filename
            if output_filename is None:
                output_filename = "corner_plot.png"
            output_path = self.output_dir / output_filename
            
            # Plot
            sns.set_theme(style=VIZ_CONFIG.SEABORN_STYLE)
            g = sns.JointGrid(x=mass_samples, y=spin_samples, space=0)
            
            g.plot_joint(sns.kdeplot,
                        fill=True,
                        thresh=0,
                        levels=VIZ_CONFIG.KDE_LEVELS,
                        cmap=VIZ_CONFIG.COLOR_SECONDARY)
            
            g.plot_marginals(sns.histplot,
                            color=VIZ_CONFIG.COLOR_PRIMARY,
                            alpha=VIZ_CONFIG.HISTOGRAM_ALPHA,
                            bins=VIZ_CONFIG.HISTOGRAM_BINS)
            
            g.set_axis_labels('Masa ($M_{\\odot}$)',
                             'Espín ($a^*$)',
                             fontsize=VIZ_CONFIG.AXIS_FONTSIZE)
            
            # Save
            plt.savefig(output_path,
                       dpi=VIZ_CONFIG.DPI_EXPORT,
                       bbox_inches=VIZ_CONFIG.BBOX_INCHES)
            plt.close()
            
            print(f"🌌 Corner Plot guardado: {output_path}")
            return str(output_path)
        
        except VisualizationException:
            raise
        except Exception as e:
            raise VisualizationException(
                f"Error generando Corner Plot: {str(e)}"
            )
    
    def plot_confusion_matrix(self,
                             cm_data: ConfusionMatrixData,
                             output_filename: Optional[str] = None) -> str:
        """
        Genera heatmap de matriz de confusión.
        
        Args:
            cm_data: Datos de matriz confusión (DTO)
            output_filename: Nombre archivo (default: confusion_matrix.png)
        
        Returns:
            Ruta completa del archivo PNG guardado
        
        Raises:
            VisualizationException: Si validación falla o matplotlib error
        """
        try:
            # Validación
            self._validate_confusion_matrix(cm_data)
            
            # Filename
            if output_filename is None:
                output_filename = "confusion_matrix.png"
            output_path = self.output_dir / output_filename
            
            # Construir matriz
            matrix = np.array([
                [cm_data.tp, cm_data.fp],
                [cm_data.fn, cm_data.tn]
            ])
            
            # Normalizar
            matrix_normalized = matrix.astype('float') / matrix.sum(axis=1, keepdims=True)
            
            # Plot
            fig, ax = plt.subplots(figsize=VIZ_CONFIG.FIGURE_SIZE_NORMAL)
            sns.heatmap(matrix_normalized,
                       annot=matrix,
                       fmt=VIZ_CONFIG.CONFUSION_MATRIX_STRING_FORMAT,
                       cmap=VIZ_CONFIG.CMAP_CONFUSION,
                       cbar_kws={'label': 'Proporción'},
                       xticklabels=['Negativo', 'Positivo'],
                       yticklabels=['Negativo', 'Positivo'],
                       ax=ax)
            
            ax.set_xlabel('Predicción', fontsize=VIZ_CONFIG.AXIS_FONTSIZE)
            ax.set_ylabel('Verdadero', fontsize=VIZ_CONFIG.AXIS_FONTSIZE)
            ax.set_title('Matriz de Confusión', fontsize=VIZ_CONFIG.TITLE_FONTSIZE)
            
            # Save
            fig.savefig(output_path,
                       dpi=VIZ_CONFIG.DPI_EXPORT,
                       bbox_inches=VIZ_CONFIG.BBOX_INCHES)
            plt.close(fig)
            
            print(f"🎯 Matriz de confusión guardada: {output_path}")
            return str(output_path)
        
        except VisualizationException:
            raise
        except Exception as e:
            raise VisualizationException(
                f"Error generando matriz confusión: {str(e)}"
            )
    
    # =========================================================================
    # VALIDACIÓN PRIVADA
    # =========================================================================
    
    @staticmethod
    def _validate_loss_history(loss_history: List[float]) -> None:
        """
        Valida que historial de pérdida sea válido.
        
        Raises:
            VisualizationException: Si validación falla
        """
        try:
            if not isinstance(loss_history, (list, np.ndarray)):
                raise TypeError(
                    f"loss_history debe ser List o np.ndarray, "
                    f"got {type(loss_history)}"
                )
            
            if len(loss_history) == 0:
                raise ValueError("loss_history no puede estar vacío")
            
            if not all(isinstance(x, (int, float)) for x in loss_history):
                raise ValueError(
                    "Todos los valores en loss_history deben ser numéricos"
                )
            
            if any(np.isnan(x) or np.isinf(x) for x in loss_history):
                raise ValueError(
                    "loss_history contiene NaN o infinito"
                )
        
        except (TypeError, ValueError) as e:
            raise VisualizationException(f"Validación loss_history falló: {str(e)}")
    
    @staticmethod
    def _validate_samples(mass_samples: np.ndarray, spin_samples: np.ndarray) -> None:
        """
        Valida que muestras sean válidas.
        
        Raises:
            VisualizationException: Si validación falla
        """
        try:
            if not isinstance(mass_samples, np.ndarray):
                raise TypeError("mass_samples debe ser np.ndarray")
            if not isinstance(spin_samples, np.ndarray):
                raise TypeError("spin_samples debe ser np.ndarray")
            
            if len(mass_samples) == 0 or len(spin_samples) == 0:
                raise ValueError("Muestras no pueden estar vacías")
            
            if len(mass_samples) != len(spin_samples):
                raise ValueError(
                    f"mass_samples y spin_samples deben tener mismo tamaño: "
                    f"{len(mass_samples)} != {len(spin_samples)}"
                )
            
            if any(np.isnan(mass_samples)) or any(np.isnan(spin_samples)):
                raise ValueError("Muestras contienen NaN")
        
        except (TypeError, ValueError) as e:
            raise VisualizationException(f"Validación muestras falló: {str(e)}")
    
    @staticmethod
    def _validate_confusion_matrix(cm: ConfusionMatrixData) -> None:
        """
        Valida que matriz confusión sea válida.
        
        Raises:
            VisualizationException: Si validación falla
        """
        try:
            if cm is None:
                raise ValueError("ConfusionMatrixData no puede ser None")
            
            if any(val < 0 for val in [cm.tp, cm.tn, cm.fp, cm.fn]):
                raise ValueError(
                    "Valores confusión deben ser non-negativos"
                )
            
            if cm.tp + cm.fp + cm.tn + cm.fn == 0:
                raise ValueError("Matriz confusión está vacía (todos ceros)")
        
        except (TypeError, ValueError) as e:
            raise VisualizationException(
                f"Validación matriz confusión falló: {str(e)}"
            )


# Funciones legacy para compatibilidad (deprecated)
def plot_training_results(loss_history, output_path="reports/figures/loss_curve.png"):
    """
    ⚠️ DEPRECATED: Usa TrainingVisualizationPresenter.plot_training_loss_curve()
    
    Legacy function kept for backward compatibility.
    """
    presenter = TrainingVisualizationPresenter()
    return presenter.plot_training_loss_curve(loss_history, Path(output_path).name)


def plot_corner_results(mass_samples, spin_samples, output_path="reports/figures/corner_plot.png"):
    """
    ⚠️ DEPRECATED: Usa TrainingVisualizationPresenter.plot_corner_distribution()
    
    Legacy function kept for backward compatibility.
    """
    presenter = TrainingVisualizationPresenter()
    return presenter.plot_corner_distribution(mass_samples, spin_samples, Path(output_path).name)


if __name__ == "__main__":
    # Datos de ejemplo para testing
    presenter = TrainingVisualizationPresenter()
    
    # Loss curve
    fake_loss = np.exp(-np.linspace(0, 5, 300)) + np.random.normal(0, 0.05, 300)
    presenter.plot_training_loss_curve(fake_loss)
    
    # Corner plot
    m_samples = np.random.normal(36.2, 1.5, 1000)
    s_samples = np.random.normal(0.0, 0.1, 1000)
    presenter.plot_corner_distribution(m_samples, s_samples)