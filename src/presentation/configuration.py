"""
Presentation Layer: Configuration
==================================

Constantes centralizadas para la capa de presentación.
Sin magic numbers, sin hardcoded strings.
"""

from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class CLIConfig:
    """Configuración para presentación CLI."""
    
    # Formatos de salida
    HEADER_WIDTH: int = 50
    DIVIDER_CHAR: str = "="
    SECTION_DIVIDER_CHAR: str = "-"
    SECTION_DIVIDER_WIDTH: int = 30
    
    # Proyectos
    PROJECT_NAME: str = "QNIM: Quantum Native Inverse Models for GW"
    PROJECT_SUBTITLE: str = "Framework de Inferencia Cuántica v1.0"
    
    # Precisiones
    PERCENTAGE_PRECISION: int = 2  # 2 decimales
    FLOAT_PRECISION: int = 3       # 3 decimales
    SCIENTIFIC_PRECISION: int = 2  # Para notación científica


@dataclass(frozen=True)
class VisualizationConfig:
    """Configuración para visualizaciones matplotlib/seaborn."""
    
    # Directorio de salida
    DEFAULT_OUTPUT_DIR: Path = Path("reports/figures")
    
    # Tamaños de figura
    FIGURE_SIZE_LARGE: tuple = (12, 8)    # Para gráficos complejos
    FIGURE_SIZE_NORMAL: tuple = (10, 6)   # Para gráficos estándar
    FIGURE_SIZE_SMALL: tuple = (8, 5)     # Para gráficos simples
    
    # Fuentes
    TITLE_FONTSIZE: int = 14
    AXIS_FONTSIZE: int = 12
    LABEL_FONTSIZE: int = 11
    LEGEND_FONTSIZE: int = 10
    
    # Resolución y calidad
    DPI_EXPORT: int = 150
    BBOX_INCHES: str = "tight"
    
    # Colores (paleta consistente con domain)
    COLOR_PRIMARY: str = "teal"           # Color principal (física cuántica)
    COLOR_SECONDARY: str = "viridis"      # Para heatmaps
    COLOR_GRID: str = "gray"
    GRID_LINESTYLE: str = "--"
    GRID_ALPHA: float = 0.6
    
    # Estilos seaborn
    SEABORN_STYLE: str = "white"
    
    # Gráficos específicos
    # Loss Curve
    LOSS_CURVE_LINEWIDTH: float = 2.0
    LOSS_CURVE_LABEL: str = "SPSA Loss"
    
    # Corner Plot (distribuciones)
    KDE_LEVELS: int = 10               # Contornos KDE
    HISTOGRAM_ALPHA: float = 0.4       # Transparencia histogramas
    HISTOGRAM_BINS: int = 15           # Número de bins
    
    # Confusion Matrix
    CONFUSION_MATRIX_STRING_FORMAT: str = "d"  # Integers
    CMAP_CONFUSION: str = "Blues"


# Instancias globales (singletons)
CLI_CONFIG = CLIConfig()
VIZ_CONFIG = VisualizationConfig()
