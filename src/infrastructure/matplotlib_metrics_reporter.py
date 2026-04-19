"""
Infrastructure: Matplotlib Metrics Reporter Adapter
===================================================

Adaptador que implementa IMetricsReporterPort usando Matplotlib/Seaborn.

Métodos:
    - report_confusion_matrix(): Genera heatmap de matriz de confusión
    - report_inference_trace(): Genera reporte markdown de inferencia
    
Encapsula toda la visualización (matplotlib, seaborn) fuera de layers
de aplicación/domain.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns

from src.application.dto import ConfusionMatrixData
from src.application.ports import IMetricsReporterPort
from src.infrastructure.exceptions import ReportingException


class MatplotlibMetricsReporter(IMetricsReporterPort):
    """
    Reportar métricas usando Matplotlib + Seaborn.
    
    Implements:
        - report_confusion_matrix(): Matriz de confusión visual
        - report_inference_trace(): Trace de inferencia como markdown
    
    Encapsulation:
        Todo código de matplotlib está aquí.
        Application solo llama report_X() y get_path().
    """
    
    def __init__(self, output_dir: str = "reports/figures"):
        """
        Args:
            output_dir: Directorio donde guardar figuras
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def report_confusion_matrix(self,
                                cm_data: ConfusionMatrixData,
                                output_filename: Optional[str] = None) -> str:
        """
        Genera heatmap de matriz de confusión.
        
        Args:
            cm_data: ConfusionMatrixData con TP, TN, FP, FN
            output_filename: Nombre de archivo (default: timestamp-based)
        
        Returns:
            Ruta completa del archivo .png guardado
        
        Raises:
            ReportingException: Si falla la visualización o escritura
        """
        try:
            # Generar filename si no existe
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"confusion_matrix_{timestamp}.png"
            
            output_path = self.output_dir / output_filename
            
            # Construir matriz 2x2
            matrix = np.array([
                [cm_data.tp, cm_data.fp],  # True Positive, False Positive
                [cm_data.fn, cm_data.tn]   # False Negative, True Negative
            ])
            
            # Normalizar para ver proporciones
            matrix_normalized = matrix.astype('float') / matrix.sum(axis=1, keepdims=True)
            
            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                matrix_normalized,
                annot=matrix,  # Mostrar números absolutos
                fmt='d',
                cmap='Blues',
                cbar_kws={'label': 'Proporción'},
                xticklabels=['Negativo', 'Positivo'],
                yticklabels=['Negativo', 'Positivo'],
                ax=ax
            )
            
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Verdadero')
            ax.set_title('Matriz de Confusión')
            
            # Guardar
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"📊 Matriz de confusión guardada: {output_path}")
            return str(output_path)
        
        except Exception as e:
            raise ReportingException(
                f"Error generando matriz de confusión: {str(e)}"
            )
    
    def report_inference_trace(self,
                               event_id: str,
                               classic_parameters: dict,
                               quantum_results: dict,
                               execution_time_seconds: float,
                               output_filename: Optional[str] = None) -> str:
        """
        Genera reporte markdown de una inferencia completa.
        
        Args:
            event_id: ID único del evento
            classic_parameters: Dict con parámetros clásicos del D-Wave branch
            quantum_results: Dict con resultados del IBM branch
            execution_time_seconds: Tiempo total de ejecución
            output_filename: Nombre del archivo markdown (default: timestamp-based)
        
        Returns:
            Ruta completa del archivo .md guardado
        
        Raises:
            ReportingException: Si falla la generación o escritura
        """
        try:
            # Generar filename
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"inference_trace_{event_id}_{timestamp}.md"
            
            output_path = self.output_dir / output_filename
            
            # Generar contenido markdown
            report = f"""# Inference Trace Report
## Event ID: {event_id}

**Generado:** {datetime.now().isoformat()}
**Tiempo de Ejecución:** {execution_time_seconds:.3f}s

---

## 🌊 D-Wave Branch (Classical Parameters)

| Parámetro | Valor |
|-----------|-------|
| Masa 1 (M☉) | {classic_parameters.get('m1', 'N/A')} |
| Masa 2 (M☉) | {classic_parameters.get('m2', 'N/A')} |
| Distancia (Mpc) | {classic_parameters.get('distance', 'N/A')} |
| Spin 1 | {classic_parameters.get('spin1', 'N/A')} |
| Spin 2 | {classic_parameters.get('spin2', 'N/A')} |
| Energía Orbitall (QUBO) | {classic_parameters.get('orbital_energy', 'N/A')} |
| Temperatura (K) | {classic_parameters.get('temperature', 'N/A')} |

---

## 🎯 IBM VQC Branch (Quantum Classification)

| Métrica | Valor |
|---------|-------|
| Teoría Predicha | {quantum_results.get('theory', 'N/A')} |
| Confianza | {quantum_results.get('confidence', 'N/A')} |
| Firmas Beyond-GR | {quantum_results.get('beyond_gr_signatures', 'N/A')} |
| Capas Detectadas | {quantum_results.get('layers_detected', 'N/A')} |

---

## 📊 Resumen

- **Clasificación:** {quantum_results.get('classification', 'N/A')}
- **Veredicto:** {quantum_results.get('verdict', 'INCONCLUSO')}
- **Significancia Metrológica:** {quantum_results.get('metrological_significance', 'N/A')}

---

*Reporte generado automáticamente por QNIM Infrastructure*
"""
            
            # Guardar archivo
            output_path.write_text(report, encoding='utf-8')
            print(f"📝 Reporte de inferencia guardado: {output_path}")
            return str(output_path)
        
        except Exception as e:
            raise ReportingException(
                f"Error generando reporte de inferencia: {str(e)}"
            )
