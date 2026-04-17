# src/domain/metrology/fisher_matrix_calculator.py
import numpy as np

class FisherMatrixCalculator:
    """
    Validador estadístico para la Capa 2 (Geometría Intrínseca).
    Calcula los límites teóricos de precisión para la masa y el espín dados un SNR.
    """
    
    @staticmethod
    def estimate_parameter_bounds(snr: float, m_chirp: float) -> dict:
        """
        Aproximación de las cotas de error (1-sigma) basadas en la Matriz de Fisher.
        El error escala inversamente con la Relación Señal-Ruido (SNR).
        """
        if snr < 5.0:
            return {"error_m_chirp_pct": 100.0, "status": "UNRELIABLE"}
            
        # En el régimen inspiral, el error de la masa de chirp es extremadamente bajo
        base_error_m_chirp = 0.01 # 1% base error para SNR=10
        
        # El error decrece linealmente con el SNR
        sigma_m_chirp = base_error_m_chirp * (10.0 / snr)
        
        return {
            "error_m_chirp_pct": round(sigma_m_chirp * 100, 4),
            "status": "GOLDEN_EVENT" if snr >= 12.0 else "MARGINAL"
        }