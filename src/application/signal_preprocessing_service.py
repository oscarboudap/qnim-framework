import numpy as np
from scipy import signal as sig
from src.domain.astrophysics.entities import GWSignal

class SignalPreprocessingService:
    def __init__(self, sample_rate: int = 4096):
        self.sample_rate = sample_rate

    def whitening(self, gw_signal: GWSignal) -> GWSignal:
        """
        Aplica Blanqueo (Whitening) espectral para eliminar el color del ruido de LIGO.
        Esencial para que el VQC identifique modos cuasinormales (QNM).
        """
        data = gw_signal.strain
        
        # 1. Calculamos la PSD (Power Spectral Density) mediante el método de Welch
        frequencies, psd = sig.welch(data, self.sample_rate, nperseg=self.sample_rate)
        
        # 2. Interpolamos para que coincida con la longitud de la señal
        interp_psd = np.interp(np.fft.rfftfreq(len(data), 1/self.sample_rate), frequencies, psd)
        
        # 3. Transformada de Fourier, blanqueo en frecuencia y vuelta al tiempo
        hf = np.fft.rfft(data)
        white_hf = hf / np.sqrt(interp_psd)
        white_data = np.fft.irfft(white_hf, n=len(data))
        
        return GWSignal(
            strain=white_data,
            detector=gw_signal.detector,
            sample_rate=self.sample_rate,
            gps_start=gw_signal.gps_start
        )

    def bandpass_filter(self, gw_signal: GWSignal, low: float = 30.0, high: float = 500.0) -> GWSignal:
        """Filtro de Butterworth para limpiar artefactos fuera de la banda de interés."""
        nyq = 0.5 * self.sample_rate
        b, a = sig.butter(4, [low / nyq, high / nyq], btype='band')
        filtered_strain = sig.filtfilt(b, a, gw_signal.strain)
        
        return GWSignal(filtered_strain, gw_signal.detector, self.sample_rate, gw_signal.gps_start)