import h5py
import numpy as np
from scipy import signal

class QuantumDatasetLoader:
    """
    Cargador y Preprocesador de Grado Cuántico.
    Prepara la señal .h5 para Amplitude Encoding en la QPU.
    """

    def __init__(self, target_samples=64):
        self.target_samples = target_samples # Potencia de 2 para Amplitude Encoding

    def prepare_for_quantum_event(self, event_data):
        """
        Versión para procesar un diccionario de evento directamente en memoria.
        Útil para visualización y tests rápidos sin pasar por disco.
        """
        strain = event_data['strain']
        
        # 1. Whitening
        strain_whiten = self._whiten(strain)
        
        # 2. Bandpass (30-500Hz)
        strain_filter = self._bandpass(strain_whiten, 30, 500, fs=4096)
        
        # 3. Windowing (Peak finding)
        peak_idx = np.argmax(np.abs(strain_filter))
        half_window = self.target_samples // 2
        
        # Aseguramos que no nos salimos de los índices
        start = max(0, peak_idx - half_window)
        end = start + self.target_samples
        
        quantum_ready_data = strain_filter[start:end]
        
        # Relleno con ceros si la señal es más corta (padding)
        if len(quantum_ready_data) < self.target_samples:
            quantum_ready_data = np.pad(quantum_ready_data, (0, self.target_samples - len(quantum_ready_data)))

        # 4. Normalización Unitaria
        norm = np.linalg.norm(quantum_ready_data)
        if norm == 0: return quantum_ready_data # Evitar división por cero
        
        return quantum_ready_data / norm

    def _whiten(self, data):
        # Simplificación de whitening: dividimos por la raíz de la densidad espectral
        f, psd = signal.welch(data, fs=4096, nperseg=1024)
        interp_psd = np.interp(np.linspace(0, 0.5, len(data)), f/4096, psd)
        return np.fft.ifft(np.fft.fft(data) / np.sqrt(interp_psd)).real

    def _bandpass(self, data, low, high, fs):
        sos = signal.butter(4, [low, high], btype='band', fs=fs, output='sos')
        return signal.sosfilt(sos, data)