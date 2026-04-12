import h5py
import numpy as np
from scipy import signal

class QuantumDatasetLoader:
    """
    Cargador y Preprocesador de Grado Cuántico.
    Prepara la señal .h5 para Amplitude Encoding en la QPU.
    """

    def __init__(self, target_samples=64):
        self.target_samples = target_samples  # 64 puntos = 6 cúbits (2^6)

    def prepare_for_quantum(self, file_path):
        """
        Método principal: Carga desde disco y procesa.
        """
        with h5py.File(file_path, 'r') as f:
            strain = f['strain'][:]
            # Creamos un formato compatible con el procesador interno
            event_data = {'strain': strain}
            
        return self.prepare_for_quantum_event(event_data)

    def prepare_for_quantum_event(self, event_data):
        """
        Procesa los datos (ya sea que vengan de disco o de memoria).
        """
        strain = event_data['strain']
        
        # 1. Whitening (Eliminar el ruido de fondo coloreado)
        strain_whiten = self._whiten(strain)
        
        # 2. Bandpass (Filtro 30-500Hz para centrarse en el evento)
        strain_filter = self._bandpass(strain_whiten, 30, 500, fs=4096)
        
        # 3. Windowing (Centrado en el pico de la onda)
        peak_idx = np.argmax(np.abs(strain_filter))
        half_window = self.target_samples // 2
        start = max(0, peak_idx - half_window)
        end = start + self.target_samples
        
        quantum_ready_data = strain_filter[start:end]
        
        # Padding de seguridad
        if len(quantum_ready_data) < self.target_samples:
            quantum_ready_data = np.pad(quantum_ready_data, (0, self.target_samples - len(quantum_ready_data)))

        # 4. Normalización Unitaria (Obligatorio para amplitud cuántica)
        norm = np.linalg.norm(quantum_ready_data)
        if norm == 0: return quantum_ready_data
        
        return quantum_ready_data / norm

    def _whiten(self, data):
        f, psd = signal.welch(data, fs=4096, nperseg=max(256, len(data)//8))
        interp_psd = np.interp(np.linspace(0, 0.5, len(data)), f/4096, psd)
        return np.fft.ifft(np.fft.fft(data) / np.sqrt(interp_psd + 1e-10)).real

    def _bandpass(self, data, low, high, fs):
        sos = signal.butter(4, [low, high], btype='band', fs=fs, output='sos')
        return signal.sosfilt(sos, data)