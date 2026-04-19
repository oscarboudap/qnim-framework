import numpy as np
import h5py

try:
    from gwpy.timeseries import TimeSeries
except ImportError:
    TimeSeries = None

class QuantumDatasetLoader:
    def __init__(self, target_samples: int = 16384):
        self.target_samples = target_samples

    def prepare_for_quantum(self, file_path: str, is_real_data: bool = False) -> np.ndarray:
        """Carga y prepara la señal con alineación forzada para el VQC."""
        strain = self._load_strain(file_path)

        if is_real_data:
            strain = self._apply_ligo_whitening(strain, fs=4096)

        # CRÍTICO: Pasamos el flag para decidir cómo recortar/alinear
        return self._format_length(strain, is_real_data=is_real_data)

    def _load_strain(self, file_path: str) -> np.ndarray:
        with h5py.File(file_path, 'r') as f:
            if 'strain/Strain' in f: 
                return f['strain/Strain'][:]
            elif 'strain' in f:
                return f['strain'][:]
            else:
                keys = list(f.keys())
                return f[keys[0]][:]

    def _apply_ligo_whitening(self, strain: np.ndarray, fs: int) -> np.ndarray:
        if TimeSeries is None:
            raise ImportError("❌ Falta 'gwpy'. Ejecuta: pip install gwpy")

        print("🧹 Iniciando Whitening y Bandpass (20-300Hz)...")
        ts = TimeSeries(strain, sample_rate=fs)
        # Whitening con parámetros de alta resolución
        whitened = ts.whiten(fftlength=4, overlap=2)
        bp_whitened = whitened.bandpass(20, 300)

        # Recorte de seguridad por efectos de borde
        crop_samples = int(1.0 * fs) 
        return bp_whitened.value[crop_samples:-crop_samples]

    def _format_length(self, strain: np.ndarray, is_real_data: bool = False) -> np.ndarray:
        """
        Ajusta la señal para que el 'merger' coincida con la posición del entrenamiento.
        """
        if is_real_data:
            # --- LÓGICA DE ALINEACIÓN PARA DATOS REALES ---
            # En un archivo de 32s, buscamos el pico máximo (el merger de GW150914)
            peak_idx = np.argmax(np.abs(strain))
            
            # Para que el PCA funcione, el pico debe estar cerca del final (al 90-95%)
            # Tomamos 15,000 muestras antes del pico y 1,384 después.
            start = peak_idx - int(self.target_samples * 0.9)
            end = start + self.target_samples
            
            # Control de bordes
            if start < 0:
                return np.pad(strain[:end], (abs(start), 0), 'constant')
            if end > len(strain):
                return np.pad(strain[start:], (0, end - len(strain)), 'constant')
            
            return strain[start:end]
        
        else:
            # --- LÓGICA PARA DATOS SINTÉTICOS ---
            # Igual que en MassiveDatasetLoader: tomamos el final y pad al principio
            if len(strain) > self.target_samples:
                return strain[-self.target_samples:]
            elif len(strain) < self.target_samples:
                return np.pad(strain, (self.target_samples - len(strain), 0), 'constant')
        
        return strain