import numpy as np
import h5py
import warnings

try:
    from gwpy.timeseries import TimeSeries
except ImportError:
    TimeSeries = None

class QuantumDatasetLoader:
    """
    Infraestructura para carga de datos de GW.
    Soporta archivos sintéticos de QNIM y archivos reales HDF5 de LIGO (GWOSC).
    """
    def __init__(self, target_samples: int = 16384):
        self.target_samples = target_samples

    def prepare_for_quantum(self, file_path: str, is_real_data: bool = False) -> np.ndarray:
        """Carga y prepara la señal. Aplica Whitening si es un archivo de LIGO real."""
        strain = self._load_strain(file_path)

        if is_real_data:
            strain = self._apply_ligo_whitening(strain, fs=4096)

        return self._format_length(strain)

    def _load_strain(self, file_path: str) -> np.ndarray:
        """Extrae el array de datos adaptándose a la estructura del HDF5."""
        with h5py.File(file_path, 'r') as f:
            # 1. Comprobamos PRIMERO el formato oficial de LIGO GWOSC
            if 'strain/Strain' in f: 
                return f['strain/Strain'][:]
            # 2. Comprobamos el formato sintético de QNIM
            elif 'strain' in f:
                return f['strain'][:]
            # 3. Fallback genérico
            else:
                keys = list(f.keys())
                return f[keys[0]][:]

    def _apply_ligo_whitening(self, strain: np.ndarray, fs: int) -> np.ndarray:
        """
        [Nivel de Robustez Investigadora]
        Aplica blanqueo y filtrado usando el motor interno de GWPy.
        Evita conflictos de argumentos entre Scipy y GWPy.
        """
        if TimeSeries is None:
            raise ImportError("❌ Falta la librería 'gwpy'. Ejecuta: pip install gwpy")

        print("🧹 Iniciando proceso de Whitening (PSD interna Welch)...")
        
        # 1. Convertir a objeto TimeSeries de GWPY
        ts = TimeSeries(strain, sample_rate=fs)

        # 2. Blanqueo (Whitening)
        # En lugar de pre-calcular la PSD, pasamos los parámetros de tiempo directamente.
        # fftlength=4: usamos segmentos de 4 segundos para la PSD (resolución de 0.25Hz).
        # overlap=2: 50% de solapamiento para reducir la varianza de la estimación.
        whitened = ts.whiten(fftlength=4, overlap=2)

        # 3. Filtro de paso de banda (Bandpass)
        # Aislamos el rango 20-300Hz donde el detector es más sensible y está la señal.
        bp_whitened = whitened.bandpass(20, 300)

        # 4. Recorte de bordes (Crop)
        # El proceso de blanqueo y filtrado ensucia los extremos de la señal.
        # Recortamos 1 segundo de cada lado para asegurar que los datos sean puros.
        crop_samples = int(1.0 * fs) 
        if len(bp_whitened) > (2 * crop_samples):
            clean_strain = bp_whitened.value[crop_samples:-crop_samples]
        else:
            clean_strain = bp_whitened.value

        print(f"✅ Señal blanqueada y filtrada con éxito. Longitud final: {len(clean_strain)} muestras.")
        return clean_strain

    def _format_length(self, strain: np.ndarray) -> np.ndarray:
        """Ajusta el vector a la dimensión exacta requerida por el Pipeline PCA."""
        if len(strain) > self.target_samples:
            # Extraemos el centro de la señal (donde suele estar el merger)
            start = (len(strain) - self.target_samples) // 2
            return strain[start:start + self.target_samples]
        elif len(strain) < self.target_samples:
            # Hacemos padding con ceros si la señal es más corta
            return np.pad(strain, (0, self.target_samples - len(strain)), 'constant')
        return strain