import h5py
import os
import time
from pathlib import Path
import numpy as np

class HDF5Exporter:
    """
    Exportador QNIM: Genera archivos compatibles con LIGO donde la señal
    está separada de las etiquetas de entrenamiento.
    """

    def __init__(self, base_path="/qnim/data/synthetic"):
        # Aseguramos que la ruta sea relativa al proyecto si no es absoluta
        self.base_path = Path(os.getcwd()) / "data" / "synthetic"

    def save_batch(self, dataset):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        folder_path = self.base_path / timestamp
        folder_path.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(dataset):
            file_name = f"event_{i:05d}.h5" # Nombre genérico para no dar pistas
            file_path = folder_path / file_name

            with h5py.File(file_path, "w") as f:
                # LA SEÑAL BRUTA (Única entrada para el Quantum Bridge)
                f.create_dataset("strain", data=sample["strain"], compression="gzip")
                
                # METADATOS (Solo para validación posterior, el QML no los lee)
                # Guardamos la 'etiqueta' que el QML debe intentar adivinar
                f.attrs["target_label"] = np.bytes_(sample["label"]) # Para compatibilidad NumPy 2.0                
                # Parámetros físicos reales (Ground Truth)
                meta = sample["metadata"]
                f.attrs["true_m1"] = float(meta["m1"])
                f.attrs["true_distance"] = float(meta["distance"])
                f.attrs["true_theory"] = np.bytes_(meta["theory"]) # Para compatibilidad NumPy 2.0

                if "extra_dims" in meta:
                    f.attrs["anomaly_parameter"] = float(meta["extra_dims"])

        return str(folder_path)