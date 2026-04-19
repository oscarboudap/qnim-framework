import h5py
import numpy as np
import glob
import sys
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Asegurar que el path del proyecto sea visible
current_file = Path(__file__).resolve()
project_root = next(p for p in current_file.parents if p.name == 'qnim')
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class MassiveDatasetLoader:
    def __init__(self, data_path="data/synthetic/massive_dataset/*.h5"):
        self.files = sorted(glob.glob(str(project_root / data_path)))
        self.encoder = LabelEncoder()
        self.theories = [
            "Kerr_Vacuum", 
            "LQG_Area_Quantization", 
            "String_Theory_Fuzzball", 
            "Horndeski_DHOST", 
            "Exotic_Compact_Object"
        ]
        self.encoder.fit(self.theories)

    def load_and_preprocess(self, n_components=12, fixed_length=16384):
        """Carga, estandariza longitud y comprime los datos."""
        X_raw, y_raw = [], []
        
        print(f"📦 Cargando y alineando {len(self.files)} eventos...")
        for f in self.files:
            with h5py.File(f, 'r') as h5:
                strain = h5['strain'][:]
                
                # --- LÓGICA DE ALINEACIÓN ---
                # Tomamos los últimos 'fixed_length' puntos (el final de la onda es el choque)
                if len(strain) > fixed_length:
                    strain = strain[-fixed_length:]
                else:
                    # Si es más corta, rellenamos con ceros al principio (padding)
                    strain = np.pad(strain, (fixed_length - len(strain), 0), 'constant')
                
                # Whitening: Normalización
                strain_whitened = (strain - np.mean(strain)) / (np.std(strain) + 1e-10)
                
                X_raw.append(strain_whitened)
                y_raw.append(h5.attrs['theory'])

        # Ahora sí: todas miden exactamente 16384, Numpy será feliz
        X_np = np.array(X_raw)
        
        print(f"🧩 Aplicando PCA (Reducción de {fixed_length} -> {n_components})...")
        pca = PCA(n_components=n_components)
        X_compressed = pca.fit_transform(X_np)
        
        # Escala para el VQC [0, pi]
        X_final = (X_compressed - X_compressed.min()) / (X_compressed.max() - X_compressed.min()) * np.pi
        
        return X_final, self.encoder.transform(y_raw), pca