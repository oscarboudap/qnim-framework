import h5py
import os
import numpy as np
from src.domain.quantum.interfaces import IGWRepository
from src.domain.astrophysics.entities import GWSignal
from src.domain.astrophysics.value_objects import DetectorType, GPSTime

class H5GWRepository(IGWRepository):
    def __init__(self, base_path: str):
        self.base_path = base_path

    def get_signal_by_detector(self, detector_name: str) -> GWSignal:
        # Mapeo de nombres de archivo estándar de LIGO
        file_map = {
            "H1": "H-H1_LOSC_4_V2-1126259446-32.hdf5",
            "L1": "L-L1_LOSC_4_V2-1126259446-32.hdf5"
        }
        
        file_path = os.path.join(self.base_path, "data", "raw", file_map[detector_name])
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encuentra el archivo HDF5 en: {file_path}")

        with h5py.File(file_path, 'r') as f:
            # Extracción de la serie temporal (Strain)
            strain = f['strain']['Strain'][:]
            
            # Extracción de metadatos (manejando diferentes versiones de HDF5 de LIGO)
            if 'dt' in f['meta']:
                dt = f['meta']['dt'][()]
            else:
                dt = f['strain']['Strain'].attrs['Xspacing']
                
            gps_start = f['meta']['gpsstart'][()] if 'gpsstart' in f['meta'] else f['meta']['GPSstart'][()]
            
            return GWSignal(
                strain=strain,
                detector=DetectorType[detector_name],
                sample_rate=int(1/dt),
                gps_start=GPSTime(float(gps_start))
            )