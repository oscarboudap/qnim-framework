"""
Gravitational Wave Analysis Module

Análisis AGNÓSTICO de CUALQUIER onda gravitatoria basado en lo aprendido 
durante el entrenamiento del modelo QNIM.

NO particulariza la fuente. El análisis se basa en características aprendidas
por el modelo durante la etapa de entrenamiento, no en análisis manual.

Soporta:
- Coalescencias binarias de cualquier tipo (BH+BH, NS+NS, BH+NS, exóticas)
- Defectos topológicos (cuerdas cósmicas, paredes de dominio)
- Fuentes exóticas (estrellas de bosones, superradiancia, etc.)
- Cualquier otro evento que genere una solución a las ecuaciones de Einstein
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import h5py
import os
from scipy import signal
from dataclasses import dataclass, field


@dataclass
class GravitationalWaveSignal:
    """Representación agnóstica de una señal de onda gravitatoria"""
    strain: np.ndarray
    sampling_rate: float
    gps_time: Optional[float] = None
    detector_name: Optional[str] = None
    duration: Optional[float] = None
    
    def __post_init__(self):
        if self.duration is None:
            self.duration = len(self.strain) / self.sampling_rate


class HDF5DataLoader:
    """
    Cargador agnóstico de datos HDF5.
    
    NO asume estructura fija. Busca automáticamente datasets
    que contengan strain y metadatos.
    """
    
    @staticmethod
    def explore_structure(h5_path: str) -> Dict[str, Any]:
        """Explora estructura del archivo HDF5"""
        structure = {}
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Archivo no encontrado: {h5_path}")
        
        with h5py.File(h5_path, 'r') as f:
            def _traverse(name, obj):
                if isinstance(obj, h5py.Dataset):
                    structure[name] = {
                        'type': 'dataset',
                        'shape': obj.shape,
                        'dtype': str(obj.dtype)
                    }
                elif isinstance(obj, h5py.Group):
                    structure[name] = {'type': 'group'}
            
            f.visititems(_traverse)
        
        return structure
    
    @staticmethod
    def find_strain_dataset(h5_path: str) -> str:
        """Auto-detecta el dataset que contiene el strain"""
        structure = HDF5DataLoader.explore_structure(h5_path)
        
        # Rutas comunes
        common_paths = [
            'strain/Strain', '/strain/Strain',
            'Strain', '/Strain',
            'strain', '/strain'
        ]
        
        for path in common_paths:
            if path in structure and structure[path]['type'] == 'dataset':
                return path
        
        # Si no, buscar dataset más grande (float64)
        largest_dataset = None
        largest_size = 0
        
        for path, info in structure.items():
            if info['type'] == 'dataset':
                size = np.prod(info['shape']) if info['shape'] else 0
                if size > largest_size and 'float' in info['dtype']:
                    largest_size = size
                    largest_dataset = path
        
        if largest_dataset:
            return largest_dataset
        
        raise ValueError(f"No strain dataset encontrado en {h5_path}")
    
    @staticmethod
    def extract_metadata(h5_path: str) -> Dict[str, Any]:
        """Extrae metadatos disponibles"""
        metadata = {}
        
        with h5py.File(h5_path, 'r') as f:
            if 'meta' in f:
                for key in f['meta'].keys():
                    try:
                        metadata[key] = f['meta'][key][()]
                    except:
                        pass
            
            for key, val in f.attrs.items():
                metadata[key] = val
        
        return metadata
    
    @staticmethod
    def load_signal(
        h5_path: str,
        strain_path: Optional[str] = None,
        detector_name: Optional[str] = None
    ) -> GravitationalWaveSignal:
        """
        Carga una señal de onda gravitatoria desde HDF5
        
        Args:
            h5_path: Ruta al archivo HDF5
            strain_path: Path específico (si no, auto-detecta)
            detector_name: Nombre del detector
        
        Returns:
            GravitationalWaveSignal
        """
        if strain_path is None:
            strain_path = HDF5DataLoader.find_strain_dataset(h5_path)
        
        with h5py.File(h5_path, 'r') as f:
            strain = f[strain_path][:]
            metadata = HDF5DataLoader.extract_metadata(h5_path)
            
            # Inferir sampling rate (priorizar SamplingRate si está disponible)
            if 'SamplingRate' in metadata:
                fs = float(metadata['SamplingRate'])
            elif 'Duration' in metadata or 'duration' in metadata:
                duration = float(metadata.get('Duration', metadata.get('duration', 1.0)))
                fs = len(strain) / duration
            else:
                fs = 16384.0  # Default LIGO
            
            gps_time = metadata.get('GPSstart', None)
            if gps_time is not None:
                gps_time = float(gps_time)
        
        return GravitationalWaveSignal(
            strain=strain,
            sampling_rate=fs,
            gps_time=gps_time,
            detector_name=detector_name
        )
    
    @staticmethod
    def load_detector_pair(
        detector1_path: str,
        detector2_path: str,
        detector1_name: str = "D1",
        detector2_name: str = "D2"
    ) -> Tuple[GravitationalWaveSignal, GravitationalWaveSignal]:
        """Carga par de detectores"""
        sig1 = HDF5DataLoader.load_signal(detector1_path, detector_name=detector1_name)
        sig2 = HDF5DataLoader.load_signal(detector2_path, detector_name=detector2_name)
        
        # Mismo tamaño
        min_len = min(len(sig1.strain), len(sig2.strain))
        sig1.strain = sig1.strain[:min_len]
        sig2.strain = sig2.strain[:min_len]
        
        if abs(sig1.sampling_rate - sig2.sampling_rate) > 1.0:
            raise ValueError(f"Sampling rates no coinciden")
        
        return sig1, sig2


@dataclass
class GroundTruthParameters:
    """Parámetros conocidos (CUANDO están disponibles)"""
    event_name: str
    source_type: Optional[str] = None  # "BH-BH", "NS-NS", "BH-NS", "exotic", etc.
    mass1_msun: Optional[float] = None
    mass2_msun: Optional[float] = None
    distance_mpc: Optional[float] = None
    
    # Opcionales
    spin1: Optional[float] = None
    spin2: Optional[float] = None
    redshift: Optional[float] = None
    network_snr: Optional[float] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


class GravitationalWaveAnalyzer:
    """
    Analizador agnóstico de ondas gravitacionales.
    
    El análisis se basa en lo aprendido por el modelo durante entrenamiento.
    NO asume nada sobre la fuente, solo analiza la forma de onda
    pasándola a través del modelo entrenado.
    """
    
    @staticmethod
    def preprocess_signal(signal: GravitationalWaveSignal) -> np.ndarray:
        """
        Preprocesa la señal para entrada al modelo
        
        Normalización, whitening, y preparación de características
        basadas en lo que el modelo espera.
        """
        strain = signal.strain.copy()
        
        # Remover DC offset
        strain -= np.mean(strain)
        
        # Normalizar amplitud
        scale = np.std(strain)
        if scale > 0:
            strain = strain / scale
        
        return strain
    
    @staticmethod
    def analyze_signal(
        signal: GravitationalWaveSignal,
        ground_truth: Optional[GroundTruthParameters] = None,
        model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Análisis completo agnóstico de una onda gravitatoria.
        
        Procesa la señal a través del modelo entrenado para obtener
        predicciones y características aprendidas.
        
        Args:
            signal: Señal a analizar (datos reales o sintéticos)
            ground_truth: Parámetros conocidos (opcional)
            model: Modelo entrenado (si None, retorna características básicas)
        
        Returns:
            Dict con predicciones y análisis
        """
        
        # Preprocesar
        processed_strain = GravitationalWaveAnalyzer.preprocess_signal(signal)
        
        result = {
            "detector": signal.detector_name,
            "sampling_rate": signal.sampling_rate,
            "duration": signal.duration,
            "gps_time": signal.gps_time
        }
        
        # Si hay modelo entrenado, usarlo
        if model is not None:
            try:
                predictions = model.predict(processed_strain)
                result["model_predictions"] = predictions
            except Exception as e:
                result["model_error"] = str(e)
        
        # Información de ground truth si está disponible
        if ground_truth is not None:
            result["ground_truth"] = {
                "event_name": ground_truth.event_name,
                "source_type": ground_truth.source_type,
                "mass1_msun": ground_truth.mass1_msun,
                "mass2_msun": ground_truth.mass2_msun,
                "distance_mpc": ground_truth.distance_mpc,
                "network_snr": ground_truth.network_snr
            }
        
        return result
    
    @staticmethod
    def analyze_detector_pair(
        signal1: GravitationalWaveSignal,
        signal2: GravitationalWaveSignal,
        ground_truth: Optional[GroundTruthParameters] = None,
        model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Análisis de par de detectores"""
        
        analysis1 = GravitationalWaveAnalyzer.analyze_signal(signal1, ground_truth, model)
        analysis2 = GravitationalWaveAnalyzer.analyze_signal(signal2, ground_truth, model)
        
        return {
            "detector1": analysis1,
            "detector2": analysis2,
            "detectors": [signal1.detector_name, signal2.detector_name]
        }
