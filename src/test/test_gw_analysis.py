"""
Tests para GravitationalWaveAnalyzer

Usa GW150914 como EJEMPLO de prueba (datos reales del LIGO GWOSC).
Pero el módulo gw_analysis.py es completamente agnóstico - funciona
con CUALQUIER onda gravitatoria de cualquier fuente.
"""

import pytest
import numpy as np
import os
import tempfile
import h5py
from pathlib import Path

from src.application.gw_analysis import (
    HDF5DataLoader,
    GravitationalWaveSignal,
    GravitationalWaveAnalyzer,
    GroundTruthParameters
)


class TestHDF5DataLoader:
    """Tests para carga agnóstica de HDF5"""
    
    @pytest.fixture
    def sample_h5_file(self):
        """Crea archivo HDF5 de prueba (formato GWOSC)"""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name
        
        # Crear estructura similar a GWOSC
        with h5py.File(temp_path, 'w') as f:
            # Strain data
            f.create_dataset('strain/Strain', data=np.random.randn(131072))
            
            # Metadata
            meta = f.create_group('meta')
            meta.create_dataset('Duration', data=32.0)
            meta.create_dataset('GPSstart', data=1126259446)
            
            # Atributos
            f.attrs['SamplingRate'] = 16384
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    def test_explore_structure(self, sample_h5_file):
        """Test exploración agnóstica de estructura"""
        structure = HDF5DataLoader.explore_structure(sample_h5_file)
        
        assert 'strain/Strain' in structure
        assert structure['strain/Strain']['type'] == 'dataset'
        assert 'float' in structure['strain/Strain']['dtype']
    
    def test_find_strain_dataset(self, sample_h5_file):
        """Test detección automática de strain"""
        strain_path = HDF5DataLoader.find_strain_dataset(sample_h5_file)
        
        assert strain_path == 'strain/Strain'
    
    def test_extract_metadata(self, sample_h5_file):
        """Test extracción agnóstica de metadatos"""
        metadata = HDF5DataLoader.extract_metadata(sample_h5_file)
        
        assert 'SamplingRate' in metadata
        assert metadata['SamplingRate'] == 16384
        assert 'GPSstart' in metadata
    
    def test_load_signal(self, sample_h5_file):
        """Test carga de señal"""
        signal = HDF5DataLoader.load_signal(
            sample_h5_file,
            detector_name="TestDetector"
        )
        
        assert isinstance(signal, GravitationalWaveSignal)
        assert signal.sampling_rate == 16384
        assert signal.detector_name == "TestDetector"
        assert signal.duration > 0
        assert len(signal.strain) == 131072
    
    def test_load_nonexistent_file(self):
        """Test manejo de archivo inexistente"""
        with pytest.raises(FileNotFoundError):
            HDF5DataLoader.load_signal('/nonexistent/path.h5')


class TestGravitationalWaveSignal:
    """Tests para dataclass de señal"""
    
    def test_signal_creation(self):
        """Test creación de señal"""
        strain = np.random.randn(1000)
        signal = GravitationalWaveSignal(
            strain=strain,
            sampling_rate=16384,
            detector_name="H1"
        )
        
        assert len(signal.strain) == 1000
        assert signal.sampling_rate == 16384
        assert signal.detector_name == "H1"
        assert abs(signal.duration - 1000/16384) < 1e-6
    
    def test_signal_duration_auto_compute(self):
        """Test cómputo automático de duración"""
        strain = np.random.randn(16384)
        signal = GravitationalWaveSignal(
            strain=strain,
            sampling_rate=16384
        )
        
        assert abs(signal.duration - 1.0) < 1e-6
    
    def test_signal_with_metadata(self):
        """Test señal con metadatos completos"""
        signal = GravitationalWaveSignal(
            strain=np.random.randn(100),
            sampling_rate=8192,
            gps_time=1126259446.5,
            detector_name="L1",
            duration=0.0122
        )
        
        assert signal.gps_time == 1126259446.5
        assert signal.detector_name == "L1"


class TestGravitationalWaveAnalyzer:
    """Tests para análisis agnóstico"""
    
    @pytest.fixture
    def sample_signal(self):
        """Señal de prueba"""
        # Simulación: chirp simple
        t = np.linspace(0, 1, 16384)
        strain = signal_chirp(t, 35, 250)
        
        return GravitationalWaveSignal(
            strain=strain,
            sampling_rate=16384,
            detector_name="H1",
            gps_time=1126259446
        )
    
    def test_preprocess_signal(self, sample_signal):
        """Test preprocesamiento agnóstico"""
        processed = GravitationalWaveAnalyzer.preprocess_signal(sample_signal)
        
        # Verificar normalización
        assert np.mean(processed) < 1e-10
        assert abs(np.std(processed) - 1.0) < 0.1
        assert len(processed) == len(sample_signal.strain)
    
    def test_analyze_signal_basic(self, sample_signal):
        """Test análisis básico sin modelo"""
        result = GravitationalWaveAnalyzer.analyze_signal(sample_signal)
        
        assert result["detector"] == "H1"
        assert result["sampling_rate"] == 16384
        assert result["gps_time"] == 1126259446
    
    def test_analyze_signal_with_ground_truth(self, sample_signal):
        """Test análisis con parámetros conocidos"""
        ground_truth = GroundTruthParameters(
            event_name="test-event",
            source_type="BH-BH",
            mass1_msun=36.0,
            mass2_msun=29.0,
            distance_mpc=410,
            network_snr=24.4
        )
        
        result = GravitationalWaveAnalyzer.analyze_signal(
            sample_signal,
            ground_truth=ground_truth
        )
        
        assert "ground_truth" in result
        assert result["ground_truth"]["source_type"] == "BH-BH"
        assert result["ground_truth"]["mass1_msun"] == 36.0
    
    def test_analyze_signal_with_mock_model(self, sample_signal):
        """Test análisis con modelo (mock)"""
        
        class MockModel:
            def predict(self, signal):
                return {"prediction": "test", "confidence": 0.95}
        
        model = MockModel()
        result = GravitationalWaveAnalyzer.analyze_signal(
            sample_signal,
            model=model
        )
        
        assert "model_predictions" in result
        assert result["model_predictions"]["confidence"] == 0.95


class TestGroundTruthParameters:
    """Tests para parámetros conocidos"""
    
    def test_gw150914_example(self):
        """Test con parámetros de GW150914 (ejemplo)"""
        # Estos son los valores REALES de GW150914
        # Pero el módulo es agnóstico - estos parámetros podrían
        # ser de cualquier evento
        
        gw150914 = GroundTruthParameters(
            event_name="GW150914",
            source_type="BH-BH",
            mass1_msun=36.3,
            mass2_msun=29.1,
            distance_mpc=410,
            spin1=0.324,
            spin2=-0.212,
            network_snr=24.4
        )
        
        assert gw150914.event_name == "GW150914"
        assert gw150914.mass1_msun > gw150914.mass2_msun
        assert gw150914.network_snr > 0
    
    def test_exotic_source_example(self):
        """Test con fuente exótica (boson stars)"""
        exotic = GroundTruthParameters(
            event_name="Hypothetical-Boson-Star-Merger",
            source_type="exotic-boson-stars",
            distance_mpc=100,
            extra_params={
                "spin_frequency_hz": 5000,
                "boson_mass_ev": 1e-11
            }
        )
        
        assert exotic.source_type == "exotic-boson-stars"
        assert "boson_mass_ev" in exotic.extra_params


class TestDetectorPairAnalysis:
    """Tests para análisis de pares de detectores"""
    
    @pytest.fixture
    def detector_pair(self):
        """Par de detectores simulado"""
        t = np.linspace(0, 1, 16384)
        chirp = signal_chirp(t, 35, 250)
        
        sig1 = GravitationalWaveSignal(
            strain=chirp,
            sampling_rate=16384,
            detector_name="H1"
        )
        
        sig2 = GravitationalWaveSignal(
            strain=chirp + np.random.randn(16384) * 0.01,
            sampling_rate=16384,
            detector_name="L1"
        )
        
        return sig1, sig2
    
    def test_analyze_pair(self, detector_pair):
        """Test análisis de par agnóstico"""
        sig1, sig2 = detector_pair
        
        result = GravitationalWaveAnalyzer.analyze_detector_pair(sig1, sig2)
        
        assert "detector1" in result
        assert "detector2" in result
        assert result["detectors"] == ["H1", "L1"]
        assert result["detector1"]["detector"] == "H1"
        assert result["detector2"]["detector"] == "L1"


class TestHDF5LoaderReallyData:
    """Tests con datos REALES si están disponibles"""
    
    def get_real_data_paths(self):
        """Obtiene paths a datos reales si existen"""
        base_path = Path("c:/Users/oscar/Desktop/TFM/qnim/qnim/data/raw")
        
        if base_path.exists():
            h1_file = base_path / "H-H1_LOSC_4_V2-1126259446-32.hdf5"
            l1_file = base_path / "L-L1_LOSC_4_V2-1126259446-32.hdf5"
            
            if h1_file.exists() and l1_file.exists():
                return str(h1_file), str(l1_file)
        
        return None, None
    
    def test_load_real_gw150914_h1(self):
        """Test carga de H1 real (GW150914)"""
        h1_path, _ = self.get_real_data_paths()
        
        if h1_path is None:
            pytest.skip("Datos reales no disponibles")
        
        signal = HDF5DataLoader.load_signal(h1_path, detector_name="H1")
        
        assert signal.detector_name == "H1"
        assert signal.sampling_rate == 16384
        assert len(signal.strain) > 0
        assert signal.duration > 0
    
    def test_load_real_gw150914_l1(self):
        """Test carga de L1 real (GW150914)"""
        _, l1_path = self.get_real_data_paths()
        
        if l1_path is None:
            pytest.skip("Datos reales no disponibles")
        
        signal = HDF5DataLoader.load_signal(l1_path, detector_name="L1")
        
        assert signal.detector_name == "L1"
        assert len(signal.strain) > 0
    
    def test_load_detector_pair_real(self):
        """Test carga de par real (agnóstico - funciona con cualquier par)"""
        h1_path, l1_path = self.get_real_data_paths()
        
        if h1_path is None or l1_path is None:
            pytest.skip("Datos reales no disponibles")
        
        sig1, sig2 = HDF5DataLoader.load_detector_pair(
            h1_path, l1_path,
            detector1_name="H1",
            detector2_name="L1"
        )
        
        assert len(sig1.strain) == len(sig2.strain)
        assert sig1.sampling_rate == sig2.sampling_rate
    
    def test_analyze_real_pair(self):
        """Test análisis de par real agnóstico"""
        h1_path, l1_path = self.get_real_data_paths()
        
        if h1_path is None or l1_path is None:
            pytest.skip("Datos reales no disponibles")
        
        # Cargar
        sig1, sig2 = HDF5DataLoader.load_detector_pair(
            h1_path, l1_path,
            detector1_name="H1",
            detector2_name="L1"
        )
        
        # Analizar (agnóstico - no asume nada de GW150914)
        result = GravitationalWaveAnalyzer.analyze_detector_pair(sig1, sig2)
        
        assert result["detector1"]["detector"] == "H1"
        assert result["detector2"]["detector"] == "L1"


# Utilidades

def signal_chirp(t, f0, f1):
    """Genera chirp simple para test"""
    from scipy.signal import chirp
    return chirp(t, f0, t[-1], f1, method='linear')
