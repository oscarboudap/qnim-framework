"""
Tests para Scripts de Visualización (PASO 5)

Valida que los scripts generan plots correctamente sin errores.
"""

import pytest
import numpy as np
import os
from pathlib import Path
import sys
import importlib.util

# Importar scripts dinámicamente desde scripts/
scripts_dir = Path(__file__).parent.parent.parent / 'scripts'

def load_script_module(script_name):
    """Carga dinámicamente un script de la carpeta scripts/"""
    script_path = scripts_dir / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Cargar los módulos
pc = load_script_module('plot_convergence_comparison')
pp = load_script_module('plot_posterior_distributions')
pt = load_script_module('plot_theory_comparison')


class TestConvergencePlots:
    """Tests para gráficos de convergencia"""
    
    def test_plot_convergence_comparison(self, tmp_path):
        """Test generación de plot convergencia"""
        classical = np.exp(-0.02 * np.arange(100)) + np.random.randn(100) * 0.05
        quantum = np.exp(-0.05 * np.arange(100)) + np.random.randn(100) * 0.03
        output = str(tmp_path / "convergence.png")
        pc.plot_convergence_comparison(classical, quantum, output_path=output)
        assert os.path.exists(output) and os.path.getsize(output) > 0
    
    def test_plot_snr_convergence(self, tmp_path):
        """Test plot SNR vs convergencia"""
        snrs = [5, 10, 15, 20]
        classical = [500, 300, 150, 80]
        quantum = [200, 100, 50, 30]
        output = str(tmp_path / "snr_conv.png")
        pc.plot_snr_vs_convergence_time(snrs, classical, quantum, output_path=output)
        assert os.path.exists(output) and os.path.getsize(output) > 0
    
    def test_plot_walltime(self, tmp_path):
        """Test plot walltime"""
        snrs = [5, 10, 15, 20]
        classical_time = [100, 60, 30, 16]
        quantum_time = [50, 25, 12, 8]
        output = str(tmp_path / "walltime.png")
        pc.plot_walltime_comparison(snrs, classical_time, quantum_time, output_path=output)
        assert os.path.exists(output) and os.path.getsize(output) > 0


class TestPosteriorPlots:
    """Tests para gráficos de distribuciones posteriores"""
    
    @pytest.fixture
    def sample_data(self):
        """Datos de ejemplo"""
        n_samples = 5000
        samples = np.array([
            np.random.normal(30, 2, n_samples),
            np.random.normal(25, 1.5, n_samples),
            np.random.normal(0.5, 0.2, n_samples)
        ]).T
        param_names = ["M1", "M2", "chi"]
        true_values = [30.0, 25.0, 0.5]
        return samples, param_names, true_values
    
    def test_plot_posterior_distributions(self, sample_data, tmp_path):
        """Test gráfico distribuciones posteriores"""
        samples, param_names, true_values = sample_data
        output = str(tmp_path / "posterior.png")
        pp.plot_posterior_distributions(samples, param_names, true_values, output_path=output)
        assert os.path.exists(output) and os.path.getsize(output) > 0
    
    def test_plot_corner(self, sample_data, tmp_path):
        """Test corner plot"""
        samples, param_names, true_values = sample_data
        output = str(tmp_path / "corner.png")
        pp.plot_corner_plot(samples, param_names, true_values, output_path=output, nbins=20)
        assert os.path.exists(output) and os.path.getsize(output) > 0
    
    def test_plot_credible_intervals(self, sample_data, tmp_path):
        """Test credible intervals"""
        samples, param_names, true_values = sample_data
        samples_dict = {
            "GR": samples,
            "LQG": samples + np.random.normal(0, 0.5, samples.shape),
            "ECO": samples + np.random.normal(0, 1, samples.shape)
        }
        output = str(tmp_path / "credible.png")
        pp.plot_credible_intervals(samples_dict, param_names, true_values, output_path=output)
        assert os.path.exists(output) and os.path.getsize(output) > 0


class TestTheoryComparisonPlots:
    """Tests para gráficos de comparación de teorías"""
    
    def test_theory_accuracy_by_snr(self, tmp_path):
        """Test accuracy por SNR"""
        snrs = [5, 10, 15, 20]
        accuracies = {
            "GR": [0.3, 0.5, 0.7, 0.85],
            "LQG": [0.25, 0.4, 0.55, 0.7],
            "ECO": [0.2, 0.3, 0.4, 0.5]
        }
        output = str(tmp_path / "accuracy.png")
        pt.plot_theory_accuracy_by_snr(snrs, accuracies, output_path=output)
        assert os.path.exists(output) and os.path.getsize(output) > 0
    
    def test_log_odds(self, tmp_path):
        """Test log-odds ratios"""
        theories = ["GR", "LQG", "ECO"]
        log_odds_matrix = np.array([[0, 2, 3], [-2, 0, 1], [-3, -1, 0]], dtype=float)
        output = str(tmp_path / "log_odds.png")
        pt.plot_log_odds_ratios(theories, log_odds_matrix, output_path=output)
        assert os.path.exists(output) and os.path.getsize(output) > 0
    
    def test_confusion_matrix(self, tmp_path):
        """Test confusion matrix"""
        confusion = np.array([[45, 3, 2], [2, 40, 8], [3, 7, 40]])
        theories = ["GR", "LQG", "ECO"]
        output = str(tmp_path / "confusion.png")
        pt.plot_confusion_matrix(confusion, theories, output_path=output)
        assert os.path.exists(output) and os.path.getsize(output) > 0
    
    def test_feature_importance(self, tmp_path):
        """Test feature importance"""
        features = ["f1", "f2", "f3", "f4"]
        theories = ["GR", "LQG", "ECO"]
        importances = np.random.rand(3, 4)
        output = str(tmp_path / "importance.png")
        pt.plot_feature_importance(features, importances, theories, output_path=output)
        assert os.path.exists(output) and os.path.getsize(output) > 0
    
    def test_theory_separation(self, tmp_path):
        """Test theory separation"""
        snrs = [5, 10, 15, 20]
        separations = {
            "GR vs LQG": [0.1, 0.3, 0.6, 0.8],
            "GR vs ECO": [0.2, 0.4, 0.7, 0.9],
            "LQG vs ECO": [0.15, 0.35, 0.65, 0.85]
        }
        output = str(tmp_path / "separation.png")
        pt.plot_theory_separation(snrs, separations, output_path=output)
        assert os.path.exists(output) and os.path.getsize(output) > 0


class TestPlotIntegration:
    """Tests de integración PASO 5"""
    
    def test_all_convergence_plots_generation(self, tmp_path):
        """Test generación de todos los plots de convergencia"""
        classical = np.exp(-0.02 * np.arange(100))
        quantum = np.exp(-0.05 * np.arange(100))
        pc.plot_convergence_comparison(classical, quantum, output_path=str(tmp_path / "conv1.png"))
        
        snrs = [5, 10, 15, 20]
        pc.plot_snr_vs_convergence_time(snrs, [500, 300, 150, 80], [200, 100, 50, 30],
                                        output_path=str(tmp_path / "conv2.png"))
        pc.plot_walltime_comparison(snrs, [100, 60, 30, 16], [50, 25, 12, 8],
                                   output_path=str(tmp_path / "conv3.png"))
        
        files = list(tmp_path.glob("conv*.png"))
        assert len(files) == 3
    
    def test_all_theory_comparison_plots(self, tmp_path):
        """Test generación de todos los plots de comparación"""
        snrs = [5, 10, 15, 20]
        theories = ["GR", "LQG", "ECO"]
        
        accuracies = {t: [0.25 + 0.5*np.sin(i/5) for i in snrs] for t in theories}
        pt.plot_theory_accuracy_by_snr(snrs, accuracies, output_path=str(tmp_path / "cmp1.png"))
        
        log_odds = np.random.randn(3, 3)
        pt.plot_log_odds_ratios(theories, log_odds, output_path=str(tmp_path / "cmp2.png"))
        
        confusion = np.array([[40, 5, 5], [5, 40, 5], [5, 5, 40]])
        pt.plot_confusion_matrix(confusion, theories, output_path=str(tmp_path / "cmp3.png"))
        
        features = ["f1", "f2", "f3", "f4"]
        importances = np.random.rand(3, 4)
        pt.plot_feature_importance(features, importances, theories, output_path=str(tmp_path / "cmp4.png"))
        
        separations = {f"{t1} vs {t2}": [0.1, 0.3, 0.6, 0.8] 
                      for t1 in theories for t2 in theories if t1 < t2}
        pt.plot_theory_separation(snrs, separations, output_path=str(tmp_path / "cmp5.png"))
        
        files = list(tmp_path.glob("cmp*.png"))
        assert len(files) == 5


class TestPlotValidation:
    """Tests para validación de plots"""
    
    def test_plot_file_size_reasonable(self, tmp_path):
        """Test que los archivos generados tienen tamaño razonable"""
        classical = np.exp(-0.02 * np.arange(50))
        quantum = np.exp(-0.05 * np.arange(50))
        output = str(tmp_path / "test.png")
        pc.plot_convergence_comparison(classical, quantum, output_path=output)
        file_size = os.path.getsize(output)
        assert 5000 < file_size < 500000  # Entre 5KB y 500KB
    
    def test_plot_directory_creation(self, tmp_path):
        """Test que se crean directorios si no existen"""
        nested_path = str(tmp_path / "a" / "b" / "c" / "plot.png")
        classical = np.exp(-0.02 * np.arange(50))
        quantum = np.exp(-0.05 * np.arange(50))
        pc.plot_convergence_comparison(classical, quantum, output_path=nested_path)
        assert os.path.exists(nested_path)
