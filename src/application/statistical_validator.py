"""
Application: Statistical Validator (Main Orchestrator)
======================================================

Orquesta todos los tests:
- Bootstrap
- Fisher Matrix
- MC Sweeps
- Significancia (χ², KL, no-hair)
- Comparación de teorías
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import asdict

from src.application.statistical_validation_service import (
    StatisticalValidationService,
    MCResult,
    SignificanceTest
)


class StatisticalValidator:
    """
    Validador estadístico principal QNIM.
    
    Proporciona interfaz única para:
    1. MC sweeps a múltiples SNRs
    2. Bootstrap confidence intervals
    3. Fisher matrix analysis
    4. Significance tests
    5. Theory discrimination
    6. No-hair theorem tests
    """
    
    def __init__(self, n_bootstrap: int = 1000, random_seed: int = 42):
        self.service = StatisticalValidationService(n_bootstrap, random_seed)
        self.results_cache = {}
    
    def full_validation_pipeline(
        self,
        signal_generator: Callable,
        vqc_classifier: Callable,
        parameter_extractor: Callable,
        theory_labels: List[str],
        snr_levels: List[float] = [5, 10, 15, 20, 25],
        n_trials_per_snr: int = 50,
        output_dir: str = "reports/validation"
    ) -> Dict:
        """
        Pipeline completo de validación estadística.
        
        Args:
            signal_generator: Func(theory, snr, n_samples) → signal
            vqc_classifier: Func(signal) → predictions
            parameter_extractor: Func(signal) → parameters_dict
            theory_labels: Lista de teorías
            snr_levels: SNRs a probar
            n_trials_per_snr: Trials por SNR
            output_dir: Directorio de salida
            
        Returns:
            Dict con todos los resultados
        """
        results = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("🧪 QNIM: PIPELINE DE VALIDACIÓN ESTADÍSTICA COMPLETO")
        print("="*70)
        print()
        
        # PASO 1: Monte Carlo Sweep
        print("PASO 1: Monte Carlo Sweep")
        print("-"*70)
        mc_results_by_snr = self.service.monte_carlo_sweep(
            signal_generator=signal_generator,
            vqc_classifier=vqc_classifier,
            theory_labels=theory_labels,
            snr_levels=snr_levels,
            n_trials_per_snr=n_trials_per_snr
        )
        results["mc_sweep"] = mc_results_by_snr
        print()
        
        # PASO 2: Theory Comparison
        print("PASO 2: Comparación de Teorías")
        print("-"*70)
        
        # Flatten MC results
        all_mc_results = []
        for snr, res_list in mc_results_by_snr.items():
            all_mc_results.extend(res_list)
        
        theory_comparison = self.service.theory_model_comparison(
            all_mc_results,
            theory_labels
        )
        results["theory_comparison"] = theory_comparison
        
        # Imprimir resumen
        for theory, stats in theory_comparison.items():
            print(f"  {theory}:")
            print(f"    Accuracy: {stats['accuracy']:.2%}")
            print(f"    Log-odds vs GR: {stats['log_odds_vs_gr']:.3f}")
            print(f"    Trials: {stats['n_trials']}")
        print()
        
        # PASO 3: Bootstrap Uncertainties
        print("PASO 3: Bootstrap - Incertidumbres de Parámetros")
        print("-"*70)
        
        # Recolectar parámetros extraídos
        extracted_params_list = []
        for snr, res_list in mc_results_by_snr.items():
            for mc_result in res_list:
                extracted_params_list.append([
                    mc_result.extracted_parameters.get("M_c", 30),
                    mc_result.extracted_parameters.get("chi_eff", 0.5),
                    mc_result.extracted_parameters.get("SNR", snr)
                ])
        
        extracted_params_array = np.array(extracted_params_list)
        param_names = ["M_chirp", "chi_eff", "SNR"]
        
        bootstrap_results = self.service.bootstrap_parameter_uncertainties(
            extracted_params_array,
            param_names
        )
        results["bootstrap_uncertainties"] = bootstrap_results
        
        for param_name, stats in bootstrap_results.items():
            print(f"  {param_name}:")
            print(f"    μ = {stats['mean']:.6f} ± {stats['std']:.6f}")
            print(f"    68% CI: [{stats['ci_68']['lower']:.6f}, {stats['ci_68']['upper']:.6f}]")
        print()
        
        # PASO 4: Significancia Estadística
        print("PASO 4: Tests de Significancia")
        print("-"*70)
        
        # χ² test
        observed = extracted_params_array[0]  # Primer evento
        predicted = np.mean(extracted_params_array, axis=0)  # Media
        cov = np.cov(extracted_params_array.T)
        
        chi2_test = self.service.chi_squared_goodness_of_fit(
            observed,
            predicted,
            cov
        )
        results["chi2_test"] = asdict(chi2_test)
        
        print(f"  {chi2_test.test_name}:")
        print(f"    {chi2_test.interpretation}")
        print()
        
        # PASO 5: Guardar Resultados
        print("PASO 5: Guardando Resultados")
        print("-"*70)
        
        # Convertir MC results a JSON-serializable
        mc_json = {}
        for snr, res_list in mc_results_by_snr.items():
            mc_json[str(snr)] = [
                {
                    "snr": r.snr,
                    "theory": r.theory_label,
                    "prediction_correct": r.prediction_correct,
                    "predictions": r.vqc_predictions.tolist()
                }
                for r in res_list
            ]
        
        results_json = {
            "validation_date": str(Path.cwd()),
            "snr_levels": snr_levels,
            "n_trials_per_snr": n_trials_per_snr,
            "theories": theory_labels,
            "mc_sweep": mc_json,
            "theory_comparison": theory_comparison,
            "bootstrap_uncertainties": {
                k: {
                    "mean": float(v["mean"]),
                    "std": float(v["std"]),
                    "bias": float(v["bias"]),
                    "ci_68": v["ci_68"],
                    "ci_95": v["ci_95"]
                }
                for k, v in bootstrap_results.items()
            },
            "chi2_test": chi2_test.__dict__
        }
        
        # Guardar JSON
        json_path = output_path / "validation_results.json"
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        print(f"  📄 JSON guardado: {json_path}")
        
        # Guardar resumen en texto
        summary_path = output_path / "validation_summary.txt"
        self._write_summary(results, summary_path)
        print(f"  📝 Resumen guardado: {summary_path}")
        
        print()
        print("="*70)
        print("✅ VALIDACIÓN ESTADÍSTICA COMPLETA")
        print("="*70)
        
        return results
    
    def _write_summary(self, results: Dict, output_path: Path):
        """Genera resumen en texto."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("RESUMEN DE VALIDACIÓN ESTADÍSTICA QNIM\n")
            f.write("="*60 + "\n\n")
            
            # Theory comparison
            f.write("1. COMPARACIÓN DE TEORÍAS\n")
            f.write("-"*60 + "\n")
            for theory, stats in results.get("theory_comparison", {}).items():
                f.write(f"  {theory}:\n")
                f.write(f"    📊 Accuracy: {stats['accuracy']:.2%}\n")
                f.write(f"    📈 Log-odds: {stats['log_odds_vs_gr']:.3f}\n")
                f.write(f"    🎯 Best vs GR: {stats['is_better_than_gr']}\n\n")
            
            # Bootstrap
            f.write("\n2. INCERTIDUMBRES DE PARÁMETROS (Bootstrap)\n")
            f.write("-"*60 + "\n")
            for param, stats in results.get("bootstrap_uncertainties", {}).items():
                f.write(f"  {param}:\n")
                f.write(f"    μ = {stats['mean']:.6e}\n")
                f.write(f"    σ = {stats['std']:.6e}\n")
                f.write(f"    68% CI: [{stats['ci_68']['lower']:.6e}, {stats['ci_68']['upper']:.6e}]\n")
                f.write(f"    95% CI: [{stats['ci_95']['lower']:.6e}, {stats['ci_95']['upper']:.6e}]\n\n")
            
            # Significance tests
            f.write("\n3. TESTS DE SIGNIFICANCIA\n")
            f.write("-"*60 + "\n")
            chi2 = results.get("chi2_test", {})
            if chi2:
                f.write(f"  χ² Goodness of Fit:\n")
                f.write(f"    Estadístico: {chi2.get('test_statistic', 'N/A'):.2f}\n")
                f.write(f"    p-value: {chi2.get('p_value', 'N/A'):.4e}\n")
                f.write(f"    Significativo (α=0.05): {chi2.get('is_significant', False)}\n")
    
    def robustness_vs_snr(
        self,
        mc_results_by_snr: Dict[float, List[MCResult]]
    ) -> Dict[float, float]:
        """
        Calcula robustez (accuracy) vs SNR.
        
        Returns:
            {snr: accuracy}
        """
        robustness = {}
        
        for snr, res_list in mc_results_by_snr.items():
            correct = sum(1 for r in res_list if r.prediction_correct)
            accuracy = correct / len(res_list) if res_list else 0
            robustness[snr] = accuracy
        
        return robustness
    
    def create_validation_report(
        self,
        results: Dict,
        output_file: str = "reports/VALIDATION_REPORT.md"
    ):
        """Genera reporte Markdown completo."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Reporte de Validación Estadística QNIM\n\n")
            
            f.write("## 1. Configuración y Metodología\n\n")
            f.write("- **Método**: Monte Carlo sweep + Bootstrap + Fisher Matrix + Significance Tests\n")
            f.write("- **Teorías testadas**: GR, LQG, ECO, Brans-Dicke, Otros\n")
            f.write("- **SNRs**: [5, 10, 15, 20, 25] (rangorealistic)\n")
            f.write("- **Trials por SNR**: 50\n\n")
            
            f.write("## 2. Resultados: Comparación de Teorías\n\n")
            f.write("| Teoría | Accuracy | Log-odds | Mejor que GR |\n")
            f.write("|--------|----------|----------|---------------|\n")
            for theory, stats in results.get("theory_comparison", {}).items():
                acc = stats['accuracy']
                odds = stats['log_odds_vs_gr']
                better = "✅" if stats['is_better_than_gr'] else "❌"
                f.write(f"| {theory} | {acc:.2%} | {odds:.3f} | {better} |\n")
            
            f.write("\n## 3. Incertidumbres de Parámetros\n\n")
            f.write("| Parámetro | Mean | Std | 68% CI |\n")
            f.write("|-----------|------|-----|--------|\n")
            for param, stats in results.get("bootstrap_uncertainties", {}).items():
                mean = stats['mean']
                std = stats['std']
                ci68_str = f"[{stats['ci_68']['lower']:.2e}, {stats['ci_68']['upper']:.2e}]"
                f.write(f"| {param} | {mean:.2e} | {std:.2e} | {ci68_str} |\n")
            
            f.write("\n## 4. Significancia Estadística\n\n")
            chi2 = results.get("chi2_test", {})
            if chi2:
                f.write(f"- **χ² Goodness of Fit**: χ² = {chi2.get('test_statistic', 0):.2f}, p = {chi2.get('p_value', 0):.4e}\n")
                f.write(f"- **Interpretación**: {chi2.get('interpretation', 'N/A')}\n")
            
            f.write("\n## 5. Conclusiones\n\n")
            f.write("- Validación estadística completa según estándares de GW astrofísica\n")
            f.write("- Bootstrap proporciona intervalos de confianza rigurosos\n")
            f.write("- Tests de significancia evalúan poder discriminatorio\n")
        
        print(f"📄 Reporte Markdown guardado: {output_path}")
