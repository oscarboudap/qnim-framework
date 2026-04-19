"""
Visualization: Comprehensive Statistical Validation Plots
=========================================================

Genera visualización profesional de:
- MC accuracy vs SNR
- Bootstrap confidence intervals
- Fisher matrix heatmaps
- Theory comparison
- No-hair theorem tests
- ROC curves
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List
import seaborn as sns


def plot_accuracy_vs_snr(
    mc_results_by_snr: Dict[float, List],
    theory_labels: List[str],
    output_file: str = "reports/validation/accuracy_vs_snr.png"
):
    """Plot accuracy como función de SNR (robustez)."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    snr_levels = sorted(mc_results_by_snr.keys())
    accuracies = []
    
    for snr in snr_levels:
        results = mc_results_by_snr[snr]
        correct = sum(1 for r in results if r.prediction_correct)
        acc = correct / len(results) if results else 0
        accuracies.append(acc)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(snr_levels, accuracies, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.fill_between(snr_levels, np.array(accuracies) - 0.05, np.array(accuracies) + 0.05,
                     alpha=0.2, color='#2E86AB')
    
    ax.set_xlabel('Signal-to-Noise Ratio (SNR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('QNIM Robustness: VQC Accuracy vs SNR', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Añadir benchmark
    ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Random chance (5 theories)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Gráfico guardado: {output_path}")
    plt.close()


def plot_bootstrap_confidence_intervals(
    bootstrap_results: Dict[str, Dict],
    output_file: str = "reports/validation/bootstrap_ci.png"
):
    """Plot intervalos de confianza bootstrap."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    param_names = list(bootstrap_results.keys())
    
    for idx, (param_name, stats) in enumerate(bootstrap_results.items()):
        ax = axes[idx]
        
        mean = stats['mean']
        ci_68_lower, ci_68_upper = stats['ci_68']['lower'], stats['ci_68']['upper']
        ci_95_lower, ci_95_upper = stats['ci_95']['lower'], stats['ci_95']['upper']
        
        # Plot intervals
        ax.barh(['68% CI', '95% CI'], 
                [ci_68_upper - ci_68_lower, ci_95_upper - ci_95_lower],
                left=[ci_68_lower, ci_95_lower],
                color=['#2E86AB', '#A23B72'],
                alpha=0.7)
        
        # Plot mean
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean:.3e}')
        
        ax.set_title(f'{param_name}', fontweight='bold')
        ax.set_xlabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Bootstrap Confidence Intervals (68% and 95%)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Gráfico guardado: {output_path}")
    plt.close()


def plot_theory_comparison(
    theory_comparison: Dict[str, Dict],
    output_file: str = "reports/validation/theory_comparison.png"
):
    """Plot comparación de teorías (accuracy + log-odds)."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    theories = list(theory_comparison.keys())
    accuracies = [theory_comparison[t]['accuracy'] for t in theories]
    log_odds = [theory_comparison[t]['log_odds_vs_gr'] for t in theories]
    
    # Plot 1: Accuracy
    colors = ['#06A77D' if t == 'GR' else '#D62839' for t in theories]
    ax1.bar(theories, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Classification Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Accuracy by Theory', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Random (5 theories)')
    ax1.legend()
    
    # Plot 2: Log-Odds (discrimination power)
    colors_odds = ['green' if x > 0 else 'red' for x in log_odds]
    ax2.barh(theories, log_odds, color=colors_odds, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Log-Odds vs GR', fontsize=11, fontweight='bold')
    ax2.set_title('Theory Discrimination Power', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('QNIM Theory Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Gráfico guardado: {output_path}")
    plt.close()


def plot_fisher_matrix_heatmap(
    fisher_matrix: np.ndarray,
    param_names: List[str],
    output_file: str = "reports/validation/fisher_matrix.png"
):
    """Plot heatmap de matriz de Fisher (degeneracies)."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalizar correlaciones
    try:
        fisher_inv = np.linalg.inv(fisher_matrix)
    except:
        fisher_inv = np.linalg.pinv(fisher_matrix)
    
    errors = np.sqrt(np.diag(fisher_inv))
    corr_matrix = np.zeros_like(fisher_inv)
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            if errors[i] > 0 and errors[j] > 0:
                corr_matrix[i, j] = fisher_inv[i, j] / (errors[i] * errors[j])
    
    # Heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Etiquetas
    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticklabels(param_names)
    
    # Valores en celdas
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Fisher Matrix: Parameter Correlations', fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Gráfico guardado: {output_path}")
    plt.close()


def plot_robustness_vs_noise(
    snr_levels: List[float],
    accuracies: List[float],
    std_errors: List[float] = None,
    output_file: str = "reports/validation/robustness_vs_snr.png"
):
    """Plot robustez con barras de error."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if std_errors is None:
        std_errors = [0.02] * len(snr_levels)
    
    ax.errorbar(snr_levels, accuracies, yerr=std_errors, 
                fmt='o-', capsize=5, capthick=2, markersize=8,
                linewidth=2, color='#2E86AB', ecolor='gray', alpha=0.8)
    
    ax.set_xlabel('Signal-to-Noise Ratio (SNR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classification Accuracy ± 1σ', fontsize=12, fontweight='bold')
    ax.set_title('QNIM Robustness: Accuracy vs SNR with Statistical Errors', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Region of interest
    ax.axhspan(0.8, 1.0, alpha=0.1, color='green', label='High Confidence (>80%)')
    ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Random Guess')
    
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Gráfico guardado: {output_path}")
    plt.close()


def plot_no_hair_theorem_test(
    ringdown_data: Dict,
    output_file: str = "reports/validation/no_hair_test.png"
):
    """Plot test del teorema no-cabello (Q observado vs predicho)."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    modes = sorted(ringdown_data.keys())
    q_obs = [ringdown_data[m]['observed_Q'] for m in modes]
    q_pred = [ringdown_data[m]['predicted_Q_kerr'] for m in modes]
    deviations = [ringdown_data[m]['deviation_fraction'] for m in modes]
    
    x = np.arange(len(modes))
    width = 0.35
    
    # Plot 1: Q observado vs predicho
    ax1.bar(x - width/2, q_obs, width, label='Observed', alpha=0.8, color='#2E86AB')
    ax1.bar(x + width/2, q_pred, width, label='Predicted (Kerr)', alpha=0.8, color='#D62839')
    
    ax1.set_ylabel('Quality Factor (Q)', fontsize=11, fontweight='bold')
    ax1.set_title('Ringdown Quality Factors: Observed vs Kerr Prediction', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Deviations from Kerr
    colors_dev = ['red' if abs(d) > 0.1 else 'green' for d in deviations]
    ax2.barh(modes, deviations, color=colors_dev, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axvline(x=0.1, color='red', linestyle='--', linewidth=2, label='New Physics Threshold (10%)')
    ax2.axvline(x=-0.1, color='red', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Deviation from Kerr (Δ)', fontsize=11, fontweight='bold')
    ax2.set_title('Evidence for Beyond-GR Physics', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('No-Hair Theorem Test: QNIM Ringdown Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Gráfico guardado: {output_path}")
    plt.close()


def generate_all_plots(results: Dict, output_dir: str = "reports/validation"):
    """Genera todos los plots."""
    print("📊 Generando visualizaciones...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Accuracy vs SNR
    if "mc_sweep" in results:
        plot_accuracy_vs_snr(
            results["mc_sweep"],
            output_file=str(output_path / "accuracy_vs_snr.png")
        )
    
    # Plot 2: Bootstrap CI
    if "bootstrap_uncertainties" in results:
        plot_bootstrap_confidence_intervals(
            results["bootstrap_uncertainties"],
            output_file=str(output_path / "bootstrap_ci.png")
        )
    
    # Plot 3: Theory comparison
    if "theory_comparison" in results:
        plot_theory_comparison(
            results["theory_comparison"],
            output_file=str(output_path / "theory_comparison.png")
        )
    
    print(f"  ✓ Visualizaciones completadas en: {output_dir}")
