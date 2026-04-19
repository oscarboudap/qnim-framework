"""
Script: Plot Theory Comparison

PASO 5: Visualización de comparación entre teorías (GR vs cuantización vs exóticas)
- Accuracy por SNR
- Log-odds ratios
- Feature importance heatmaps
- Theory classification confusion matrices
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List


def plot_theory_accuracy_by_snr(
    snr_levels: List[float],
    theory_accuracies: Dict[str, List[float]],
    output_path: str = "reports/figures/theory_accuracy_by_snr.png"
) -> None:
    """
    Gráfico: Accuracy de clasificación de teorías vs SNR.
    
    Args:
        snr_levels: Niveles de SNR
        theory_accuracies: {theory: [accuracy_snr1, accuracy_snr2, ...]}
        output_path: Ruta de salida
    """
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(theory_accuracies)))
    
    for (theory, accuracies), color in zip(theory_accuracies.items(), colors):
        ax.plot(snr_levels, accuracies, 'o-', label=theory, 
               linewidth=2.5, markersize=8, color=color, markeredgewidth=1.5)
    
    # Referencia: random chance
    n_theories = len(theory_accuracies)
    random_chance = 1 / n_theories
    ax.axhline(random_chance, color='gray', linestyle='--', linewidth=2, 
              alpha=0.7, label=f'Random ({random_chance:.1%})')
    
    ax.set_xlabel('SNR', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Theory Classification Accuracy vs SNR', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_log_odds_ratios(
    theories: List[str],
    log_odds: np.ndarray,  # [n_theories, n_theories]
    reference_theory: str = "GR",
    output_path: str = "reports/figures/log_odds_ratios.png"
) -> None:
    """
    Heatmap de log-odds ratios entre teorías.
    
    Args:
        theories: Nombres de teorías
        log_odds: Matriz [n_theories, n_theories] de log-odds
        reference_theory: Teoría de referencia
        output_path: Ruta de salida
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Heatmap
    im = ax.imshow(log_odds, cmap='RdBu_r', aspect='auto', vmin=-5, vmax=5)
    
    # Etiquetas
    ax.set_xticks(np.arange(len(theories)))
    ax.set_yticks(np.arange(len(theories)))
    ax.set_xticklabels(theories, fontsize=11)
    ax.set_yticklabels(theories, fontsize=11)
    
    # Rotar etiquetas X
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Valores en celdas
    for i in range(len(theories)):
        for j in range(len(theories)):
            text = ax.text(j, i, f'{log_odds[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_xlabel('Theory', fontsize=12, fontweight='bold')
    ax.set_ylabel('Theory', fontsize=12, fontweight='bold')
    ax.set_title('Log-Odds Ratios Between Theories\n(Positive = row theory favored)', 
                fontsize=13, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Log-Odds')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_confusion_matrix(
    confusion: np.ndarray,  # [n_theories, n_theories]
    theory_names: List[str],
    output_path: str = "reports/figures/confusion_matrix.png"
) -> None:
    """
    Matriz de confusión de clasificación de teorías.
    
    Args:
        confusion: Matriz confusión [n_theories, n_theories]
        theory_names: Nombres de teorías
        output_path: Ruta de salida
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalizar por fila
    confusion_normalized = confusion / confusion.sum(axis=1, keepdims=True)
    
    # Heatmap
    im = ax.imshow(confusion_normalized, cmap='Blues', aspect='auto')
    
    # Etiquetas
    ax.set_xticks(np.arange(len(theory_names)))
    ax.set_yticks(np.arange(len(theory_names)))
    ax.set_xticklabels(theory_names, fontsize=11)
    ax.set_yticklabels(theory_names, fontsize=11)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Valores
    for i in range(len(theory_names)):
        for j in range(len(theory_names)):
            pct = confusion_normalized[i, j]
            count = confusion[i, j]
            text = ax.text(j, i, f'{count}\n({pct:.1%})',
                          ha="center", va="center", 
                          color="white" if pct > 0.5 else "black", 
                          fontsize=10)
    
    ax.set_xlabel('Predicted Theory', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Theory', fontsize=12, fontweight='bold')
    ax.set_title('Theory Classification Confusion Matrix\n(normalized by row)', 
                fontsize=13, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Fraction')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,  # [n_theories, n_features]
    theory_names: List[str] = None,
    output_path: str = "reports/figures/feature_importance.png"
) -> None:
    """
    Importance de features para distinguir teorías.
    
    Args:
        feature_names: Nombres de features
        importances: Matrix [n_theories, n_features]
        theory_names: Nombres de teorías
        output_path: Ruta de salida
    """
    
    if theory_names is None:
        theory_names = [f"Theory {i}" for i in range(importances.shape[0])]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(feature_names))
    width = 0.8 / len(theory_names)
    
    for i, theory in enumerate(theory_names):
        offset = width * (i - len(theory_names)/2 + 0.5)
        ax.bar(x + offset, importances[i], width, label=theory, alpha=0.8)
    
    ax.set_xlabel('Feature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance for Theory Discrimination', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_theory_separation(
    snr_levels: List[float],
    separations: Dict[str, List[float]],  # {pair: [sep_snr1, sep_snr2, ...]}
    output_path: str = "reports/figures/theory_separation.png"
) -> None:
    """
    Separabilidad de teorías vs SNR.
    
    Args:
        snr_levels: Niveles de SNR
        separations: {theory_pair: separabilities}
        output_path: Ruta de salida
    """
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(separations)))
    
    for (pair, seps), color in zip(separations.items(), colors):
        ax.plot(snr_levels, seps, 'o-', label=pair,
               linewidth=2.5, markersize=8, color=color, markeredgewidth=1.5)
    
    ax.set_xlabel('SNR', fontsize=12, fontweight='bold')
    ax.set_ylabel('Separation Metric (higher = more distinguishable)', fontsize=12, fontweight='bold')
    ax.set_title('Theory Separability vs SNR', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, None])
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    """Genera plots de ejemplo"""
    
    np.random.seed(42)
    
    print("📊 Generando comparación de teorías...")
    
    # 1. Accuracy por SNR
    snr_levels = [5, 10, 15, 20, 30, 50]
    theories = ["GR", "LQG", "ECO", "String"]
    
    theory_accuracies = {}
    for theory in theories:
        # Accuracy sigue sigmoide con SNR
        accs = 0.25 + 0.75 * (1 / (1 + np.exp(-0.5 * (np.array(snr_levels) - 15))))
        accs += np.random.randn(len(snr_levels)) * 0.05
        theory_accuracies[theory] = np.clip(accs, 0.2, 0.99).tolist()
    
    plot_theory_accuracy_by_snr(snr_levels, theory_accuracies)
    
    # 2. Log-odds ratios
    n_theories = len(theories)
    log_odds = np.random.randn(n_theories, n_theories) * 2
    log_odds = (log_odds - log_odds.T)  # Antisimétrica
    
    plot_log_odds_ratios(theories, log_odds)
    
    # 3. Confusion matrix
    confusion = np.array([
        [45, 3, 2, 0],
        [2, 40, 5, 3],
        [3, 8, 35, 4],
        [1, 2, 5, 42]
    ])
    
    plot_confusion_matrix(confusion, theories)
    
    # 4. Feature importance
    features = ['f_peak', 'bandwidth', 'SNR', 'chirp_mass', 'ringdown_Q', 'spin']
    importances = np.random.rand(len(theories), len(features))
    
    plot_feature_importance(features, importances, theories)
    
    # 5. Theory separation
    separations = {}
    for i, t1 in enumerate(theories):
        for t2 in theories[i+1:]:
            pair = f"{t1} vs {t2}"
            seps = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0]) + np.random.randn(6) * 0.05
            separations[pair] = np.clip(seps, 0, 1).tolist()
    
    plot_theory_separation(snr_levels, separations)
    
    print("\n✅ Todos los plots de comparación generados exitosamente!")


if __name__ == "__main__":
    main()
