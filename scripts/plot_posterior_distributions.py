"""
Script: Plot Posterior Distributions

PASO 5: Visualización de distribuciones posteriores de parámetros
- Corner plot (pairwise correlations)
- 1D histogramas
- Credible intervals (68% y 95%)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path
from typing import List, Dict, Tuple


def plot_posterior_distributions(
    samples: np.ndarray,  # [n_samples, n_params]
    param_names: List[str],
    true_values: List[float] = None,
    output_path: str = "reports/figures/posterior_distributions.png"
) -> None:
    """
    Gráfico de distribuciones posteriores 1D con credible intervals.
    
    Args:
        samples: Muestras MCMC/variacional [n_samples, n_params]
        param_names: Nombres de parámetros
        true_values: Valores verdaderos (si disponibles)
        output_path: Ruta de salida
    """
    
    n_params = samples.shape[1]
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for i in range(n_params):
        ax = axes[i]
        param_samples = samples[:, i]
        
        # Histogram
        ax.hist(param_samples, bins=50, density=True, alpha=0.7, 
                color='#1f77b4', edgecolor='black', linewidth=0.5)
        
        # KDE overlay
        kde = stats.gaussian_kde(param_samples)
        x_range = np.linspace(param_samples.min(), param_samples.max(), 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2.5, label='KDE')
        
        # Credible intervals
        ci_16, ci_50, ci_84 = np.percentile(param_samples, [16, 50, 84])
        ci_025, ci_975 = np.percentile(param_samples, [2.5, 97.5])
        
        # Marcar intervalos
        ax.axvline(ci_50, color='green', linestyle='-', linewidth=2.5, 
                   label=f'Median: {ci_50:.3f}')
        ax.axvline(ci_16, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(ci_84, color='orange', linestyle='--', linewidth=2, alpha=0.7, 
                   label=f'68% CI: [{ci_16:.3f}, {ci_84:.3f}]')
        
        # Valor verdadero (si disponible)
        if true_values is not None and i < len(true_values):
            ax.axvline(true_values[i], color='red', linestyle=':', linewidth=3, 
                       label=f'True: {true_values[i]:.3f}')
        
        ax.set_xlabel(param_names[i], fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # Ocultar ejes extras
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_corner_plot(
    samples: np.ndarray,  # [n_samples, n_params]
    param_names: List[str],
    true_values: List[float] = None,
    output_path: str = "reports/figures/corner_plot.png",
    nbins: int = 30
) -> None:
    """
    Corner plot (matriz de correlaciones pairwise).
    
    Diagonal: distribuciones 1D
    Triángulo inferior: scatterplots 2D
    
    Args:
        samples: Muestras MCMC [n_samples, n_params]
        param_names: Nombres de parámetros
        true_values: Valores verdaderos (opcional)
        output_path: Ruta de salida
        nbins: Bins para histogramas
    """
    
    n_params = samples.shape[1]
    
    fig, axes = plt.subplots(n_params, n_params, figsize=(3*n_params, 3*n_params))
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histograma 1D
                ax.hist(samples[:, i], bins=nbins, density=True, alpha=0.7, 
                        color='#1f77b4', edgecolor='black', linewidth=0.5)
                
                # KDE
                kde = stats.gaussian_kde(samples[:, i])
                x = np.linspace(samples[:, i].min(), samples[:, i].max(), 100)
                ax.plot(x, kde(x), 'r-', linewidth=2)
                
                # Valor verdadero
                if true_values is not None:
                    ax.axvline(true_values[i], color='red', linestyle=':', linewidth=2)
                
                ax.set_ylabel('Density', fontsize=10)
                
            elif i > j:
                # Triángulo inferior: scatter 2D con densidad
                ax.scatter(samples[:, j], samples[:, i], alpha=0.3, s=10, color='#1f77b4')
                
                # Contour de densidad
                try:
                    from scipy.stats import gaussian_kde
                    xy = np.vstack([samples[:, j], samples[:, i]])
                    kde = gaussian_kde(xy)
                    
                    x_min, x_max = samples[:, j].min(), samples[:, j].max()
                    y_min, y_max = samples[:, i].min(), samples[:, i].max()
                    
                    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    f = np.reshape(kde(positions).T, xx.shape)
                    
                    ax.contour(xx, yy, f, colors='red', levels=3, alpha=0.5)
                except:
                    pass
                
                # Valores verdaderos
                if true_values is not None:
                    ax.plot(true_values[j], true_values[i], 'r*', markersize=15)
                
            else:
                # Triángulo superior: vacío
                ax.set_visible(False)
                continue
            
            # Etiquetas
            if i == n_params - 1:
                ax.set_xlabel(param_names[j], fontsize=10)
            else:
                ax.set_xticklabels([])
            
            if j == 0 and i != j:
                ax.set_ylabel(param_names[i], fontsize=10)
            elif i != j:
                ax.set_yticklabels([])
            
            ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_credible_intervals(
    samples: Dict[str, np.ndarray],  # {theory: [n_samples, n_params]}
    param_names: List[str],
    true_values: List[float] = None,
    output_path: str = "reports/figures/credible_intervals.png"
) -> None:
    """
    Comparación de credible intervals entre teorías.
    
    Args:
        samples: Dictionary {theory_name: samples}
        param_names: Nombres de parámetros
        true_values: Valores verdaderos (opcional)
        output_path: Ruta de salida
    """
    
    theories = list(samples.keys())
    n_params = samples[theories[0]].shape[1]
    
    fig, axes = plt.subplots(1, n_params, figsize=(4*n_params, 6))
    if n_params == 1:
        axes = [axes]
    
    for param_idx, ax in enumerate(axes):
        
        y_pos = 0
        y_ticks = []
        y_labels = []
        
        for theory_idx, theory_name in enumerate(theories):
            theory_samples = samples[theory_name][:, param_idx]
            
            mean = np.mean(theory_samples)
            ci_16, ci_84 = np.percentile(theory_samples, [16, 84])
            
            # Plot intervalo
            ax.plot([ci_16, ci_84], [y_pos, y_pos], 'o-', linewidth=3, 
                   markersize=8, color=f'C{theory_idx}', label=theory_name)
            ax.plot(mean, y_pos, 'D', markersize=10, color=f'C{theory_idx}')
            
            y_ticks.append(y_pos)
            y_labels.append(theory_name)
            y_pos += 1
        
        # Valor verdadero
        if true_values is not None and param_idx < len(true_values):
            ax.axvline(true_values[param_idx], color='red', linestyle='--', 
                      linewidth=2.5, alpha=0.7, label='True value')
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel(param_names[param_idx], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        if param_idx == 0:
            ax.legend(fontsize=10, loc='best')
    
    plt.suptitle('Credible Intervals (68%) by Theory', fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    """Genera plots de ejemplo"""
    
    np.random.seed(42)
    
    print("📊 Generando distribuciones posteriores...")
    
    # Datos sintéticos: 3 parámetros
    n_samples = 10000
    samples = np.array([
        np.random.normal(30, 2, n_samples),      # Masa1
        np.random.normal(25, 1.5, n_samples),    # Masa2
        np.random.normal(0.5, 0.2, n_samples)    # Spin
    ]).T
    
    param_names = [r'$M_1$ [M$_\odot$]', r'$M_2$ [M$_\odot$]', r'$\chi_{eff}$']
    true_values = [30.0, 25.0, 0.5]
    
    # Plot 1: Distribuciones posteriores
    plot_posterior_distributions(samples, param_names, true_values)
    
    # Plot 2: Corner plot
    plot_corner_plot(samples, param_names, true_values)
    
    # Plot 3: Comparación entre teorías
    samples_dict = {
        "GR": samples,
        "LQG": samples + np.random.normal(0, 0.5, samples.shape),
        "ECO": samples + np.random.normal(0, 1.0, samples.shape)
    }
    
    plot_credible_intervals(samples_dict, param_names, true_values)
    
    print("\n✅ Todos los plots generados exitosamente!")


if __name__ == "__main__":
    main()
