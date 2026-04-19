"""
Script: Plot Convergence Comparison (Classical vs Quantum)

PASO 5: Genera gráficos de convergencia comparando:
- MCMC clásico (Metropolis-Hastings)
- Kernel cuántico (VQC)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple


def plot_convergence_comparison(
    classical_results: np.ndarray,  # [n_iterations]
    quantum_results: np.ndarray,    # [n_iterations]
    output_path: str = "reports/figures/convergence_comparison.png",
    title: str = "Convergence: Classical MCMC vs Quantum VQC"
) -> None:
    """
    Gráfico de convergencia clásica vs cuántica.
    
    Args:
        classical_results: Array de errores/log-likelihood clásico
        quantum_results: Array de errores/log-likelihood cuántico
        output_path: Ruta de salida
        title: Título del gráfico
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Panel 1: Convergencia por iteración ---
    iterations = np.arange(len(classical_results))
    
    ax1.plot(iterations, classical_results, 'o-', color='#1f77b4', 
             label='Classical MCMC', linewidth=2, markersize=4, alpha=0.8)
    ax1.plot(iterations, quantum_results, 's-', color='#ff7f0e', 
             label='Quantum VQC', linewidth=2, markersize=4, alpha=0.8)
    
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Error / Negative Log-Likelihood', fontsize=12, fontweight='bold')
    ax1.set_title('Convergence Trajectory', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # --- Panel 2: Convergencia acumulativa (moving average) ---
    window = max(1, len(classical_results) // 10)
    
    classical_ma = np.convolve(classical_results, np.ones(window)/window, mode='valid')
    quantum_ma = np.convolve(quantum_results, np.ones(window)/window, mode='valid')
    
    iterations_ma = np.arange(len(classical_ma))
    
    ax2.plot(iterations_ma, classical_ma, 'o-', color='#1f77b4', 
             label=f'Classical (MA-{window})', linewidth=2.5, markersize=5, alpha=0.8)
    ax2.plot(iterations_ma, quantum_ma, 's-', color='#ff7f0e', 
             label=f'Quantum (MA-{window})', linewidth=2.5, markersize=5, alpha=0.8)
    
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Moving Average Error', fontsize=12, fontweight='bold')
    ax2.set_title('Smoothed Convergence', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Estadísticas
    classical_convergence_time = np.argmin(classical_results)
    quantum_convergence_time = np.argmin(quantum_results)
    speedup = classical_convergence_time / (quantum_convergence_time + 1e-10)
    
    fig.suptitle(
        f'{title}\n'
        f'Classical convergence @ iter {classical_convergence_time} | '
        f'Quantum @ {quantum_convergence_time} | '
        f'Speedup: {speedup:.1f}x',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_snr_vs_convergence_time(
    snr_levels: List[float],
    classical_times: List[int],
    quantum_times: List[int],
    output_path: str = "reports/figures/snr_vs_convergence.png"
) -> None:
    """
    Gráfico: SNR vs. tiempo de convergencia
    
    Args:
        snr_levels: Niveles de SNR
        classical_times: Iteraciones hasta convergencia (clásico)
        quantum_times: Iteraciones hasta convergencia (cuántico)
        output_path: Ruta de salida
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(snr_levels, classical_times, 'o-', color='#1f77b4', 
            label='Classical MCMC', linewidth=2.5, markersize=8, markeredgewidth=2)
    ax.plot(snr_levels, quantum_times, 's-', color='#ff7f0e', 
            label='Quantum VQC', linewidth=2.5, markersize=8, markeredgewidth=2)
    
    # Llenar entre curvas
    ax.fill_between(snr_levels, classical_times, quantum_times, 
                     alpha=0.2, color='green', label='Quantum advantage')
    
    ax.set_xlabel('SNR', fontsize=12, fontweight='bold')
    ax.set_ylabel('Iterations to Convergence', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Time vs SNR', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_walltime_comparison(
    snr_levels: List[float],
    classical_walltime: List[float],
    quantum_walltime: List[float],
    output_path: str = "reports/figures/walltime_comparison.png"
) -> None:
    """
    Gráfico: Tiempo de pared (CPU/GPU) vs SNR
    
    Args:
        snr_levels: Niveles de SNR
        classical_walltime: Tiempo en segundos (clásico)
        quantum_walltime: Tiempo en segundos (cuántico)
        output_path: Ruta de salida
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Tiempos absolutos
    ax1.semilogy(snr_levels, classical_walltime, 'o-', color='#1f77b4', 
                 label='Classical MCMC', linewidth=2.5, markersize=8)
    ax1.semilogy(snr_levels, quantum_walltime, 's-', color='#ff7f0e', 
                 label='Quantum VQC', linewidth=2.5, markersize=8)
    
    ax1.set_xlabel('SNR', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Walltime [seconds]', fontsize=12, fontweight='bold')
    ax1.set_title('Absolute Walltime', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Panel 2: Speedup
    speedup = np.array(classical_walltime) / (np.array(quantum_walltime) + 1e-10)
    
    ax2.plot(snr_levels, speedup, 'D-', color='#2ca02c', 
             linewidth=2.5, markersize=8, markeredgewidth=2)
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No speedup')
    ax2.fill_between(snr_levels, speedup, 1, where=(speedup > 1), 
                      alpha=0.2, color='green', label='Quantum faster')
    
    ax2.set_xlabel('SNR', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup Factor (Classical / Quantum)', fontsize=12, fontweight='bold')
    ax2.set_title('Walltime Speedup', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    """Genera plots de ejemplo"""
    
    # Datos sintéticos para exemplo
    np.random.seed(42)
    
    # Convergencia típica (exponencial decaimiento)
    iterations = np.arange(100)
    classical_results = 10 * np.exp(-0.02 * iterations) + np.random.randn(100) * 0.1
    quantum_results = 5 * np.exp(-0.05 * iterations) + np.random.randn(100) * 0.05
    
    print("📊 Generando gráficos de convergencia...")
    plot_convergence_comparison(classical_results, quantum_results)
    
    # SNR vs convergence time
    snr_levels = [5, 10, 15, 20, 30, 50]
    classical_times = [500, 300, 150, 80, 40, 20]
    quantum_times = [200, 100, 50, 30, 15, 8]
    
    print("📊 Generando gráfico SNR vs convergencia...")
    plot_snr_vs_convergence_time(snr_levels, classical_times, quantum_times)
    
    # Walltime comparison
    classical_walltime = [100, 60, 30, 16, 8, 4]  # segundos
    quantum_walltime = [50, 25, 12, 8, 4, 2]
    
    print("📊 Generando gráfico de tiempo de pared...")
    plot_walltime_comparison(snr_levels, classical_walltime, quantum_walltime)
    
    print("\n✅ Todos los gráficos generados exitosamente!")


if __name__ == "__main__":
    main()
