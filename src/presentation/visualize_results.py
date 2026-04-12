import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_training_results(loss_history, output_path="reports/figures/loss_curve.png"):
    """Genera la curva de aprendizaje (Loss Curve) para el TFM."""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='SPSA Loss', color='teal', lw=2)
    plt.title('Evolución del Aprendizaje del VQC (12 Cúbits)', fontsize=14)
    plt.xlabel('Iteraciones (Evaluaciones)', fontsize=12)
    plt.ylabel('Función de Coste (Binary Cross-Entropy)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(output_path)
    print(f"📊 Curva de pérdida guardada en: {output_path}")

def plot_corner_results(mass_samples, spin_samples, output_path="reports/figures/corner_plot.png"):
    """Genera un Corner Plot simplificado para la metrología de D-Wave."""
    sns.set_theme(style="white")
    g = sns.JointGrid(x=mass_samples, y=spin_samples, space=0)
    g.plot_joint(sns.kdeplot, fill=True, thresh=0, levels=10, cmap="viridis")
    g.plot_marginals(sns.histplot, color="teal", alpha=0.4, bins=15)
    g.set_axis_labels('Masa ($M_{\odot}$)', 'Espín ($a^*$)', fontsize=12)
    plt.savefig(output_path)
    print(f"🌌 Corner Plot guardado en: {output_path}")

if __name__ == "__main__":
    # Datos de ejemplo basados en tu última ejecución de 300 evaluaciones
    fake_loss = np.exp(-np.linspace(0, 5, 300)) + np.random.normal(0, 0.05, 300)
    plot_training_results(fake_loss)
    
    # Simulación de la exploración de D-Wave alrededor de 36.2 y 0.0
    m_samples = np.random.normal(36.2, 1.5, 1000)
    s_samples = np.random.normal(0.0, 0.1, 1000)
    plot_corner_results(m_samples, s_samples)