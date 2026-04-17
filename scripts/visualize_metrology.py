import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

def plot_fisher_ellipses(csv_path):
    df = pd.read_csv(csv_path)
    # Tomamos un evento de alta SNR (100 Mpc) para la visualización
    sample = df[df['dist_mpc'] == 100].iloc[0]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Parámetros para la elipse (Masa vs H0)
    # La matriz de Fisher dicta que la incertidumbre es inversamente proporcional a la SNR
    width = sample['mass_precision'] * 20
    height = (1 / sample['snr_est']) * 50
    
    # Dibujamos la elipse de confianza (1-Sigma)
    ellipse = Ellipse(
        xy=(30, sample['h0_inferred']), # (Masa típica, H0 inferido)
        width=width, height=height,
        angle=15, # Correlación entre masa y distancia
        edgecolor='blue', fc='blue', lw=2, alpha=0.3,
        label='Inferencia QNIM (1-σ Confidence)'
    )
    ax.add_patch(ellipse)
    
    # Referencias de la Tensión de Hubble
    plt.axhline(67.4, color='red', linestyle='--', label='Planck (Early Universe)')
    plt.axhline(73.0, color='green', linestyle='--', label='SH0ES (Late Universe)')
    
    plt.title(f"Metrología Gravitacional: Inferencia de H0 (SNR: {sample['snr_est']})")
    plt.xlabel("Masa del Chirp ($M_{\odot}$)")
    plt.ylabel("Constante de Hubble ($H_0$)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("fisher_metrology_plot.png")
    print("📈 Gráfica de Fisher generada en 'fisher_metrology_plot.png'")

if __name__ == "__main__":
    plot_fisher_ellipses("resultados_metrologia_planck.csv")