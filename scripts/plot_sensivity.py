import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_sensitivity_plot(csv_path="reports/resultados_robustez_qnim.csv"):
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Dibujamos la evolución de Sigma
    sns.lineplot(data=df, x="SNR", y="Z_Score_Sigma", hue="True_Label", marker="o", linewidth=2.5)
    
    # Línea crítica de los 5 Sigmas (Descubrimiento)
    plt.axhline(y=5, color='r', linestyle='--', label="Umbral de Descubrimiento (5σ)")
    
    plt.title("📈 Curva de Sensibilidad QNIM: Confianza vs. SNR", fontsize=14)
    plt.xlabel("Relación Señal-Ruido (SNR)", fontsize=12)
    plt.ylabel("Significancia Estadística (Z-Score Sigma)", fontsize=12)
    plt.legend()
    
    # Guardar para la memoria
    plt.savefig("reports/curva_sensibilidad_tfm.png", dpi=300)
    print("✅ Gráfica generada en 'reports/curva_sensibilidad_tfm.png'")
    plt.show()

if __name__ == "__main__":
    generate_sensitivity_plot()