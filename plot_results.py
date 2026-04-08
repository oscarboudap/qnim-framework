import pandas as pd
import matplotlib.pyplot as plt

def plot_qnim_sensitivity():
    # Usar backend 'Agg' para generar archivos sin necesidad de entorno gráfico
    plt.switch_backend('Agg')
    
    try:
        df = pd.read_csv('resultados_robustez_qnim.csv')
    except FileNotFoundError:
        print("❌ Error: No se encuentra el archivo 'resultados_robustez_qnim.csv'. Ejecuta primero main_sweep.py")
        return

    plt.figure(figsize=(10, 6))
    
    # Curva con barras de error corregida
    plt.errorbar(
        df['epsilon'], 
        df['prob_lqg'], 
        yerr=df['std_dev'], 
        fmt='-o', 
        color='#6200ee', 
        ecolor='gray', 
        elinewidth=1.5, # Grosor de la línea de error
        capsize=3,       # Remates de las barras
        label='Confianza LQG (Inferencia)'
    )
    
    # Línea de referencia del azar (0.5)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Umbral de Ruido (Azar)')
    
    # Configuración de estética y etiquetas
    plt.title('Curva de Sensibilidad QNIM: Detección de Gravedad Cuántica', fontsize=14)
    plt.xlabel('Magnitud de la Anomalía (Epsilon)', fontsize=12)
    plt.ylabel('Probabilidad de Hipótesis (Inferencia)', fontsize=12)
    plt.ylim(0.45, 0.55) # Enfocamos el rango crítico del ruido
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Guardar y cerrar
    output_name = 'curva_sensibilidad_qnim.png'
    plt.savefig(output_name, dpi=300)
    print(f"📈 Gráfica generada con éxito: {output_name}")

if __name__ == "__main__":
    plot_qnim_sensitivity()