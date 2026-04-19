#!/usr/bin/env python3
"""
CASO DE USO: Validación Estadística de QNIM

Este script es un CASO DE USO de Application Layer,
no un script suelto. Orquesta a través de StatisticalValidator.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Domain + Infrastructure + Application (DDD)
from src.application.statistical_validator import StatisticalValidator


def dummy_signal_generator(theory: str, snr: float, n_samples: int = 16384) -> np.ndarray:
    """Generador dummy (en producción: SSTG complete)"""
    import numpy as np
    t = np.linspace(0, 1, n_samples)
    signal = np.sin(2 * np.pi * 50 * t) * np.exp(-5 * t)
    
    if theory == "LQG":
        signal += 0.05 * np.sin(4 * np.pi * 50 * t) * np.exp(-5 * t)
    elif theory == "ECO":
        t_merger = 0.7
        mask = t > t_merger
        signal[mask] += 0.1 * np.sin(2 * np.pi * 100 * (t[mask] - t_merger)) * np.exp(-10 * (t[mask] - t_merger))
    elif theory == "Brans-Dicke":
        signal *= (1.05 + 0.02 * t)
    
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (snr ** 2)
    signal = signal * np.sqrt(signal_power / (np.mean(signal ** 2) + 1e-10))
    
    return signal


def dummy_vqc_classifier(signal: np.ndarray) -> np.ndarray:
    """VQC dummy (en producción: VQC entrenado)"""
    import numpy as np
    rms = np.sqrt(np.mean(signal ** 2))
    energy_ratio = np.sum(signal[int(0.7*len(signal)):] ** 2) / (np.sum(signal[:int(0.3*len(signal))] ** 2) + 1e-10)
    
    probs = np.array([0.45, 0.20, 0.15, 0.15, 0.05])
    
    if energy_ratio > 1.0:
        probs[2] += 0.15
    if rms > 0.5:
        probs[0] -= 0.10
    
    probs = np.clip(probs, 0, 1)
    probs /= np.sum(probs)
    
    return probs


def dummy_parameter_extractor(signal: np.ndarray) -> dict:
    """Extractor dummy"""
    import numpy as np
    m_chirp = 30.0 + np.random.normal(0, 2)
    chi_eff = 0.5 + np.random.normal(0, 0.1)
    snr_estimado = np.sqrt(np.sum(signal ** 2))
    
    return {
        "M_chirp": m_chirp,
        "chi_eff": chi_eff,
        "snr_estimated": snr_estimado
    }


def main():
    import numpy as np
    
    print("="*70)
    print("🧪 QNIM VALIDACIÓN ESTADÍSTICA - CASO DE USO")
    print("="*70)
    print()
    
    load_dotenv()
    os.makedirs("reports/validation", exist_ok=True)
    
    SNR_LEVELS = [5, 10, 15, 20, 25]
    N_TRIALS_PER_SNR = 50
    THEORY_LABELS = ["GR", "LQG", "ECO", "Brans-Dicke", "Other"]
    
    print("Paso 1: Inicializando StatisticalValidator...")
    validator = StatisticalValidator(n_bootstrap=1000, random_seed=42)
    print("  ✓ Validador listo")
    print()
    
    print("Paso 2: Ejecutando pipeline de validación completo...")
    results = validator.full_validation_pipeline(
        signal_generator=dummy_signal_generator,
        vqc_classifier=dummy_vqc_classifier,
        parameter_extractor=dummy_parameter_extractor,
        theory_labels=THEORY_LABELS,
        snr_levels=SNR_LEVELS,
        n_trials_per_snr=N_TRIALS_PER_SNR,
        output_dir="reports/validation"
    )
    print()
    
    print("Paso 3: Generando reporte final...")
    validator.create_validation_report(results, "reports/VALIDATION_REPORT.md")
    print()
    
    print("="*70)
    print("✅ VALIDACIÓN ESTADÍSTICA COMPLETADA")
    print("="*70)
    print()
    print("Archivos generados:")
    print("  📄 reports/validation/validation_results.json")
    print("  📄 reports/VALIDATION_REPORT.md")


if __name__ == "__main__":
    sys.exit(main())



def dummy_signal_generator(theory: str, snr: float, n_samples: int = 16384) -> np.ndarray:
    """Generador dummy (en producción: SSTG complete)"""
    import numpy as np
    t = np.linspace(0, 1, n_samples)
    signal = np.sin(2 * np.pi * 50 * t) * np.exp(-5 * t)
    
    if theory == "LQG":
        signal += 0.05 * np.sin(4 * np.pi * 50 * t) * np.exp(-5 * t)
    elif theory == "ECO":
        t_merger = 0.7
        mask = t > t_merger
        signal[mask] += 0.1 * np.sin(2 * np.pi * 100 * (t[mask] - t_merger)) * np.exp(-10 * (t[mask] - t_merger))
    elif theory == "Brans-Dicke":
        signal *= (1.05 + 0.02 * t)
    
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (snr ** 2)
    signal = signal * np.sqrt(signal_power / (np.mean(signal ** 2) + 1e-10))
    
    return signal


def dummy_vqc_classifier(signal: np.ndarray) -> np.ndarray:
    """VQC dummy (en producción: VQC entrenado)"""
    import numpy as np
    rms = np.sqrt(np.mean(signal ** 2))
    energy_ratio = np.sum(signal[int(0.7*len(signal)):] ** 2) / (np.sum(signal[:int(0.3*len(signal))] ** 2) + 1e-10)
    
    probs = np.array([0.45, 0.20, 0.15, 0.15, 0.05])
    
    if energy_ratio > 1.0:
        probs[2] += 0.15
    if rms > 0.5:
        probs[0] -= 0.10
    
    probs = np.clip(probs, 0, 1)
    probs /= np.sum(probs)
    
    return probs


def dummy_parameter_extractor(signal: np.ndarray) -> dict:
    """Extractor dummy"""
    import numpy as np
    m_chirp = 30.0 + np.random.normal(0, 2)
    chi_eff = 0.5 + np.random.normal(0, 0.1)
    snr_estimado = np.sqrt(np.sum(signal ** 2))
    
    return {
        "M_chirp": m_chirp,
        "chi_eff": chi_eff,
        "snr_estimated": snr_estimado
    }


def main():
    import numpy as np
    
    print("="*70)
    print("🧪 QNIM VALIDACIÓN ESTADÍSTICA - CASO DE USO")
    print("="*70)
    print()
    
    load_dotenv()
    os.makedirs("reports/validation", exist_ok=True)
    
    SNR_LEVELS = [5, 10, 15, 20, 25]
    N_TRIALS_PER_SNR = 50
    THEORY_LABELS = ["GR", "LQG", "ECO", "Brans-Dicke", "Other"]
    
    print("Paso 1: Inicializando StatisticalValidator...")
    validator = StatisticalValidator(n_bootstrap=1000, random_seed=42)
    print("  ✓ Validador listo")
    print()
    
    print("Paso 2: Ejecutando pipeline de validación completo...")
    results = validator.full_validation_pipeline(
        signal_generator=dummy_signal_generator,
        vqc_classifier=dummy_vqc_classifier,
        parameter_extractor=dummy_parameter_extractor,
        theory_labels=THEORY_LABELS,
        snr_levels=SNR_LEVELS,
        n_trials_per_snr=N_TRIALS_PER_SNR,
        output_dir="reports/validation"
    )
    print()
    
    print("Paso 3: Generando reporte final...")
    validator.create_validation_report(results, "reports/VALIDATION_REPORT.md")
    print()
    
    print("="*70)
    print("✅ VALIDACIÓN ESTADÍSTICA COMPLETADA")
    print("="*70)
    print()
    print("Archivos generados:")
    print("  📄 reports/validation/validation_results.json")
    print("  📄 reports/VALIDATION_REPORT.md")


if __name__ == "__main__":
    sys.exit(main())
