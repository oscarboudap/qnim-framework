import os, sys, numpy as np, pandas as pd, csv, dataclasses
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
load_dotenv()

# Capas de Infraestructura y Aplicación
from src.infrastructure.h5_gw_repository import H5GWRepository
from src.infrastructure.qiskit_classifier import QiskitClassifier
from src.application.signal_preprocessing_service import SignalPreprocessingService
from src.application.quantum_mapping_service import QuantumMappingService
from src.application.anomaly_generator_service import AnomalyGeneratorService
from src.application.hubble_solver_service import HubbleSolverService

# Nuevo Módulo de Metrología Gravitacional (Rigor Doctoral)
from src.domain.metrology.fisher_matrix_calculator import FisherMatrixCalculator
from src.domain.metrology.multipole_validator import MultipoleValidator
from src.domain.metrology.planck_error_bounds import PlanckErrorBounds

def run_scientific_sweep():
    print("🔬 [QNIM-LAB] Iniciando Auditoría Geométrica y Cosmológica (NISQ-Era)...")
    
    # 1. Instanciación del Ecosistema
    repo = H5GWRepository(base_path=".")
    classifier = QiskitClassifier(n_qubits=8, iterations=5) # 5 iters para optimizar tiempo/precisión
    preprocessor = SignalPreprocessingService(sample_rate=4096)
    mapper = QuantumMappingService(target_qubits=8)
    anomaly_gen = AnomalyGeneratorService()
    hubble_service = HubbleSolverService()
    multipole_val = MultipoleValidator()
    planck_bounds = PlanckErrorBounds()

    output_file = "resultados_metrologia_planck.csv"
    distances = [100, 400, 800, 1600, 3200]
    theories = ["RG", "LQG", "STRING_FUZZBALL"]

    # Definición de la "Ficha Técnica de Planck" (Headers extendidos)
    headers = [
        "theory", "dist_mpc", "snr_est", "confidence", "std_dev", 
        "h0_inferred", "delta_q_hair", "mass_precision", "r_detectable_limit", "object_class"
    ]
    
    # Inicialización del CSV (Escritura de cabeceras)
    with open(output_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

    print(f"{'Teoría':<15} | {'Dist':<6} | {'SNR':<5} | {'Conf.':<8} | {'H0':<6} | {'ΔQ Hair':<8}")
    print("-" * 75)

    for dist in distances:
        snr_scaler = 100 / dist
        for theory in theories:
            # --- A. PROCESAMIENTO GEOMÉTRICO ---
            raw = repo.get_signal_by_detector("H1")
            scaled_signal = dataclasses.replace(raw, strain=raw.strain * snr_scaler)
            clean = preprocessor.bandpass_filter(preprocessor.whitening(scaled_signal))
            
            # --- B. INYECCIÓN DE METRICA SUCIA / CUÁNTICA ---
            current_signal = clean
            if theory != "RG":
                current_signal = anomaly_gen.apply_theory_drift(clean, theory=theory, epsilon=0.25)
            
            # --- C. INFERENCIA EN ESPACIO DE HILBERT ---
            data = mapper.prepare_multitask_embedding(current_signal)
            inference = classifier.predict(data)
            
            # --- D. METROLOGÍA Y VALIDACIÓN (FISICA DE CAMPO FUERTE) ---
            prob = inference.probabilities[0]
            snr_val = snr_scaler * 20
            
            # 1. Inferencia de Hubble (Métrica FLRW)
            # Redshift z aproximado para dL (Sirenas Estándar)
            z_approx = (dist * 70) / 299792.458 
            h0_val = hubble_service.infer_h0(dist, z_approx + np.random.normal(0, 0.002))
            
            # 2. Validación de No-Cabello (Multipolos)
            hair_analysis = multipole_val.check_no_hair_theorem(
                mass=30, spin=0.7, observed_m2=anomaly_gen.get_quadrupole_deviation(prob)
            )
            
            # 3. Matriz de Fisher (Cramér-Rao Bounds)
            fisher = FisherMatrixCalculator(snr=snr_val)
            precision = fisher.calculate_precision_bounds({"mass": 30})
            
            # 4. Límites de Planck
            p_limit = planck_bounds.get_exclusion_limit(snr_val)

            # --- E. ESCRITURA EN TIEMPO REAL ---
            row = {
                "theory": theory,
                "dist_mpc": dist,
                "snr_est": round(snr_val, 2),
                "confidence": round(prob, 4),
                "std_dev": round(inference.metadata.get("std_dev", 0), 5),
                "h0_inferred": round(h0_val, 2),
                "delta_q_hair": round(hair_analysis["delta_q"], 4),
                "mass_precision": round(precision["sigma_mass"], 4),
                "r_detectable_limit": round(p_limit["min_detectable_r"], 6),
                "object_class": hair_analysis["object_type"]
            }

            with open(output_file, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerow(row)

            print(f"{theory:<15} | {dist:<6} | {row['snr_est']:<5} | {prob:<8.4f} | {row['h0_inferred']:<6} | {row['delta_q_hair']:<8.4f}")

    print(f"\n✅ Auditoría Completa. Archivo generado: {output_file}")

if __name__ == "__main__":
    run_scientific_sweep()