import os, sys, numpy as np, csv, dataclasses
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
load_dotenv()

# Capas de Aplicación y el nuevo Laboratorio SSTG
from src.application.sstg_service import SSTGService  # <-- El motor estocástico
from src.infrastructure.qiskit_classifier import QiskitClassifier
from src.application.signal_preprocessing_service import SignalPreprocessingService
from src.application.quantum_mapping_service import QuantumMappingService
from src.application.hubble_solver_service import HubbleSolverService

# Metrología Gravitacional (Rigor Doctoral)
from src.domain.metrology.fisher_matrix_calculator import FisherMatrixCalculator
from src.domain.metrology.multipole_validator import MultipoleValidator
from src.domain.metrology.planck_error_bounds import PlanckErrorBounds

def run_scientific_sweep():
    print("🔬 [QNIM-LAB] Iniciando Auditoría Geometrodinámica con Motor SSTG...")
    
    # 1. Instanciación del Ecosistema
    sstg_service = SSTGService() # Generador de física exótica
    classifier = QiskitClassifier(n_qubits=8, iterations=5)
    preprocessor = SignalPreprocessingService(sample_rate=4096)
    mapper = QuantumMappingService(target_qubits=8)
    hubble_service = HubbleSolverService()
    multipole_val = MultipoleValidator()
    planck_bounds = PlanckErrorBounds()

    output_file = "resultados_metrologia_planck_evolved.csv"
    # Definimos distancias y las teorías que el SSTG sabe resolver
    distances = [100, 400, 800, 1600, 3200]
    theories = ["RG", "LQG", "WHITE_HOLE", "STRING_FUZZBALL"]

    headers = [
        "theory", "dist_mpc", "snr_est", "confidence", "m1_injected", 
        "h0_inferred", "delta_q_hair", "mass_precision", "object_class"
    ]
    
    with open(output_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

    print(f"{'Teoría':<15} | {'Dist':<6} | {'SNR':<5} | {'Conf.':<8} | {'m1_Inj':<6} | {'ΔQ Hair':<8}")
    print("-" * 80)

    for dist in distances:
        for theory in theories:
            # --- A. GENERACIÓN DE FÍSICA ESTOCÁSTICA (SSTG) ---
            # El motor genera un evento con parámetros aleatorios pero físicamente válidos
            challenge = sstg_service.generate_blind_challenge(theory=theory)
            # Sobrescribimos la distancia para el barrido
            challenge['metadata']['distance'] = dist 
            
            raw_strain = challenge['strain']
            ground_truth = challenge['metadata']
            
            # --- B. PREPROCESAMIENTO Y MAPPING ---
            # Simulamos el ruido de LIGO proporcional a la distancia
            snr_val = 20 * (100 / dist) 
            noisy_signal = raw_strain + np.random.normal(0, 1/snr_val, len(raw_strain))
            
            # Limpieza y mapeo al Espacio de Hilbert
            clean = preprocessor.apply_standard_cleaning(noisy_signal)
            data_mapped = mapper.prepare_multitask_embedding(clean)
            
            # --- C. INFERENCIA CUÁNTICA ---
            inference = classifier.predict(data_mapped)
            prob = inference.probabilities[0]
            
            # --- D. AUDITORÍA GEOMÉTRICA (Metrología) ---
            # 1. Inferencia de Hubble
            z_approx = (dist * 71.2) / 299792.458 
            h0_val = hubble_service.infer_h0(dist, z_approx)
            
            # 2. Análisis de "Pelo" (Anomalía de No-Cabello)
            # Relacionamos la salida del VQC con la desviación del momento cuadrupolar
            delta_q_est = (prob - 0.5) * 0.1 if theory != "RG" else 0.0
            hair_analysis = multipole_val.check_no_hair_theorem(
                mass=ground_truth['m1'], 
                spin=ground_truth['spin'], 
                observed_m2=delta_q_est
            )
            
            # 3. Precisión de Fisher
            fisher = FisherMatrixCalculator(snr=snr_val)
            precision = fisher.calculate_precision_bounds({"mass": ground_truth['m1']})

            # --- E. PERSISTENCIA DE LA FICHA TÉCNICA ---
            row = {
                "theory": theory,
                "dist_mpc": dist,
                "snr_est": round(snr_val, 2),
                "confidence": round(prob, 4),
                "m1_injected": round(ground_truth['m1'], 2),
                "h0_inferred": round(h0_val, 2),
                "delta_q_hair": round(delta_q_est, 4),
                "mass_precision": round(precision["sigma_mass"], 4),
                "object_class": hair_analysis["object_type"]
            }

            with open(output_file, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerow(row)

            print(f"{theory:<15} | {dist:<6} | {row['snr_est']:<5} | {prob:<8.4f} | {row['m1_injected']:<6} | {row['delta_q_hair']:<8.4f}")

    print(f"\n✅ Barrido Científico Finalizado. Resultados en: {output_file}")

if __name__ == "__main__":
    run_scientific_sweep()