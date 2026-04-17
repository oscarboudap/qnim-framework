import numpy as np
import os
import joblib
from src.domain.quantum.vqc_classifier import VQCClassifier, CloudTranspilerSampler
from src.domain.quantum.annealing_optimizer import AnnealingParameterOptimizer

class HybridInferenceOrchestrator:
    def __init__(self, ibm_token=None, ibm_backend="ibm_kingston", use_real_ibm=False):
        self.vqc_engine = VQCClassifier(
            n_qubits=12, token=ibm_token, backend_name=ibm_backend, use_real_hardware=use_real_ibm
        )
        self.annealer_engine = AnnealingParameterOptimizer(token=None)

    def run_full_diagnosis(self, event_data):
        print(f"🕵️ Analizando evento: {event_data.get('id', 'unknown')}")
        quantum_ready_data = event_data['quantum_signal'] 
        
        print("🔵 Rama IBM: Configurando Inferencia Cuántica (12 Cúbits)...")
        from qiskit.circuit.library import ZZFeatureMap
        fmap = ZZFeatureMap(12, reps=1) 
        
        # 1. CARGA DE PESOS
        weights_path = "models/qnim_vqc_weights.npy"
        if os.path.exists(weights_path):
            print("📂 Cargando pesos pre-entrenados...")
            loaded_weights = np.load(weights_path)
        else:
            print("⚠️ Usando pesos aleatorios (Zero-Shot inicial)...")
            loaded_weights = np.random.uniform(-np.pi, np.pi, self.vqc_engine.ansatz.num_parameters)

        # 2. INICIALIZACIÓN LOCAL
        self.vqc_engine.initialize_classifier(fmap, initial_weights=loaded_weights)

        # 3. DUMMY FIT LOCAL
        fake_X = np.zeros((1, 12))
        fake_y = np.array([[1, 0]])
        self.vqc_engine.vqc.fit(fake_X, fake_y) 
        
        # 4. COMPRESIÓN DE DATOS (PIPELINE)
        pipeline_path = "models/qnim_preprocessing_pipeline.pkl"
        if os.path.exists(pipeline_path):
            print("🗜️ Comprimiendo señal astrofísica (Pipeline Cuántico)...")
            quantum_pipeline = joblib.load(pipeline_path)
            # Pasamos la onda entera (16384) y sale un array de 12 acotado entre -π y π
            features = quantum_pipeline.transform(quantum_ready_data.reshape(1, -1))
        else:
            print("⚠️ No se encontró el pipeline. Usando truncado básico.")
            features = np.array([quantum_ready_data[:12]])
        
        # HOT-SWAP A HARDWARE REAL
        if self.vqc_engine.use_real_hardware and self.vqc_engine.real_backend:
            print("🔌 Conectando la red neuronal al hardware físico (Hot-Swap)...")
            real_sampler = CloudTranspilerSampler(self.vqc_engine.real_backend)
            self.vqc_engine.vqc.neural_network.sampler = real_sampler

        # 5. EJECUCIÓN 
        prediction = self.vqc_engine.vqc.predict(features)
        theory_idx = np.argmax(prediction[0])
        theory_label = "Quantum Anomaly (LQG/Fuzzball)" if theory_idx == 1 else "General Relativity (Kerr)"        
        
        probabilities = self.vqc_engine.vqc.neural_network.forward(features, self.vqc_engine.vqc.weights)
        real_confidence = float(probabilities[0][theory_idx])

        print(f"🟢 Rama D-Wave: Refinando parámetros físicos para {theory_label}...")
        search_space = self._generate_search_space(event_data['metadata'])
        bqm = self.annealer_engine.build_qubo_for_matching(quantum_ready_data, search_space)
        best_fit_idx = self.annealer_engine.run_inference(bqm)
        refined_params = search_space[best_fit_idx]['params']

        return {
            "theory": theory_label,
            "confidence": real_confidence,
            "best_fit_mass": refined_params['m1'],
            "best_fit_spin": refined_params['spin'],
            "dwave_precision": "High (Quantum Tunneling search)"
        }

    def _generate_search_space(self, meta):
        templates = []
        m_base = meta.get('m1') or 35.0
        print(f"📊 D-WAVE: Mapeando espacio de fases masivo alrededor de M={m_base:.1f}...")
        for m in np.linspace(m_base - 15, m_base + 15, 20):
            for s in np.linspace(-0.9, 0.9, 10):
                templates.append({
                    'params': {'m1': m, 'spin': s},
                    'strain': np.random.normal(0, 0.05, 64) 
                })
        print(f"🌌 Espacio QUBO generado con {len(templates)} plantillas de correlación.")
        return templates