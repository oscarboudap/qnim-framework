import numpy as np
from neal import SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel

class AnnealingParameterOptimizer:
    def __init__(self, token=None):
        # Mantenemos Neal para máxima fidelidad local sin token
        self.sampler = SimulatedAnnealingSampler()
        print("🤖 Simulación de Recocido Térmico de alta fidelidad (Neal).")

    def build_qubo_for_matching(self, observed_strain, templates):
        bqm = BinaryQuadraticModel(vartype='BINARY')
        for i, temp in enumerate(templates):
            t_strain = temp['strain'][:len(observed_strain)]
            # Función de coste con precisión de punto flotante
            error = np.sum((observed_strain - t_strain)**2)
            bqm.add_variable(i, float(error))
            
        # Restricción 'One-Hot' con multiplicador de Lagrange reforzado
        bqm.add_linear_equality_constraint(
            [(i, 1.0) for i in range(len(templates))],
            constant=-1.0,
            lagrange_multiplier=10.0 
        )
        return bqm

    def run_inference(self, bqm):
        # Configuramos parámetros para que se parezca a una QPU real
        # 2000 sweeps emulan una búsqueda exhaustiva en el espacio de fases
        sampleset = self.sampler.sample(
            bqm, 
            num_reads=1000, 
            num_sweeps=2000, 
            beta_range=[0.1, 15.0] # Rango de temperatura inversa
        )
        best_sample = sampleset.first.sample
        selected = [k for k, v in best_sample.items() if v == 1]
        return selected[0] if selected else 0