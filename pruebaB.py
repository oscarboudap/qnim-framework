import numpy as np

import numpy as np
from src.infrastructure.sstg_adapter import SSTGAdapter
from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer

adapter = SSTGAdapter()
dataset = adapter.generate_balanced_dataset(
    n_per_class=80, n_val_per_class=20, target_snr_range=(8, 30), seed=42,
)

# max_iterations bajo a propósito: el diagnóstico sugiere que el MLP
# ya resuelve casi todo SOLO con el warm-start, sin apenas entrenar el ansatz.
trainer = QiskitVQCTrainer(
    mode="sim", ansatz_reps=2, patience=30, learning_rate=0.03,
    readout_hidden_size=32, readout_refit_every=10,
)
result = trainer.train_and_evaluate(dataset, n_qubits=12, shots=512, max_iterations=60)
print(f"Accuracy combinado: {result.accuracy_sim:.3f}")
print(f"Épocas: {result.n_epochs}")

cm = np.array(result.confusion_matrix)
np.set_printoptions(precision=2, suppress=True)
print(cm)

# Pares de clases con mas confusion mutua (fuera de la diagonal)
errores = [(i, j, cm[i,j]) for i in range(13) for j in range(13) if i != j and cm[i,j] > 0.02]
errores.sort(key=lambda t: -t[2])
print("\nMayores confusiones (clase_real, clase_predicha, fraccion):")
for i, j, v in errores[:10]:
    print(f"  {i} -> {j}: {v:.3f}")