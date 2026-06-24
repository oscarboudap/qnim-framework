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