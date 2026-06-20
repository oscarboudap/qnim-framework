"""
scripts/diagnose_loss_noise.py
================================
Diagnostico AISLADO (segundos, no minutos) de si el "+epsilon-clip" en la
cross-entropy es lo que mantiene el loss pegado a ln(n_classes) (azar puro).

NO genera dataset, NO entrena, NO toca IBM. Construye UN circuito con pesos
fijos, lo mide a distintos niveles de shots, y compara dos formas de pasar
de "counts" a "probabilidades":

  - probs_old: la version original con np.clip(probs, 1e-10, 1.0) -> si una
    clase recibe 0 shots, su log-perdida es -log(1e-10) = 23.03 (outlier).
  - probs_new: suavizado de Laplace (alpha=1) -> el peor caso es mucho mas
    acotado: -log(1/(total+n_classes)).

Si ves que loss_old tiene picos enormes a pocos shots y loss_new se
mantiene cerca de ln(n_classes), el bug de clipping es (al menos parte
de) la explicacion del estancamiento. Si AMBOS se mantienen igual de
lejos de ln(n_classes) incluso con muchos shots (16384+), el problema no
es de ruido de shots sino de capacidad del ansatz / feature map.

Uso:
    python scripts/diagnose_loss_noise.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import StatevectorSampler

N_QUBITS = 12
N_CLASSES = 13
N_BITS = max(1, int(np.ceil(np.log2(N_CLASSES))))  # = 4

print("=" * 70)
print("  Diagnostico aislado: ruido de shots vs estancamiento del loss")
print("=" * 70)
print(f"  n_qubits={N_QUBITS}  n_classes={N_CLASSES}  n_bits={N_BITS}")
print(f"  ln(n_classes) = {np.log(N_CLASSES):.4f}  (baseline de azar puro)")
print()

ansatz = EfficientSU2(N_QUBITS, reps=2, entanglement="linear")
ansatz.measure_all()

rng = np.random.default_rng(0)
theta = rng.normal(0.0, 0.01, ansatz.num_parameters)
bound = ansatz.assign_parameters(theta)

sampler = StatevectorSampler()


def probs_old(counts: dict, total: int) -> np.ndarray:
    """Version ORIGINAL (con el bug de epsilon-clip)."""
    p = np.zeros(N_CLASSES)
    for bitstring, c in counts.items():
        idx = int(bitstring[-N_BITS:], 2) % N_CLASSES
        p[idx] += c / total
    p = np.clip(p, 1e-10, 1.0)
    p /= p.sum()
    return p


def probs_new(counts: dict, total: int) -> np.ndarray:
    """Version CORREGIDA (suavizado de Laplace, alpha=1)."""
    raw = np.zeros(N_CLASSES)
    for bitstring, c in counts.items():
        idx = int(bitstring[-N_BITS:], 2) % N_CLASSES
        raw[idx] += c
    return (raw + 1.0) / (total + N_CLASSES)


y_true_idx = 7  # clase "verdadera" de juguete, fija para comparar limpio

print(f"{'shots':>8} | {'loss_old (clip)':>16} | {'loss_new (laplace)':>19} | {'p_true_old':>10} | {'p_true_new':>10}")
print("-" * 76)

for shots in [64, 128, 512, 2048, 8192, 32768]:
    job = sampler.run([(bound,)], shots=shots)
    counts = job.result()[0].data.meas.get_counts()

    p_old = probs_old(counts, shots)
    p_new = probs_new(counts, shots)

    loss_old = -np.log(p_old[y_true_idx])
    loss_new = -np.log(p_new[y_true_idx])

    print(
        f"{shots:8d} | {loss_old:16.4f} | {loss_new:19.4f} | "
        f"{p_old[y_true_idx]:10.6f} | {p_new[y_true_idx]:10.6f}"
    )

print()
print(f"  Referencia (azar puro): ln({N_CLASSES}) = {np.log(N_CLASSES):.4f}")
print()
print("  INTERPRETACION:")
print("  - Si loss_old tiene picos >> ln(n_classes) a pocos shots y baja al")
print("    converger con loss_new -> el clipping SI es (parte de) el problema.")
print("    Aplica el parche de Laplace y vuelve a entrenar.")
print("  - Si AMBOS se mantienen lejos de ln(n_classes) incluso con 32768")
print("    shots -> el problema no es ruido de shots. Sospecha del ansatz")
print("    (EfficientSU2 reps=2 puede no tener suficiente capacidad para")
print("    separar 13 clases con este feature map) o de la codificacion")
print("    Chebyshev de las features.")