"""
scripts/verify_chebyshev_consistency.py
==========================================
Diagnostico instantaneo (puro numpy, sin Qiskit) de si chebyshev_preprocess
normaliza train y val de forma CONSISTENTE.

Uso:
    python scripts/verify_chebyshev_consistency.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.infrastructure.qiskit_vqc_trainer import chebyshev_preprocess, compute_chebyshev_stats

rng = np.random.default_rng(0)

# Simula: "train" y "val" con rangos distintos para la MISMA feature fisica
# (esto pasa siempre que se generan por separado, como en este pipeline).
X_train = rng.uniform(0, 10, (100, 1))
X_val = rng.uniform(2, 6, (20, 1))

valor_fisico_compartido = 4.0
X_train_test = np.vstack([X_train, [[valor_fisico_compartido]]])
X_val_test = np.vstack([X_val, [[valor_fisico_compartido]]])

print("=" * 70)
print("  SIN compartir estadisticas (comportamiento ANTES del fix)")
print("=" * 70)
angle_train_old = chebyshev_preprocess(X_train_test)[-1, 0]
angle_val_old = chebyshev_preprocess(X_val_test)[-1, 0]
print(f"  Mismo valor fisico ({valor_fisico_compartido}) ->")
print(f"    angulo usando stats de TRAIN: {angle_train_old:.4f} rad")
print(f"    angulo usando stats de VAL:   {angle_val_old:.4f} rad")
print(f"    diferencia: {abs(angle_train_old - angle_val_old):.4f} rad")
print(f"    (si fuera 0, train y val codificarian igual; NO lo es)")

print()
print("=" * 70)
print("  Compartiendo estadisticas de TRAIN (comportamiento CON el fix)")
print("=" * 70)
train_stats = compute_chebyshev_stats(X_train_test)
angle_train_new = chebyshev_preprocess(X_train_test, stats=train_stats)[-1, 0]
angle_val_new = chebyshev_preprocess(X_val_test, stats=train_stats)[-1, 0]
print(f"  Mismo valor fisico ({valor_fisico_compartido}) ->")
print(f"    angulo (stats de train, aplicado a train): {angle_train_new:.4f} rad")
print(f"    angulo (stats de train, aplicado a val):    {angle_val_new:.4f} rad")
print(f"    diferencia: {abs(angle_train_new - angle_val_new):.8f} rad")

assert abs(angle_train_new - angle_val_new) < 1e-9, "El fix no esta aplicado correctamente"
print()
print("  ✅ OK: con el fix, el mismo valor fisico SIEMPRE produce el mismo")
print("  angulo, sin importar si viene de train, val, o un subconjunto de")
print("  validacion enviado a hardware.")