"""
scripts/verify_feature_map_fix.py
===================================
Verificacion RAPIDA (segundos) de que el feature map ya esta conectado:
distintas muestras (xi) deben producir circuitos -y distribuciones de
medida- DISTINTOS, incluso con los mismos pesos (theta).

Antes del fix v3, esto fallaba: el circuito ejecutado era identico para
cualquier xi (solo dependia de theta).

Uso:
    python scripts/verify_feature_map_fix.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.infrastructure.qiskit_vqc_trainer import (
    _build_feature_map_and_ansatz,
    _bind_sample,
    _n_bits_for_classes,
)
from qiskit.primitives import StatevectorSampler

print("=" * 70)
print("  Verificacion: el circuito combinado depende de xi (no solo de theta)")
print("=" * 70)

n_qubits = 12
combined, x_params, ansatz_params = _build_feature_map_and_ansatz(n_qubits, reps=2)
print(f"  x_params (features): {len(x_params)}")
print(f"  ansatz_params (pesos entrenables): {len(ansatz_params)}")
print(f"  total parametros del circuito combinado: {combined.num_parameters}")
assert combined.num_parameters == len(x_params) + len(ansatz_params), (
    "El circuito combinado deberia tener x_params + ansatz_params parametros"
)

combined_meas = combined.copy()
combined_meas.measure_all()

rng = np.random.default_rng(0)
theta_fit = rng.normal(0.0, 0.01, len(ansatz_params))  # mismos pesos para ambas muestras

xi_a = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20])
xi_b = np.array([2.90, -2.50, 1.70, -0.30, -1.80, 2.10, 0.05, -2.90, 1.50, -0.90, 2.70, -1.20])

bound_a = _bind_sample(combined_meas.copy(), x_params, ansatz_params, xi_a, theta_fit)
bound_b = _bind_sample(combined_meas.copy(), x_params, ansatz_params, xi_b, theta_fit)

sampler = StatevectorSampler()
job = sampler.run([(bound_a,), (bound_b,)], shots=4096)
result = job.result()

counts_a = result[0].data.meas.get_counts()
counts_b = result[1].data.meas.get_counts()

n_bits = _n_bits_for_classes(13)
top_a = max(counts_a, key=counts_a.get)
top_b = max(counts_b, key=counts_b.get)

print(f"\n  Muestra A (xi={xi_a[:3]}...): bitstring mas frecuente = {top_a} "
      f"({counts_a[top_a]}/4096 shots)")
print(f"  Muestra B (xi={xi_b[:3]}...): bitstring mas frecuente = {top_b} "
      f"({counts_b[top_b]}/4096 shots)")

if counts_a == counts_b:
    print("\n  ❌ FALLO: las distribuciones de medida son IDENTICAS.")
    print("     El circuito sigue sin depender de xi. Revisa _bind_sample / ")
    print("     _build_feature_map_and_ansatz.")
    sys.exit(1)
else:
    print("\n  ✅ OK: las distribuciones de medida son DISTINTAS para xi distinta.")
    print("     El circuito combinado SI depende de las features de entrada.")