"""
scripts/debug_eml_explosion.py
================================
Verifica en SEGUNDOS (no minutos) dos cosas, usando la función de coste
SINTÉTICA (pura numpy, sin simular circuitos cuánticos):

  1) ¿Está el parche realmente aplicado? Si no ves ninguna línea
     "[diagnóstico] lambda_min=..." en la salida, el código no se
     guardó / no se está ejecutando la versión parcheada.

  2) ¿lambda_min está pegado al suelo de regularización (~1e-8) como
     predice el diagnóstico? Y con el tope (eml_max_scale) puesto,
     ¿deja de explotar el loss?

Uso:
    python scripts/debug_eml_explosion.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

import numpy as np
from src.infrastructure.qnspsa_eml_feynman import (
    QNSPSAConfig, QNSPSAEMLFeynman, make_synthetic_loss_fn,
)

print("=" * 70)
print("Verificando si el campo eml_max_scale existe en QNSPSAConfig...")
cfg_test = QNSPSAConfig()
if hasattr(cfg_test, "eml_max_scale"):
    print(f"OK -- eml_max_scale = {cfg_test.eml_max_scale}")
else:
    print("FALTA -- el parche de QNSPSAConfig NO está aplicado. "
          "Añade el campo `eml_max_scale: float = 10.0` a la clase "
          "y vuelve a correr esto antes de tocar nada más.")
    sys.exit(1)
print("=" * 70)

loss_fn = make_synthetic_loss_fn(n_classes=4, n_params=64, seed=42)
x0 = np.random.default_rng(0).normal(0, 0.01, 64)

print(f"\nLoss inicial: {loss_fn(x0):.4f}  (ln(4)={np.log(4):.4f})\n")

cfg = QNSPSAConfig(maxiter=20, patience=100, lr=0.03, seed=42)
opt = QNSPSAEMLFeynman(config=cfg)
result = opt.minimize(loss_fn, x0.copy())

print(f"\nLoss final (mejor encontrado): {result.final_loss:.4f}")
print(f"Historia de loss (iter a iter): {[round(l, 3) for l in result.loss_history]}")

if "[diagnóstico]" not in "".join([]):
    pass  # el propio logger ya lo habrá impreso arriba si el parche está aplicado

print("\n" + "=" * 70)
print("Si NO viste ninguna línea '[diagnóstico] iter=N lambda_min=...' arriba,")
print("el parche del PASO 5 tampoco está aplicado -- revisa el archivo.")
print("Si SÍ la viste y lambda_min ronda 1e-08 en casi todas las iteraciones,")
print("el diagnóstico original queda confirmado con datos.")
print("Si el loss_history final converge por debajo de ln(4) de forma estable")
print("(sin los picos erráticos de antes), el tope eml_max_scale está")
print("funcionando correctamente.")