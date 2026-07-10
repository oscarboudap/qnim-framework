"""
validate_physics_pipeline.py
=============================
Prueba rápida del pipeline completo con PhysicsSSTGAdapter + clasificador
clásico (sin el VQC, para no gastar tiempo de simulación cuántica).

Objetivo: confirmar que:
  1. PyCBC genera las ondas base sin errores para las 13 clases.
  2. Los inyectores beyond-GR producen señales distinguibles (separabilidad > azar).
  3. El extractor de fase funciona de extremo a extremo.

Ejecutar desde la raíz del proyecto:
    python validate_physics_pipeline.py

Tiempo esperado: ~3-10 min (5 eventos/clase × 13 clases con PyCBC IMRPhenomD).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))   # asegura importar desde aquí

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from src.infrastructure.physics_sstg_adapter import PhysicsSSTGAdapter

def main():
    print("=" * 60)
    print("  Validación del pipeline con física real (PyCBC)")
    print("=" * 60)

    # Dataset pequeño: 5 train + 3 val por clase
    adapter = PhysicsSSTGAdapter(
        n_features=12,
        approximant="IMRPhenomD",   # rápido
        waveform_duration=4.0,
    )
    dataset = adapter.generate_balanced_dataset(
        n_per_class=5,
        n_val_per_class=3,
        target_snr_range=(8, 30),
        seed=42,
        verbose=True,
    )

    n_cls = dataset.n_classes
    baseline = 1.0 / n_cls

    # Clasificador clásico para medir separabilidad sin el VQC
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=0)
    clf.fit(dataset.X_train, dataset.y_train)
    acc_train = clf.score(dataset.X_train, dataset.y_train)
    acc_val   = clf.score(dataset.X_val,   dataset.y_val)

    print(f"\n  Accuracy clásica (train): {acc_train:.3f}")
    print(f"  Accuracy clásica (val):   {acc_val:.3f}   (azar = {baseline:.3f})")

    if acc_val > baseline + 0.05:
        print(f"\n  ✅ El dataset con física real ES separable clásicamente.")
        print(f"     Listo para conectar al VQC (--physics-generator).")
    else:
        print(f"\n  ⚠️  Separabilidad baja ({acc_val:.3f} vs azar {baseline:.3f}).")
        print(f"     Revisa los inyectores o amplía n_per_class.")

    # Medias por clase (primeras 4 features) para diagnóstico visual
    print("\n  Media de las primeras 4 features por clase:")
    print(f"  {'Clase':<30} {'f0_Re':>8} {'f0_Im':>8} {'f1_Re':>8} {'f1_Im':>8}")
    print("  " + "-" * 66)
    for c, name in enumerate(dataset.theory_names):
        mask = dataset.y_train == c
        if mask.any():
            means = dataset.X_train[mask, :4].mean(axis=0)
            print(f"  {name:<30} {means[0]:8.4f} {means[1]:8.4f} {means[2]:8.4f} {means[3]:8.4f}")

    print("\n" + "=" * 60)
    print("  Validación completada.  Si el accuracy val >> azar,")
    print("  lanza el pipeline completo con:")
    print()
    print("  python scripts/generate_results.py --mode sim \\")
    print("    --physics-generator \\")
    print("    --n-per-class 20 --n-val-per-class 5 \\")
    print("    --max-iter 60 --readout-hidden-size 32")
    print("=" * 60)


if __name__ == "__main__":
    main()