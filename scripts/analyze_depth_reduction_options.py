"""
scripts/analyze_depth_reduction_options.py
============================================
Responde a la pregunta real de Rodrigo ("¿se puede reducir la
profundidad?") probando las DOS palancas que sí tienen fundamento
técnico, en vez de la hipótesis "más qubits" -- que los datos ya
midieron como FALSA con la arquitectura actual (ver
qubits_vs_depth_*.csv: profundidad y fidelidad empeoran con más
qubits, no mejoran).

PALANCA A -- entrelazado del ansatz: circular vs linear
  El feature map ya usa entanglement='linear' (cadena). El ansatz
  (EfficientSU2) usa 'circular' -- que añade UN enlace extra
  qubit_(n-1)--qubit_0 para cerrar el anillo. heavy-hex NO es un
  anillo (es un grafo disperso tipo panal), así que ese único enlace
  extra obliga al compilador a rodear con SWAPs para conectar los dos
  extremos -- probablemente la causa de los overheads de 55-84% que
  ya mediste. Si el ansatz también usa 'linear', ambas capas piden
  solo conectividad en cadena, que EXISTE SIEMPRE en cualquier grafo
  conexo (heavy-hex lo es), para cualquier n. Esto se prueba aquí
  transpilando ambas variantes y comparando.

PALANCA B -- reps del feature map: 2 (actual, hardcoded) vs 1
  El parámetro `reps` que recibe _build_feature_map_and_ansatz() NO
  se usa -- dentro de la función el feature map está fijado a reps=2
  y el ansatz a reps=1 sin mirar el argumento. Aquí se construye una
  versión standalone (sin tocar el módulo de producción) con reps=1
  en el feature map para medir cuánto se recorta la profundidad, y
  con qué coste en fidelidad estimada.

Ninguna de las dos palancas depende de aumentar el número de qubits --
por eso conviene medirlas ANTES de pedirle a Rodrigo más qubits/Credits
basándose en una hipótesis que los propios datos contradicen.

Uso:
    export IBM_QUANTUM_TOKEN='...'
    python scripts/analyze_depth_reduction_options.py --backend ibm_fez \
        --qubit-counts 12 27 33 40 50
"""
import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", type=str, default="ibm_fez")
    p.add_argument("--qubit-counts", type=int, nargs="+", default=[12, 27, 33, 40, 50])
    p.add_argument("--n-trials", type=int, default=5)
    return p.parse_args()


def _build_variant(n_qubits: int, feature_reps: int, ansatz_entanglement: str):
    """
    Construye una variante standalone (no toca qiskit_vqc_trainer.py)
    con el feature_map reps y el ansatz entanglement que se quieran
    probar, para poder compararlos entre sí sin arriesgar el código de
    producción.
    """
    import warnings
    from qiskit.circuit.library import EfficientSU2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            from qiskit.circuit.library import zz_feature_map
            feature_map = zz_feature_map(
                feature_dimension=n_qubits, reps=feature_reps, entanglement="linear",
            )
        except ImportError:
            from qiskit.circuit.library import ZZFeatureMap
            feature_map = ZZFeatureMap(
                feature_dimension=n_qubits, reps=feature_reps, entanglement="linear",
            )
        ansatz = EfficientSU2(
            num_qubits=n_qubits, reps=1, entanglement=ansatz_entanglement,
        )

    combined = feature_map.compose(ansatz)
    combined.measure_all()
    return combined


def transpile_variant(n_qubits, feature_reps, ansatz_entanglement, backend, n_trials):
    from qiskit import transpile

    circuit = _build_variant(n_qubits, feature_reps, ansatz_entanglement)

    depths, n_2q_list = [], []
    for seed in range(n_trials):
        isa = transpile(circuit, backend=backend, optimization_level=3, seed_transpiler=seed)
        depths.append(isa.depth())
        n2q = sum(1 for instr in isa.data if len(instr.qubits) >= 2)
        n_2q_list.append(n2q)

    return statistics.median(depths), statistics.median(n_2q_list), isa


def estimate_fidelity_quick(isa_circuit, backend):
    """Versión resumida de la fidelidad por error de puerta (ver analyze_qubits_vs_depth.py v2)."""
    target = backend.target
    fid = 1.0
    got_any = False
    for instr in isa_circuit.data:
        name = instr.operation.name
        if name in ("barrier", "measure", "delay"):
            continue
        qargs = tuple(isa_circuit.find_bit(q).index for q in instr.qubits)
        try:
            props = target[name][qargs]
            if props is not None and props.error is not None:
                fid *= (1.0 - props.error)
                got_any = True
        except (KeyError, TypeError):
            continue
    return fid if got_any else None


def main():
    args = parse_args()
    import os
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    if not token:
        print("ERROR: exporta IBM_QUANTUM_TOKEN antes de correr esto.")
        sys.exit(1)

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = service.backend(args.backend)
    print(f"Backend: {backend.name}\n")

    variantes = [
        ("ACTUAL (feature_reps=2, ansatz=circular)", 2, "circular"),
        ("Palanca A sola (feature_reps=2, ansatz=linear)", 2, "linear"),
        ("Palanca B sola (feature_reps=1, ansatz=circular)", 1, "circular"),
        ("Palancas A+B (feature_reps=1, ansatz=linear)", 1, "linear"),
    ]

    for label, feature_reps, ansatz_ent in variantes:
        print(f"=== {label} ===")
        header = f"{'qubits':>7} {'profundidad':>12} {'puertas_2q':>11} {'fidelidad_err':>14}"
        print(header)
        for n in args.qubit_counts:
            depth, n2q, isa = transpile_variant(n, feature_reps, ansatz_ent, backend, args.n_trials)
            fid = estimate_fidelity_quick(isa, backend)
            fid_str = f"{fid:.4f}" if fid is not None else "N/D"
            print(f"{n:>7} {depth:>12} {n2q:>11} {fid_str:>14}")
        print()

    print("Comparación clave: mira la fila 'ACTUAL' vs 'Palancas A+B' para el MISMO n_qubits.")
    print("Si la profundidad baja de forma sustancial y la fidelidad mejora, ahí está la")
    print("respuesta real a 'se puede reducir la profundidad' -- sin tocar el número de qubits.")


if __name__ == "__main__":
    main()