"""
scripts/estimate_time_from_calibration.py
==========================================
El pilotos de facturación real (job.metrics()["usage"]["seconds"]) NO
sirve para comparar qubits entre sí: IBM redondea a segundos enteros, y
la ejecución real de estos circuitos dura microsegundos-milisegundos,
muy por debajo de esa resolución (los 5 jobs de la corrida anterior
dieron los 3 segundos exactos, sea cual sea la profundidad).

Este script usa en su lugar la duración de circuito por CALIBRACIÓN
REAL (backend.target[gate][qubits].duration, en segundos, típicamente
ns-us), que sí es sensible a la profundidad porque es una suma de
duraciones de puerta reales, no un contador de facturación redondeado.

Limitación conocida (a diferencia del pilot anterior, aquí NO se
ejecuta nada en hardware, es cálculo puro a partir de la calibración
publicada): esto suma la duración de TODAS las puertas de forma
secuencial, no solo las del camino crítico -- es decir, sobreestima
el tiempo real si hay puertas en paralelo sobre qubits distintos
(que es justo lo que un compilador intenta maximizar). Por eso se
reporta también una estimación de "camino crítico" aproximada
(duración_total / paralelismo_medio, con paralelismo_medio =
n_2q_gates / profundidad como proxy burdo) -- etiquetada explícitamente
como aproximación, no como medida.

Uso:
    export IBM_QUANTUM_TOKEN='...'
    python scripts/estimate_time_from_calibration.py --backend ibm_fez \
        --qubit-counts 12 27 33 40 50 --shots 512 \
        --n-samples-retrain 1500 --iters-retrain-low 5 --iters-retrain-high 8
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.infrastructure.qiskit_vqc_trainer import _build_feature_map_and_ansatz  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", type=str, default="ibm_fez")
    p.add_argument("--qubit-counts", type=int, nargs="+", default=[12, 27, 33, 40, 50])
    p.add_argument("--ansatz-reps", type=int, default=1)
    p.add_argument("--shots", type=int, default=512)
    p.add_argument("--n-samples-retrain", type=int, default=1500)
    p.add_argument("--iters-retrain-low", type=int, default=5)
    p.add_argument("--iters-retrain-high", type=int, default=8)
    return p.parse_args()


def circuit_duration_from_calibration(n_qubits: int, ansatz_reps: int, backend):
    """
    Devuelve (duracion_secuencial_s, duracion_camino_critico_aprox_s,
    profundidad, n_2q_gates) para el circuito real transpilado a este
    backend, usando duraciones de puerta de la calibración publicada.
    """
    from qiskit import transpile

    combined, _, _ = _build_feature_map_and_ansatz(n_qubits, reps=ansatz_reps)
    combined = combined.copy()
    combined.measure_all()
    isa = transpile(combined, backend=backend, optimization_level=3, seed_transpiler=0)

    target = backend.target
    total_duration_s = 0.0
    n_2q = 0
    for instr in isa.data:
        name = instr.operation.name
        if name in ("barrier", "measure", "delay"):
            continue
        qargs = tuple(isa.find_bit(q).index for q in instr.qubits)
        if len(qargs) >= 2:
            n_2q += 1
        try:
            props = target[name][qargs]
            if props is not None and props.duration is not None:
                total_duration_s += props.duration
        except (KeyError, TypeError):
            continue

    depth = isa.depth()
    # proxy burdo de paralelismo: cuantas puertas "caben" en promedio en
    # cada capa de la profundidad -- NO es una medida real de camino
    # crítico, solo una cota inferior razonable para no reportar solo
    # el peor caso (secuencial puro).
    parallelism_proxy = max(1.0, n_2q / max(depth, 1))
    critical_path_approx_s = total_duration_s / parallelism_proxy

    return total_duration_s, critical_path_approx_s, depth, n_2q


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

    header = (f"{'qubits':>7} {'prof.':>6} {'2Q':>5} {'dur.secuencial':>16} "
              f"{'dur.camino_crit~':>18} {'horas (iter=' + str(args.iters_retrain_low) + ')':>16} "
              f"{'horas (iter=' + str(args.iters_retrain_high) + ')':>16}")
    print(header)

    for n in args.qubit_counts:
        dur_seq_s, dur_crit_s, depth, n_2q = circuit_duration_from_calibration(n, args.ansatz_reps, backend)

        # tiempo por evaluación = duración del circuito x shots
        # (asumiendo que los shots se repiten secuencialmente, que es
        # el modelo mas conservador -- si el backend paraleliza shots
        # internamente esto sobreestima, hay que confirmarlo).
        t_per_eval_seq = dur_seq_s * args.shots
        t_per_eval_crit = dur_crit_s * args.shots

        rows = []
        for n_iter in (args.iters_retrain_low, args.iters_retrain_high):
            n_evals_total = args.n_samples_retrain * n_iter
            hours_seq = n_evals_total * t_per_eval_seq / 3600.0
            hours_crit = n_evals_total * t_per_eval_crit / 3600.0
            rows.append((hours_seq, hours_crit))

        print(f"{n:>7} {depth:>6} {n_2q:>5} "
              f"{dur_seq_s*1e6:>13.1f}us {dur_crit_s*1e6:>15.1f}us "
              f"{rows[0][0]:>7.4f}/{rows[0][1]:>7.4f} {rows[1][0]:>7.4f}/{rows[1][1]:>7.4f}")

    print("\nFormato de horas: secuencial/camino_crítico_aproximado (el real está entre ambos).")
    print("Estas horas son ÓRDENES DE MAGNITUD más pequeñas que las de la facturación "
          "redondeada -- si esto es correcto, el cuello de botella real del programa de "
          "Credits probablemente no es el tiempo de cómputo puro, sino: nº de jobs, cola "
          "compartida, o límites de shots/job del plan, no queda claro sin mirar la "
          "documentacion de cuotas de IBM para el plan Credits en concreto.")


if __name__ == "__main__":
    main()