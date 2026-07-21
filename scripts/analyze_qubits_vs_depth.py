"""
scripts/analyze_qubits_vs_depth.py  (v2 -- métricas ampliadas)
================================================================
Además de profundidad y tiempo (v1), esta versión añade las métricas
que hacen falta para justificar el número de qubits de verdad ante
Rodrigo, no solo describirlo:

  - PROFUNDIDAD IDEAL vs REAL: profundidad si el hardware tuviera
    conectividad completa (sin necesidad de SWAPs) vs la profundidad
    real tras enrutar sobre heavy-hex. La diferencia es el overhead de
    routing -- exactamente el efecto que Óscar alegaba en el email del
    1 de julio ("más qubits -> menos SWAPs"), aquí MEDIDO, no supuesto.
  - DESGLOSE DE PUERTAS por tipo (1 qubit, 2 qubits, medida).
  - FIDELIDAD ESTIMADA a partir de las tasas de error de calibración
    reales del backend (product de (1-error) por puerta y por
    qubit de lectura) -- esto es lo que conecta directamente con la
    tabla de fidelidad 0.71->0.88-0.92 que Óscar prometió en su email.
  - FIDELIDAD LIMITADA POR DECOHERENCIA (T1/T2 de los qubits físicos
    concretos que el compilador elige, vs el tiempo real que tarda el
    circuito en ejecutarse).
  - MÉTRICAS DE COLA/EJECUCIÓN REALES del job (no estimaciones):
    pending_jobs del backend en el momento del envío, tiempo de cola
    medido por IBM (creation->running) y tiempo de ejecución
    (running->finished), por separado.

Sigue funcionando igual que v1 en cuanto a flujo (fase de envío no
bloqueante + fase de espera con progreso), solo que ahora cada fila de
la tabla trae muchas más columnas.

Uso: igual que v1.
    export IBM_QUANTUM_TOKEN='...'
    python scripts/analyze_qubits_vs_depth.py --backend ibm_fez \
        --qubit-counts 12 27 33 40 50
"""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.infrastructure.qiskit_vqc_trainer import _build_feature_map_and_ansatz  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Tabla real qubits -> profundidad -> tiempo -> fidelidad")
    p.add_argument("--backend", type=str, default="ibm_fez")
    p.add_argument("--qubit-counts", type=int, nargs="+", default=[12, 27, 33, 40, 50])
    p.add_argument("--ansatz-reps", type=int, default=1)
    p.add_argument("--n-trials", type=int, default=5,
                   help="Nº de transpilaciones independientes por n_qubits (Sabre es estocástico).")
    p.add_argument("--pilot-evals", type=int, default=5,
                   help="Nº de evaluaciones reales en QPU para medir tiempo/eval a cada n_qubits.")
    p.add_argument("--shots", type=int, default=512)
    p.add_argument("--n-samples-retrain", type=int, default=1500)
    p.add_argument("--iters-retrain-low", type=int, default=5)
    p.add_argument("--iters-retrain-high", type=int, default=8)
    p.add_argument("--out-dir", type=str, default="results")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────
#  PROFUNDIDAD / PUERTAS: real (con routing) vs ideal (sin routing)
# ─────────────────────────────────────────────────────────────────────────

def _count_ops_2q_1q(circuit):
    """Separa el conteo de operaciones en (n_1q, n_2q, n_medidas)."""
    n_1q, n_2q, n_meas = 0, 0, 0
    for instr in circuit.data:
        n_qubits_instr = len(instr.qubits)
        name = instr.operation.name
        if name == "measure":
            n_meas += 1
        elif name in ("barrier", "delay"):
            continue
        elif n_qubits_instr >= 2:
            n_2q += 1
        else:
            n_1q += 1
    return n_1q, n_2q, n_meas


def transpile_stats(n_qubits: int, ansatz_reps: int, backend, n_trials: int):
    """
    Transpila n_trials veces (seeds distintas) contra el backend real
    (CON restricción de conectividad heavy-hex) y también UNA vez sin
    coupling_map (routing_method='none', conectividad completa) como
    referencia "ideal" para poder medir el overhead de SWAPs.
    """
    from qiskit import transpile

    combined, _, _ = _build_feature_map_and_ansatz(n_qubits, reps=ansatz_reps)
    combined = combined.copy()
    combined.measure_all()

    # --- referencia ideal: sin mapa de acoplamiento, solo basis gates ---
    try:
        ideal = transpile(
            combined, basis_gates=list(backend.target.operation_names),
            optimization_level=3, seed_transpiler=0,
        )
        ideal_depth = ideal.depth()
        _, ideal_2q, _ = _count_ops_2q_1q(ideal)
    except Exception:
        ideal_depth, ideal_2q = None, None

    # --- real: con coupling_map / target completo del backend ---
    depths, cx_counts, layouts, isa_last = [], [], [], None
    for seed in range(n_trials):
        isa = transpile(combined, backend=backend, optimization_level=3, seed_transpiler=seed)
        depths.append(isa.depth())
        n_1q, n_2q, n_meas = _count_ops_2q_1q(isa)
        cx_counts.append(n_2q)
        if isa.layout is not None:
            phys_qubits = list(isa.layout.final_index_layout())
            layouts.append(phys_qubits)
        isa_last = isa

    depth_mediana = statistics.median(depths)
    cx_mediana = statistics.median(cx_counts)
    depth_min = min(depths)
    cx_min = min(cx_counts)

    overhead_depth_pct = (
        100.0 * (depth_mediana - ideal_depth) / ideal_depth
        if ideal_depth else None
    )
    overhead_2q_pct = (
        100.0 * (cx_mediana - ideal_2q) / ideal_2q
        if ideal_2q else None
    )

    n_1q_last, n_2q_last, n_meas_last = _count_ops_2q_1q(isa_last)
    used_qubits = layouts[-1] if layouts else list(range(n_qubits))

    return {
        "depth_mediana": depth_mediana,
        "depth_min": depth_min,
        "cx_mediana": cx_mediana,
        "cx_min": cx_min,
        "depth_ideal": ideal_depth,
        "cx_ideal": ideal_2q,
        "overhead_depth_pct": overhead_depth_pct,
        "overhead_2q_pct": overhead_2q_pct,
        "n_1q_gates": n_1q_last,
        "n_2q_gates": n_2q_last,
        "n_medidas": n_meas_last,
        "isa_example": isa_last,
        "physical_qubits": used_qubits,
    }


# ─────────────────────────────────────────────────────────────────────────
#  FIDELIDAD ESTIMADA a partir de la calibración real del backend
# ─────────────────────────────────────────────────────────────────────────

def estimate_fidelity(isa_circuit, backend, physical_qubits):
    """
    Fidelidad estimada = producto de (1 - error) sobre cada puerta y
    cada medida del circuito transpilado, usando las tasas de error de
    calibración REALES del backend (backend.target), no valores
    supuestos de la literatura.

    También calcula la fidelidad limitada por decoherencia: T1/T2
    medios de los qubits físicos concretos que el compilador asignó,
    comparados con la duración total estimada del circuito.

    Devuelve (fidelidad_por_error_puertas, fidelidad_por_decoherencia,
    t1_medio_ns, t2_medio_ns, duracion_circuito_estimada_ns) -- alguno
    puede ser None si el backend no publica esa calibración.
    """
    target = backend.target
    fid_gates = 1.0
    total_duration_s = 0.0
    got_any_error = False
    got_any_duration = False

    for instr in isa_circuit.data:
        name = instr.operation.name
        if name in ("barrier", "measure", "delay"):
            continue
        qargs = tuple(isa_circuit.find_bit(q).index for q in instr.qubits)
        try:
            props = target[name][qargs]
            if props is not None and props.error is not None:
                fid_gates *= (1.0 - props.error)
                got_any_error = True
            if props is not None and props.duration is not None:
                total_duration_s += props.duration
                got_any_duration = True
        except (KeyError, TypeError):
            continue

    # Error de lectura de los qubits físicos usados en la medida final
    try:
        for q in physical_qubits:
            ro = target["measure"][(q,)]
            if ro is not None and ro.error is not None:
                fid_gates *= (1.0 - ro.error)
                got_any_error = True
    except (KeyError, TypeError):
        pass

    fid_gates = fid_gates if got_any_error else None

    # T1/T2 medios de los qubits físicos usados
    t1_vals, t2_vals = [], []
    try:
        for q in physical_qubits:
            qp = target.qubit_properties[q]
            if qp.t1 is not None:
                t1_vals.append(qp.t1)
            if qp.t2 is not None:
                t2_vals.append(qp.t2)
    except Exception:
        pass

    t1_mean = statistics.mean(t1_vals) if t1_vals else None
    t2_mean = statistics.mean(t2_vals) if t2_vals else None

    fid_decoherence = None
    if got_any_duration and t2_mean:
        import math
        fid_decoherence = math.exp(-total_duration_s / t2_mean)

    return {
        "fid_error_puertas": fid_gates,
        "fid_decoherencia": fid_decoherence,
        "t1_medio_ns": (t1_mean * 1e9) if t1_mean else None,
        "t2_medio_ns": (t2_mean * 1e9) if t2_mean else None,
        "duracion_circuito_ns": total_duration_s * 1e9 if got_any_duration else None,
    }


# ─────────────────────────────────────────────────────────────────────────
#  ENVÍO / ESPERA DE PILOTOS (no bloqueante, con métricas de cola reales)
# ─────────────────────────────────────────────────────────────────────────

def submit_pilot_job(isa_circuit, backend, shots: int, n_evals: int):
    """Lanza n_evals evaluaciones reales, sin bloquear. Devuelve (job, pending_jobs_al_envio)."""
    import numpy as np
    from qiskit_ibm_runtime import SamplerV2

    try:
        pending_jobs_now = backend.status().pending_jobs
    except Exception:
        pending_jobs_now = None

    sampler = SamplerV2(mode=backend)
    rng = np.random.default_rng(0)
    n_params = isa_circuit.num_parameters

    pubs = []
    for _ in range(n_evals):
        theta = rng.uniform(-3.14, 3.14, n_params)
        bound = isa_circuit.assign_parameters(theta)
        pubs.append((bound,))

    job = sampler.run(pubs, shots=shots)
    print(f"    -> job enviado: {job.job_id()}  "
          f"(cola del backend en ese momento: {pending_jobs_now} jobs pendientes)  "
          f"https://quantum.cloud.ibm.com/jobs/{job.job_id()}")
    return job, pending_jobs_now


def wait_for_pilot(job, n_evals: int, poll_every: int = 20, max_wait_s: int = 3600):
    """
    Espera un job ya enviado, con progreso visible, y al terminar
    separa tiempo de COLA (creation->running) de tiempo de EJECUCIÓN
    (running->finished) usando los timestamps reales que reporta IBM,
    no un cronómetro local que mezcla ambos.
    """
    t0 = time.time()
    last_status = None
    while True:
        status = job.status()
        if status != last_status:
            elapsed = time.time() - t0
            print(f"    [{elapsed:6.0f}s] estado: {status}")
            last_status = status
        if str(status) in ("DONE", "ERROR", "CANCELLED"):
            break
        if time.time() - t0 > max_wait_s:
            print(f"    aviso: llevas más de {max_wait_s}s esperando este job. "
                  f"Puedes recuperar el resultado más tarde con:\n"
                  f"      service.job('{job.job_id()}').result()")
        time.sleep(poll_every)

    wall = time.time() - t0
    result = job.result()

    qpu_time, queue_time_s, exec_time_s = None, None, None
    try:
        metrics = job.metrics()
        qpu_time = metrics.get("usage", {}).get("seconds")
        ts = metrics.get("timestamps", {})
        if ts.get("created") and ts.get("running") and ts.get("finished"):
            from datetime import datetime
            fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
            def _parse(s):
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    return datetime.fromisoformat(s.replace("Z", "+00:00"))
            t_created = _parse(ts["created"])
            t_running = _parse(ts["running"])
            t_finished = _parse(ts["finished"])
            queue_time_s = (t_running - t_created).total_seconds()
            exec_time_s = (t_finished - t_running).total_seconds()
    except Exception:
        pass

    per_eval_wall = wall / n_evals
    per_eval_qpu = (qpu_time / n_evals) if qpu_time else None
    return {
        "per_eval_wall": per_eval_wall,
        "per_eval_qpu": per_eval_qpu,
        "queue_time_s": queue_time_s,
        "exec_time_s": exec_time_s,
    }


# ─────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    from qiskit_ibm_runtime import QiskitRuntimeService
    import os

    token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    if not token:
        print("ERROR: exporta IBM_QUANTUM_TOKEN antes de correr esto.")
        sys.exit(1)

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = service.backend(args.backend)
    print(f"Backend: {backend.name}  ({backend.num_qubits} qubits físicos, topología heavy-hex)")

    pending = []
    for n in args.qubit_counts:
        print(f"\n--- n_qubits = {n} ---")
        stats = transpile_stats(n, args.ansatz_reps, backend, args.n_trials)
        print(f"  Profundidad real (mediana de {args.n_trials}): {stats['depth_mediana']}"
              f"  | ideal (sin routing): {stats['depth_ideal']}"
              f"  | overhead: {stats['overhead_depth_pct']:.1f}%" if stats['overhead_depth_pct'] is not None else
              f"  Profundidad real (mediana de {args.n_trials}): {stats['depth_mediana']}")
        print(f"  Puertas 2Q reales: {stats['cx_mediana']}  | ideal: {stats['cx_ideal']}"
              f"  | overhead (SWAPs efectivos): {stats['overhead_2q_pct']:.1f}%"
              if stats['overhead_2q_pct'] is not None else
              f"  Puertas 2Q reales: {stats['cx_mediana']}")
        print(f"  Puertas 1Q: {stats['n_1q_gates']}  |  medidas: {stats['n_medidas']}")

        fid = estimate_fidelity(stats["isa_example"], backend, stats["physical_qubits"])
        if fid["fid_error_puertas"] is not None:
            print(f"  Fidelidad estimada (error de puertas, calibración real): {fid['fid_error_puertas']:.4f}")
        if fid["fid_decoherencia"] is not None:
            print(f"  Fidelidad estimada (decoherencia T2, qubits físicos usados): {fid['fid_decoherencia']:.4f}")
        if fid["t1_medio_ns"] is not None:
            print(f"  T1 medio qubits usados: {fid['t1_medio_ns']/1000:.1f} µs  |  "
                  f"T2 medio: {fid['t2_medio_ns']/1000:.1f} µs")

        job, pending_jobs_now = submit_pilot_job(stats["isa_example"], backend, args.shots, args.pilot_evals)
        pending.append((n, stats, fid, job, pending_jobs_now))

    print("\nTodos los pilotos enviados. Esperando resultados "
          "(cola compartida del plan Open, puede tardar) ...")

    rows = []
    for n, stats, fid, job, pending_jobs_now in pending:
        print(f"\nEsperando job de n_qubits={n} ({job.job_id()}) ...")
        timing = wait_for_pilot(job, args.pilot_evals)
        t_per_eval = timing["per_eval_qpu"] if timing["per_eval_qpu"] is not None else timing["per_eval_wall"]
        fuente = "QPU (job.metrics)" if timing["per_eval_qpu"] is not None else "wall-clock local (incluye cola)"
        print(f"  Tiempo/evaluación medido ({fuente}): {t_per_eval:.2f} s")
        if timing["queue_time_s"] is not None:
            print(f"  Tiempo de cola real (IBM): {timing['queue_time_s']:.1f} s  |  "
                  f"tiempo de ejecución real: {timing['exec_time_s']:.1f} s")

        for n_iter in (args.iters_retrain_low, args.iters_retrain_high):
            n_evals_total = args.n_samples_retrain * n_iter
            total_hours = n_evals_total * t_per_eval / 3600.0
            rows.append({
                "n_qubits": n,
                "profundidad_real": stats["depth_mediana"],
                "profundidad_ideal": stats["depth_ideal"],
                "overhead_profundidad_pct": (
                    round(stats["overhead_depth_pct"], 1) if stats["overhead_depth_pct"] is not None else ""
                ),
                "puertas_2q_reales": stats["cx_mediana"],
                "puertas_2q_ideal": stats["cx_ideal"],
                "overhead_2q_pct": (
                    round(stats["overhead_2q_pct"], 1) if stats["overhead_2q_pct"] is not None else ""
                ),
                "puertas_1q": stats["n_1q_gates"],
                "n_medidas": stats["n_medidas"],
                "fidelidad_error_puertas": (
                    round(fid["fid_error_puertas"], 4) if fid["fid_error_puertas"] is not None else ""
                ),
                "fidelidad_decoherencia": (
                    round(fid["fid_decoherencia"], 4) if fid["fid_decoherencia"] is not None else ""
                ),
                "t1_medio_us": (
                    round(fid["t1_medio_ns"] / 1000, 1) if fid["t1_medio_ns"] is not None else ""
                ),
                "t2_medio_us": (
                    round(fid["t2_medio_ns"] / 1000, 1) if fid["t2_medio_ns"] is not None else ""
                ),
                "duracion_circuito_us": (
                    round(fid["duracion_circuito_ns"] / 1000, 2) if fid["duracion_circuito_ns"] is not None else ""
                ),
                "pending_jobs_al_enviar": pending_jobs_now if pending_jobs_now is not None else "",
                "cola_real_s": round(timing["queue_time_s"], 1) if timing["queue_time_s"] is not None else "",
                "ejecucion_real_s": round(timing["exec_time_s"], 1) if timing["exec_time_s"] is not None else "",
                "t_por_eval_s": round(t_per_eval, 2),
                "n_iteraciones": n_iter,
                "n_muestras": args.n_samples_retrain,
                "tiempo_total_horas": round(total_hours, 2),
            })

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"qubits_vs_depth_{int(time.time())}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 100)
    print("TABLA FINAL (resumen -- el CSV completo tiene todas las columnas, incluida fidelidad)")
    print("=" * 100)
    header = (f"{'qubits':>7} {'prof.real':>9} {'prof.ideal':>10} {'ovh%':>6} "
              f"{'2Q':>5} {'fid.err':>8} {'fid.decoh':>10} {'s/eval':>8} {'horas QPU':>10}")
    print(header)
    for r in rows:
        print(f"{r['n_qubits']:>7} {r['profundidad_real']:>9} {r['profundidad_ideal']:>10} "
              f"{r['overhead_profundidad_pct']:>6} {r['puertas_2q_reales']:>5} "
              f"{r['fidelidad_error_puertas']:>8} {r['fidelidad_decoherencia']:>10} "
              f"{r['t_por_eval_s']:>8} {r['tiempo_total_horas']:>10}")
    print(f"\nGuardado en: {out_path}  (columnas completas: {list(rows[0].keys())})")


if __name__ == "__main__":
    main()