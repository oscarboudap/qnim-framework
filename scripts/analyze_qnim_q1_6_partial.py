"""Build a validation-only analysis from the preserved Q1.6 partial snapshot."""

import argparse
import hashlib
import json
import os
import statistics
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    ROOT
    / "results"
    / "scaling_study"
    / "qnim_q1_6_scaling_pilot_v1_partial_20261120.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "scaling_study"
    / "qnim_q1_6_scaling_pilot_v1_partial_analysis.json"
)
METRICS = ("balanced_accuracy", "accuracy", "macro_f1", "cross_entropy")
INTERRUPTED_RUN = {
    "model_family": "qnim",
    "arm": "fixed_input_dimension",
    "n_qubits": 16,
    "n_features": 4,
    "initialization_seed": 42,
}


def _key(row):
    return (
        row["model_family"],
        row["arm"],
        row["n_qubits"],
        row["initialization_seed"],
    )


def _key_object(key):
    return {
        "model_family": key[0],
        "arm": key[1],
        "n_qubits": key[2],
        "initialization_seed": key[3],
    }


def _expected_keys(configuration):
    keys = set()
    fixed_dimension = configuration["arms"]["fixed_input_dimension"][
        "n_features"
    ]
    for seed in configuration["training"]["initialization_seeds"]:
        for n_qubits in configuration["qubit_counts"]:
            keys.add(("pca_svm_rbf", "full_system", n_qubits, seed))
            keys.add(("qnim", "full_system", n_qubits, seed))
            if n_qubits > fixed_dimension:
                keys.add(("qnim", "fixed_input_dimension", n_qubits, seed))
    return keys


def _assert_validation_only(report):
    if report.get("experiment_id") != "QNIM_Q1_6_SCALING_PILOT_V1":
        raise ValueError("Input is not the Q1.6 pilot report")
    if report.get("final_test", {}).get("opened") is not False:
        raise ValueError("Final-test access is not locked in the source report")
    if report.get("final_test", {}).get("evaluated") is not False:
        raise ValueError("Final-test evaluation is not locked in the source report")
    test_access = report.get("dataset", {}).get("test_access", {})
    if not test_access or any(value is not False for value in test_access.values()):
        raise ValueError("The source report records final-test data access")


def _metric_summary(rows, metric):
    per_seed = [
        {
            "seed": row["initialization_seed"],
            "value": row["metrics"]["validation"][metric],
        }
        for row in rows
    ]
    values = [item["value"] for item in per_seed]
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "n_initializations": len(values),
        "per_seed": per_seed,
    }


def _recall_summary(rows, class_names):
    result = {}
    for class_name in class_names:
        values = [
            {
                "seed": row["initialization_seed"],
                "value": row["metrics"]["validation"]["class_recall"][
                    class_name
                ],
            }
            for row in rows
        ]
        recalls = [item["value"] for item in values]
        result[class_name] = {
            "mean": statistics.mean(recalls),
            "sample_std": (
                statistics.stdev(recalls) if len(recalls) > 1 else None
            ),
            "per_seed": values,
        }
    return result


def _total_seconds(row):
    timing = row.get("timing", {})
    if "training_and_evaluation_seconds" in timing:
        return timing["training_and_evaluation_seconds"]
    return timing.get("fit_and_evaluation_seconds")


def _aggregate(completed, configuration, class_names):
    grouped = {}
    for row in completed:
        grouped.setdefault((row["model_family"], row["arm"]), {}).setdefault(
            row["n_qubits"], []
        ).append(row)

    summary = {}
    for group, qubit_rows in grouped.items():
        group_key = ":".join(group)
        summary[group_key] = {}
        for n_qubits, rows in sorted(qubit_rows.items()):
            ordered_rows = sorted(rows, key=lambda row: row["initialization_seed"])
            total_times = [
                _total_seconds(row)
                for row in ordered_rows
                if _total_seconds(row) is not None
            ]
            trainer_times = [
                row["timing"]["trainer_seconds"]
                for row in ordered_rows
                if row.get("timing", {}).get("trainer_seconds") is not None
            ]
            memory_values = [
                row["memory"]["rss_peak_bytes"]
                for row in ordered_rows
                if row.get("memory", {}).get("rss_peak_bytes") is not None
            ]
            metrics = {
                metric: _metric_summary(ordered_rows, metric)
                for metric in METRICS
            }
            confusion_matrices = [
                {
                    "seed": row["initialization_seed"],
                    "matrix": row["metrics"]["validation"]["confusion_matrix"],
                }
                for row in ordered_rows
            ]
            summary[group_key][str(n_qubits)] = {
                "n_initializations": len(ordered_rows),
                "initialization_seeds": [
                    row["initialization_seed"] for row in ordered_rows
                ],
                "n_features": ordered_rows[0]["n_features"],
                "metrics": metrics,
                "validation_class_recall": _recall_summary(
                    ordered_rows, class_names
                ),
                "validation_confusion_matrices": confusion_matrices,
                "total_seconds_per_initialization": [
                    {
                        "seed": row["initialization_seed"],
                        "seconds": _total_seconds(row),
                    }
                    for row in ordered_rows
                ],
                "mean_total_seconds": (
                    statistics.mean(total_times) if total_times else None
                ),
                "trainer_seconds_per_initialization": [
                    {
                        "seed": row["initialization_seed"],
                        "seconds": row.get("timing", {}).get(
                            "trainer_seconds"
                        ),
                    }
                    for row in ordered_rows
                ],
                "mean_trainer_seconds": (
                    statistics.mean(trainer_times) if trainer_times else None
                ),
                "mean_peak_rss_bytes": (
                    statistics.mean(memory_values) if memory_values else None
                ),
                "quantum_trainable_parameters": ordered_rows[0].get(
                    "parameters", {}
                ).get("quantum_trainable_count"),
                "classical_trainable_parameters": ordered_rows[0].get(
                    "parameters", {}
                ).get("classical_trainable_count"),
                "n_observables": ordered_rows[0].get(
                    "circuit", {}
                ).get("n_observables"),
                "svm_support_vector_count": ordered_rows[0].get(
                    "classifier", {}
                ).get("support_vector_count"),
                "svm_stored_numeric_parameter_count": ordered_rows[0].get(
                    "classifier", {}
                ).get("stored_numeric_parameter_count"),
            }

    qnim_full = summary.get("qnim:full_system", {})
    qnim_fixed = summary.get("qnim:fixed_input_dimension", {})
    svm = summary.get("pca_svm_rbf:full_system", {})
    fixed_dimension = configuration["arms"]["fixed_input_dimension"][
        "n_features"
    ]
    if str(fixed_dimension) not in qnim_fixed and str(fixed_dimension) in qnim_full:
        qnim_fixed[str(fixed_dimension)] = qnim_full[str(fixed_dimension)]

    comparisons = {"qnim_full_minus_svm": [], "qnim_full_minus_fixed": []}
    for n_qubits in configuration["qubit_counts"]:
        key = str(n_qubits)
        full_rows = {
            row["initialization_seed"]: row
            for row in completed
            if row["model_family"] == "qnim"
            and row["arm"] == "full_system"
            and row["n_qubits"] == n_qubits
        }
        for name, other_rows in (
            (
                "qnim_full_minus_svm",
                {
                    row["initialization_seed"]: row
                    for row in completed
                    if row["model_family"] == "pca_svm_rbf"
                    and row["arm"] == "full_system"
                    and row["n_qubits"] == n_qubits
                },
            ),
            (
                "qnim_full_minus_fixed",
                {
                    row["initialization_seed"]: row
                    for row in completed
                    if row["model_family"] == "qnim"
                    and row["arm"] == "fixed_input_dimension"
                    and row["n_qubits"] == n_qubits
                },
            ),
        ):
            if name == "qnim_full_minus_fixed" and n_qubits == fixed_dimension:
                differences = [
                    {
                        "seed": seed,
                        "difference": 0.0,
                        "shared_fit_alias": True,
                    }
                    for seed in sorted(full_rows)
                ]
            else:
                matching_seeds = sorted(full_rows.keys() & other_rows.keys())
                differences = [
                    {
                        "seed": seed,
                        "difference": (
                            full_rows[seed]["metrics"]["validation"][
                                "balanced_accuracy"
                            ]
                            - other_rows[seed]["metrics"]["validation"][
                                "balanced_accuracy"
                            ]
                        ),
                        "shared_fit_alias": False,
                    }
                    for seed in matching_seeds
                ]
            comparisons[name].append({
                "n_qubits": n_qubits,
                "paired_differences": differences,
                "mean_difference": (
                    statistics.mean(
                        item["difference"] for item in differences
                    )
                    if differences
                    else None
                ),
                "n_matched_initializations": len(differences),
            })

    return {
        "by_model_and_arm": summary,
        "paired_balanced_accuracy_differences": comparisons,
        "statistical_caution": (
            "Each available configuration has only one model initialization "
            "and all runs share dataset seed 42. Sample variability and "
            "confirmatory significance cannot be estimated from this snapshot."
        ),
    }


def _run_summary(row):
    return {
        "model_family": row["model_family"],
        "arm": row["arm"],
        "n_qubits": row["n_qubits"],
        "n_features": row["n_features"],
        "dataset_seed": row.get("dataset_seed"),
        "initialization_seed": row["initialization_seed"],
        "train_sha256": row.get("train_sha256"),
        "validation_sha256": row.get("validation_sha256"),
        "fit_scope": row.get("fit_scope"),
        "metrics": row.get("metrics"),
        "parameters": {
            "quantum_trainable_count": row.get("parameters", {}).get(
                "quantum_trainable_count"
            ),
            "classical_trainable_count": row.get("parameters", {}).get(
                "classical_trainable_count"
            ),
            "svm_stored_numeric_parameter_count": row.get(
                "classifier", {}
            ).get("stored_numeric_parameter_count"),
        },
        "circuit": row.get("circuit"),
        "classifier": row.get("classifier"),
        "optimization": row.get("optimization"),
        "timing": row.get("timing"),
        "memory": row.get("memory"),
    }


def _build_analysis(report, source_sha256, source_name):
    _assert_validation_only(report)
    if not report.get("dataset", {}).get("dataset_hashes"):
        raise ValueError("The source report does not contain prepared split hashes")

    expected = _expected_keys(report["configuration"])
    completed = [
        row for row in report.get("runs", [])
        if row.get("status") == "completed"
    ]
    completed_keys = [_key(row) for row in completed]
    if len(completed_keys) != len(set(completed_keys)):
        raise ValueError("The source report contains duplicate completed runs")
    unexpected = set(completed_keys) - expected
    if unexpected:
        raise ValueError(f"Completed runs are outside Q1.6 configuration: {unexpected}")

    class_names = report["dataset"]["class_names"]
    aggregates = _aggregate(
        completed, report["configuration"], class_names
    )
    pending = expected - set(completed_keys)
    full_rows = aggregates["by_model_and_arm"].get(
        "qnim:full_system", {}
    )
    full_values = [
        full_rows[str(q)]["metrics"]["balanced_accuracy"]["mean"]
        for q in report["configuration"]["qubit_counts"]
        if str(q) in full_rows
    ]
    svm_rows = aggregates["by_model_and_arm"].get(
        "pca_svm_rbf:full_system", {}
    )
    qnim_below_svm = [
        q for q in report["configuration"]["qubit_counts"]
        if str(q) in full_rows
        and str(q) in svm_rows
        and full_rows[str(q)]["metrics"]["balanced_accuracy"]["mean"]
        < svm_rows[str(q)]["metrics"]["balanced_accuracy"]["mean"]
    ]

    timing_warnings = []
    for row in completed:
        timing = row.get("timing", {})
        outer = _total_seconds(row)
        trainer = timing.get("trainer_seconds")
        if outer is not None and trainer is not None and abs(outer - trainer) > 300:
            timing_warnings.append({
                "run": _key_object(_key(row)),
                "outer_total_seconds": outer,
                "trainer_seconds": trainer,
                "absolute_difference_seconds": abs(outer - trainer),
                "note": (
                    "Retained as recorded. The independently timed outer "
                    "duration is used for cross-configuration total-time comparisons."
                ),
            })

    return {
        "schema_version": 1,
        "analysis_id": "QNIM_Q1_6_PARTIAL_ANALYSIS_V1",
        "status": "partial_analysis",
        "source": {
            "file": source_name,
            "sha256": source_sha256,
            "experiment_id": report["experiment_id"],
            "source_report_status": report["status"],
        },
        "scope": {
            "expected_run_count": len(expected),
            "completed_run_count": len(completed),
            "pending_run_count": len(pending),
            "dataset_seed": report["dataset"]["dataset_seed"],
            "initialization_seeds_available": sorted({
                row["initialization_seed"] for row in completed
            }),
            "available_runs": [
                _key_object(key) for key in sorted(completed_keys)
            ],
            "pending_runs": [
                _key_object(key) for key in sorted(pending)
            ],
            "interrupted_run": {
                **INTERRUPTED_RUN,
                "persisted": False,
                "note": (
                    "The 16-qubit fixed-input QNIM run for seed 42 was active "
                    "when execution was stopped; incomplete runs are not stored "
                    "in the source report."
                ),
            },
        },
        "dataset": {
            "hashes": report["dataset"]["dataset_hashes"],
            "first_12_component_hashes_match_q1": report["dataset"].get(
                "first_12_component_hashes_match_q1"
            ),
            "test_access": report["dataset"]["test_access"],
            "final_test": report["final_test"],
        },
        "aggregates": aggregates,
        "run_details": [
            _run_summary(row)
            for row in sorted(completed, key=lambda item: _key(item))
        ],
        "source_errors": report.get("errors", []),
        "timing_warnings": timing_warnings,
        "descriptive_findings": {
            "full_system_balanced_accuracy_increases_at_each_available_step": (
                len(full_values) >= 2
                and all(a < b for a, b in zip(full_values, full_values[1:]))
            ),
            "qnim_full_system_below_svm_at_available_matched_qubit_counts": (
                qnim_below_svm
            ),
            "interpretation": (
                "Descriptive only: one model initialization and one dataset "
                "seed do not establish a reproducible scaling trend. Full-system "
                "scaling also increases PCA input dimension and readout capacity."
            ),
        },
        "methodology": {
            "analysis_only": True,
            "training_performed": False,
            "dataset_regenerated": False,
            "test_data_loaded_or_evaluated": False,
            "aggregation_source": "completed validation records only",
        },
    }


def _format_duration(seconds):
    if seconds is None:
        return "—"
    seconds = int(round(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    prefix = f"{days}d " if days else ""
    return f"{prefix}{hours:02}:{minutes:02}:{seconds:02}"


def _write_markdown(analysis, path):
    aggregates = analysis["aggregates"]["by_model_and_arm"]
    lines = [
        "# Q1.6 — Análisis parcial de escalabilidad",
        "",
        f"- **Estado:** `{analysis['status']}`",
        f"- **Fuente:** `{analysis['source']['file']}`",
        f"- **SHA-256 de la fuente:** `{analysis['source']['sha256']}`",
        f"- **Ejecuciones:** {analysis['scope']['completed_run_count']}/"
        f"{analysis['scope']['expected_run_count']} completadas; "
        f"{analysis['scope']['pending_run_count']} pendientes.",
        "- **Evaluación:** solo validation; test final bloqueado y no accedido.",
        "- **Semillas disponibles:** "
        f"{analysis['scope']['initialization_seeds_available']} (inicialización "
        "del modelo; un único dataset seed).",
        "",
        "## Métricas disponibles",
        "",
        "| Modelo | Línea | Qubits | Features | Seeds | Balanced accuracy | Accuracy | Macro-F1 | Cross-entropy | Tiempo total medio |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for group_key, family, arm in (
        ("qnim:full_system", "QNIM", "full_system"),
        ("qnim:fixed_input_dimension", "QNIM", "fixed_input_dimension"),
        ("pca_svm_rbf:full_system", "PCA + SVM RBF", "full_system"),
    ):
        for q, record in sorted(
            aggregates.get(group_key, {}).items(), key=lambda item: int(item[0])
        ):
            metrics = record["metrics"]
            ba = metrics["balanced_accuracy"]
            std = (
                f" ± {ba['sample_std']:.4f}"
                if ba["sample_std"] is not None
                else " (n=1; SD no estimable)"
            )
            n_features = (
                record["n_features"]
            )
            total_seconds = record["mean_total_seconds"]
            formatted_time = (
                f"{_format_duration(total_seconds)} ({total_seconds:.1f} s)"
                if total_seconds is not None
                else "—"
            )
            lines.append(
                f"| {family} | {arm} | {q} | {n_features} | "
                f"{record['initialization_seeds']} | "
                f"{ba['mean']:.4f}{std} | "
                f"{metrics['accuracy']['mean']:.4f} | "
                f"{metrics['macro_f1']['mean']:.4f} | "
                f"{metrics['cross_entropy']['mean']:.4f} | "
                f"{formatted_time} |"
            )
    lines.extend([
        "",
        "La dimensión fija de 4 features en q=4 es un alias del ajuste full-system "
        "de q=4; no representa una segunda ejecución independiente.",
        "",
        "## Comparaciones pareadas",
        "",
        "| Comparación | Qubits | Inicializaciones coincidentes | Diferencia de balanced accuracy (QNIM full − control) |",
        "|---|---:|---|---:|",
    ])
    for name, label in (
        ("qnim_full_minus_svm", "QNIM full − PCA + SVM"),
        ("qnim_full_minus_fixed", "QNIM full − QNIM fixed-input"),
    ):
        for item in analysis["aggregates"][
            "paired_balanced_accuracy_differences"
        ][name]:
            if item["n_matched_initializations"]:
                values = ", ".join(
                    f"seed {row['seed']}: {row['difference']:+.4f}"
                    + (" (mismo ajuste)" if row["shared_fit_alias"] else "")
                    for row in item["paired_differences"]
                )
            else:
                values = "sin comparación disponible"
            lines.append(
                f"| {label} | {item['n_qubits']} | "
                f"{item['n_matched_initializations']} | {values} |"
            )
    lines.extend([
        "",
        "## Ejecuciones pendientes",
        "",
        "La ejecución interrumpida fue QNIM fixed-input, 16 qubits, 4 features, "
        "semilla 42; no llegó a registrarse. Las demás claves pendientes son:",
        "",
    ])
    for row in analysis["scope"]["pending_runs"]:
        marker = (
            " — interrumpida, no persistida"
            if row == analysis["scope"]["interrupted_run"]
            else ""
        )
        lines.append(
            f"- `{row['model_family']}` / `{row['arm']}` / "
            f"q={row['n_qubits']} / seed={row['initialization_seed']}{marker}"
        )
    findings = analysis["descriptive_findings"]
    lines.extend([
        "",
        "## Lectura y límites",
        "",
        (
            "- La balanced accuracy de QNIM full-system aumenta en cada "
            "configuración disponible."
            if findings[
                "full_system_balanced_accuracy_increases_at_each_available_step"
            ]
            else "- No hay una secuencia completa disponible para afirmar un aumento monótono."
        ),
        "- PCA + SVM supera a QNIM full-system en los qubits disponibles: "
        + (
            ", ".join(
                f"q={q}" for q in findings[
                    "qnim_full_system_below_svm_at_available_matched_qubit_counts"
                ]
            )
            or "ninguno"
        )
        + ".",
        "- Una sola inicialización y un solo dataset no permiten estimar "
        "variabilidad entre semillas ni demostrar escalabilidad reproducible.",
        "- En full-system cambian a la vez el número de qubits y las componentes "
        "PCA; el control SVM no aísla completamente esos factores.",
        "- Los tiempos totales externos se usan para comparar configuraciones. "
        "Se conservan aparte los tiempos internos del entrenador.",
        "",
        "### Diferencias en los tiempos registrados",
        "",
    ])
    if analysis["timing_warnings"]:
        for warning in analysis["timing_warnings"]:
            run = warning["run"]
            lines.append(
                f"- `{run['model_family']}/{run['arm']}/q={run['n_qubits']}/"
                f"seed={run['initialization_seed']}`: externo "
                f"{_format_duration(warning['outer_total_seconds'])} "
                f"({warning['outer_total_seconds']:.1f} s), interno "
                f"{_format_duration(warning['trainer_seconds'])} "
                f"({warning['trainer_seconds']:.1f} s). Se reportan ambos sin "
                "intentar reconciliarlos."
            )
    else:
        lines.append("- No se encontraron diferencias internas/externas mayores a 300 s.")
    lines.extend([
        "",
        "### Errores conservados en el informe fuente",
        "",
    ])
    if analysis["source_errors"]:
        lines.extend(
            f"- `{error.get('type', 'Error')}` ({error.get('stage', 'sin etapa')}): "
            f"{error.get('message', '')}"
            for error in analysis["source_errors"]
        )
        lines.append(
            "- Son errores de intentos de preparación previos; la copia registra "
            "que el dataset se preparó posteriormente y que los hashes coincidieron."
        )
    else:
        lines.append("- Ninguno.")
    lines.extend([
        "",
        "No se entrenaron modelos, no se regeneró el dataset y no se accedió al test.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, allow_nan=False)
            temporary.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def analyze(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT):
    input_path = Path(input_path)
    output_path = Path(output_path)
    markdown_path = output_path.with_suffix(".md")
    if output_path.exists() or markdown_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing partial-analysis report"
        )
    source_bytes = input_path.read_bytes()
    report = json.loads(source_bytes)
    analysis = _build_analysis(
        report,
        hashlib.sha256(source_bytes).hexdigest().upper(),
        input_path.name,
    )
    _atomic_json(output_path, analysis)
    _write_markdown(analysis, markdown_path)
    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Analyze only completed Q1.6 validation records"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    analysis = analyze(args.input, args.output)
    print(json.dumps({
        "status": analysis["status"],
        "completed_runs": analysis["scope"]["completed_run_count"],
        "pending_runs": analysis["scope"]["pending_run_count"],
        "final_test": analysis["dataset"]["final_test"]["status"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
