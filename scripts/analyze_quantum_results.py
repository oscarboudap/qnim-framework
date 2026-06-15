#!/usr/bin/env python3
"""
Utilities for analyzing IBM Quantum job results.

Loads and analyzes output JSON files from ibm_quantum_job_executor.py,
providing comparison metrics and thesis-formatted output tables.

Usage:
    python analyze_quantum_results.py --summary-file results/summary.json
    python analyze_quantum_results.py --compare-jobs <job_id_zne> <job_id_no_zne>
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from tabulate import tabulate


def load_job_result(filepath: Path) -> Dict[str, Any]:
    """Load a single job result JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_summary(filepath: Path) -> Dict[str, Any]:
    """Load the summary JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_metrics(job_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from a job result."""
    return {
        "job_name": job_result.get("job_name", "Unknown"),
        "job_id": job_result.get("job_ids", ["N/A"])[0] if job_result.get("job_ids") else "N/A",
        "n_events": job_result.get("n_events", 0),
        "use_zne": job_result.get("use_zne", False),
        "balanced_accuracy": job_result.get("metrics", {}).get("balanced_accuracy", None),
        "precision": job_result.get("metrics", {}).get("precision", None),
        "recall": job_result.get("metrics", {}).get("recall", None),
        "f1_score": job_result.get("metrics", {}).get("f1_score", None),
        "n_correct": sum([1 for pred, true in zip(
            job_result.get("predicted_labels", []),
            job_result.get("true_labels", [])
        ) if pred == true]) if job_result.get("true_labels") else None,
    }


def extract_backend_properties(job_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract backend properties snapshot."""
    props = job_result.get("backend_properties", {})
    
    extracted = {
        "backend_name": props.get("backend_name", "N/A"),
        "backend_version": props.get("backend_version", "N/A"),
        "n_qubits": props.get("n_qubits", "N/A"),
        "calibration_date": props.get("calibration_date", "N/A"),
        "avg_t1": None,
        "std_t1": None,
        "avg_t2": None,
        "std_t2": None,
        "avg_2q_error": None,
        "std_2q_error": None,
    }
    
    if props.get("T1"):
        t1_vals = [v for v in props["T1"].values() if v is not None]
        if t1_vals:
            extracted["avg_t1"] = np.mean(t1_vals)
            extracted["std_t1"] = np.std(t1_vals)
    
    if props.get("T2"):
        t2_vals = [v for v in props["T2"].values() if v is not None]
        if t2_vals:
            extracted["avg_t2"] = np.mean(t2_vals)
            extracted["std_t2"] = np.std(t2_vals)
    
    if props.get("2qubit_gate_errors"):
        errors = [v for v in props["2qubit_gate_errors"].values() if v is not None]
        if errors:
            extracted["avg_2q_error"] = np.mean(errors)
            extracted["std_2q_error"] = np.std(errors)
    
    return extracted


def print_job_summary(job_result: Dict[str, Any]) -> None:
    """Print formatted summary of a single job."""
    metrics = extract_metrics(job_result)
    props = extract_backend_properties(job_result)
    
    print("\n" + "="*80)
    print(f"JOB: {metrics['job_name']}")
    print("="*80)
    
    print(f"\nIdentification:")
    print(f"  Job ID:           {metrics['job_id']}")
    print(f"  Events:           {metrics['n_events']}")
    print(f"  Zero-Noise Extr:  {'Yes' if metrics['use_zne'] else 'No'}")
    
    print(f"\nBackend Properties:")
    print(f"  Backend:          {props['backend_name']}")
    print(f"  Version:          {props['backend_version']}")
    print(f"  Qubits:           {props['n_qubits']}")
    print(f"  Calibrated:       {props['calibration_date']}")
    
    if props.get("avg_t1"):
        print(f"  T₁ (avg):         {props['avg_t1']:.3e} ± {props['std_t1']:.3e} s")
    if props.get("avg_t2"):
        print(f"  T₂ (avg):         {props['avg_t2']:.3e} ± {props['std_t2']:.3e} s")
    if props.get("avg_2q_error"):
        print(f"  2-Q Gate Error:   {props['avg_2q_error']:.3e} ± {props['std_2q_error']:.3e}")
    
    print(f"\nPerformance Metrics:")
    if metrics['balanced_accuracy'] is not None:
        print(f"  Balanced Acc:     {metrics['balanced_accuracy']:.4f}")
        print(f"  Precision:        {metrics['precision']:.4f}")
        print(f"  Recall:           {metrics['recall']:.4f}")
        print(f"  F₁ Score:         {metrics['f1_score']:.4f}")
        if metrics['n_correct'] is not None:
            print(f"  Correct Preds:    {metrics['n_correct']}/{metrics['n_events']}")
    else:
        print(f"  [No metrics available for {metrics['n_events']} event(s)]")


def compare_zne_vs_baseline(
    job_zne: Dict[str, Any],
    job_no_zne: Dict[str, Any]
) -> None:
    """Compare ZNE and baseline results side-by-side."""
    metrics_zne = extract_metrics(job_zne)
    metrics_no_zne = extract_metrics(job_no_zne)
    
    print("\n" + "="*80)
    print("ZNE vs BASELINE COMPARISON")
    print("="*80)
    
    # Create comparison table
    comparison_data = [
        ["Metric", "With ZNE", "No ZNE", "Δ", "% Improvement"],
        ["-" * 20, "-" * 12, "-" * 12, "-" * 8, "-" * 15]
    ]
    
    metrics_to_compare = [
        ("balanced_accuracy", "Balanced Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F₁ Score")
    ]
    
    for metric_key, metric_name in metrics_to_compare:
        val_zne = metrics_zne.get(metric_key)
        val_no_zne = metrics_no_zne.get(metric_key)
        
        if val_zne is not None and val_no_zne is not None:
            delta = val_zne - val_no_zne
            pct_improvement = (delta / val_no_zne * 100) if val_no_zne != 0 else 0
            
            comparison_data.append([
                metric_name,
                f"{val_zne:.4f}",
                f"{val_no_zne:.4f}",
                f"{delta:+.4f}",
                f"{pct_improvement:+.1f}%"
            ])
    
    print("\n" + tabulate(comparison_data, headers="firstrow", tablefmt="grid"))
    
    # Summary
    print("\nSUMMARY:")
    if metrics_zne.get("balanced_accuracy") is not None and metrics_no_zne.get("balanced_accuracy") is not None:
        delta_acc = metrics_zne["balanced_accuracy"] - metrics_no_zne["balanced_accuracy"]
        if delta_acc > 0:
            print(f"  ✓ ZNE provides {delta_acc*100:.1f}% improvement in balanced accuracy")
        elif delta_acc < 0:
            print(f"  ✗ ZNE decreases accuracy by {abs(delta_acc)*100:.1f}%")
        else:
            print(f"  ≈ No significant difference in accuracy")


def generate_thesis_table(
    job_results: List[Dict[str, Any]],
    include_backend: bool = True
) -> str:
    """
    Generate LaTeX table for thesis appendix.
    
    Returns:
        LaTeX table string ready for thesis inclusion
    """
    table_data = []
    
    for jr in job_results:
        metrics = extract_metrics(jr)
        props = extract_backend_properties(jr)
        
        row = [
            metrics['job_name'],
            metrics['job_id'][:8] + "...",
            "Yes" if metrics['use_zne'] else "No",
            metrics['n_events'],
        ]
        
        if include_backend:
            if props.get("avg_t1"):
                t1_sci = f"{props['avg_t1']:.2e}".replace("e-0", "e$-$")
                row.append(f"${t1_sci}$")
            else:
                row.append("—")
            
            if props.get("avg_2q_error"):
                err_sci = f"{props['avg_2q_error']:.2e}".replace("e-0", "e$-$")
                row.append(f"${err_sci}$")
            else:
                row.append("—")
        
        if metrics['balanced_accuracy'] is not None:
            row.append(f"{metrics['balanced_accuracy']:.4f}")
            row.append(f"{metrics['precision']:.4f}")
            row.append(f"{metrics['recall']:.4f}")
            row.append(f"{metrics['f1_score']:.4f}")
        else:
            row.extend(["—", "—", "—", "—"])
        
        table_data.append(row)
    
    # Build LaTeX table
    headers = ["Job", "ID", "ZNE", "Events"]
    if include_backend:
        headers.extend(["T₁ (s)", "2Q Error"])
    headers.extend(["Acc", "Prec", "Rec", "F₁"])
    
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{IBM Quantum VQC Inference Results}\n"
    latex += "\\label{tab:vqc_results}\n"
    latex += "\\begin{tabular}{" + "l" * len(headers) + "}\n"
    latex += "\\hline\n"
    latex += " & ".join(headers) + " \\\\\n"
    latex += "\\hline\n"
    
    for row in table_data:
        latex += " & ".join(str(x) for x in row) + " \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex


def main():
    """Main analysis CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze IBM Quantum job execution results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View summary of all jobs
  python analyze_quantum_results.py --summary-file results/summary.json
  
  # View specific job details
  python analyze_quantum_results.py --job-file results/job_cxyz123.json
  
  # Compare ZNE vs baseline
  python analyze_quantum_results.py \\
    --compare-jobs results/job_zne.json results/job_no_zne.json
  
  # Generate thesis LaTeX table
  python analyze_quantum_results.py \\
    --thesis-table results/ --output thesis_table.tex
        """
    )
    
    parser.add_argument("--summary-file", type=str,
                       help="Path to summary.json file")
    parser.add_argument("--job-file", type=str,
                       help="Path to individual job_*.json file")
    parser.add_argument("--compare-jobs", nargs=2, type=str,
                       metavar=("JOB_ZNE", "JOB_NO_ZNE"),
                       help="Compare two job results (ZNE, baseline)")
    parser.add_argument("--thesis-table", type=str,
                       help="Generate LaTeX table from job directory")
    parser.add_argument("--output", type=str, default="thesis_results_table.tex",
                       help="Output file for LaTeX table (default: thesis_results_table.tex)")
    
    args = parser.parse_args()
    
    if args.summary_file:
        print(f"Loading summary from: {args.summary_file}")
        summary = load_summary(Path(args.summary_file))
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'='*80}")
        print(f"Timestamp:    {summary.get('experiment_timestamp', 'N/A')}")
        print(f"Backend:      {summary.get('backend_name', 'N/A')}")
        print(f"Total Jobs:   {summary.get('total_jobs', 0)}")
        print(f"Job IDs:      {', '.join(summary.get('job_ids', [])[:3])}")
        
        # Metrics summary
        print(f"\n{'Job Performance':^80}")
        for job_summary in summary.get("job_summaries", []):
            metrics = job_summary.get("metrics", {})
            acc = metrics.get("balanced_accuracy")
            if acc is not None:
                print(f"  {job_summary['job_name']:40s} Acc={acc:.4f}")
            else:
                print(f"  {job_summary['job_name']:40s} (no metrics)")
    
    if args.job_file:
        print(f"Loading job result from: {args.job_file}")
        job_result = load_job_result(Path(args.job_file))
        print_job_summary(job_result)
    
    if args.compare_jobs:
        job_zne_path, job_no_zne_path = args.compare_jobs
        print(f"Comparing: {job_zne_path} vs {job_no_zne_path}")
        
        job_zne = load_job_result(Path(job_zne_path))
        job_no_zne = load_job_result(Path(job_no_zne_path))
        
        compare_zne_vs_baseline(job_zne, job_no_zne)
    
    if args.thesis_table:
        results_dir = Path(args.thesis_table)
        job_files = sorted(results_dir.glob("job_*.json"))
        
        if not job_files:
            print(f"ERROR: No job_*.json files found in {results_dir}")
            return
        
        print(f"Loading {len(job_files)} job results from {results_dir}...")
        job_results = [load_job_result(f) for f in job_files]
        
        latex_table = generate_thesis_table(job_results, include_backend=True)
        
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write(latex_table)
        
        print(f"\n✓ LaTeX table saved to: {output_path}")
        print("\nPreview:")
        print(latex_table)


if __name__ == "__main__":
    main()
