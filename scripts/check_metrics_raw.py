"""
scripts/check_metrics_raw.py
Comprueba si job.metrics() da de verdad un tiempo distinto por job, o si
los 0.60s/eval de las 5 corridas son un artefacto de medición.
Uso:
    export IBM_QUANTUM_TOKEN='...'
    python scripts/check_metrics_raw.py d9brt57u62qs738plv3g d9brt6nu62qs738plv80 \
        d9brtaug26ic73dgck70 d9brtc6g26ic73dgcka0 d9brui6g26ic73dgcn5g
"""
import sys, os, json
from qiskit_ibm_runtime import QiskitRuntimeService

token = os.environ["IBM_QUANTUM_TOKEN"]
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)

for job_id in sys.argv[1:]:
    job = service.job(job_id)
    metrics = job.metrics()
    print(f"\n{job_id}")
    print(json.dumps(metrics, indent=2, default=str))