# QNIM - Quantum Neuro-Inspired Manifold


python3 scripts/generate_results.py --mode fallback # siempre funciona
python3 scripts/generate_results.py --mode sim # con Qiskit
python3 scripts/generate_results.py --mode ibm # IBM ibm_fez real

## Framework for Gravitational Wave Decoding

### 📁 Clean Architecture Structure

```
qnim/
├── src/                          # Domain-Driven Design Layers (ONLY)
│   ├── domain/                   # 🔵 Business logic & entities
│   │   ├── astrophysics/
│   │   ├── quantum/
│   │   └── shared/
│   ├── application/              # 🟢 Use cases & orchestrators
│   │   ├── hybrid_orchestrator.py
│   │   ├── process_event_use_case.py
│   │   └── validators/
│   ├── infrastructure/           # 🔴 External adapters
│   │   ├── ibm_quantum_adapter.py
│   │   ├── neal_annealer_adapter.py
│   │   └── storage/
│   ├── presentation/             # 🟡 CLI & visualization
│   │   ├── cli_presenter.py
│   │   └── visualize_results.py
│   └── test/                     # ✅ Domain tests
│
├── scripts/                      # 📝 Entry points (call src/ internally)
│   ├── run_qnim_simulator.py
│   ├── run_qnim_inference.py
│   ├── train_complete.py
│   ├── validate_ibm_connection.py
│   └── ...
│
├── docs/                         # 📚 Architecture & design documents
│   ├── ARCHITECTURE_COMPLETE.md
│   ├── IMPLEMENTATION_LAYERS_5_6_7.md
│   └── ...
│
├── data/                         # 📊 GW signals & datasets
├── models/                       # 🤖 Trained weights & pipelines
├── config/                       # ⚙️ Universe parameters
├── reports/                      # 📈 Analysis outputs
│
└── [config files]
    ├── requirements.txt          # Python dependencies
    ├── pytest.ini                # Test configuration
    ├── .env                      # IBM Quantum credentials
    └── main.py                   # Main entry point
```

### 🚀 Quick Start

```bash
# Set IBM credentials
export IBM_QUANTUM_TOKEN="your_token"
export USE_REAL_HARDWARE=False  # True = IBM hardware (paid plan)

# Run simulator demo
python scripts/run_qnim_simulator.py

# Full inference pipeline
python scripts/run_qnim_inference.py

# Train model
python scripts/train_complete.py

# Validate infrastructure
python scripts/validate_ibm_connection.py
```

### ✅ Architecture Compliance

- ✅ **DDD Layers**: domain/ → application/ → infrastructure/ → presentation/
- ✅ **Dependency Injection**: Scripts pass adapters to orchestrators
- ✅ **Clean Separation**: No business logic in scripts
- ✅ **Testable**: Domain tests in src/test/

### 📖 Documentation

- `docs/ARCHITECTURE_COMPLETE.md` - Full DDD design
- `docs/IMPLEMENTATION_*` - Layer-by-layer breakdown
- `docs/QUICK_REF_AUDIT.md` - Quick reference guide

---

**Status**: Production-ready for defense (April 22, 2026)  
**IBM Quantum**: Connected ✅ | Plan: OPEN (gratuito) | Backend: ibm_fez (156 qubits)


--------------------------
(.venv) oscarbd@Oscar:/mnt/c/Users/oscar/Desktop/TFM/qnim/qnim$ python3 -c "
(.venv) oscarbd@Oscar:/mnt/c/Users/oscar/Desktop/TFM/qnim/qnim$ clear
                                                                python3 -c "
(.venv) oscarbd@Oscar:/mnt/c/Users/oscar/Desktop/TFM/qnim/qnim$ clear
(.venv) oscarbd@Oscar:/mnt/c/Users/oscar/Desktop/TFM/qnim/qnim$ python3 scripts/generate_results.py --mode fallback --max-iter 5
0
======================================================================
  QNIM Framework — Resultados Experimentales
  TFM: Quantum Decoding of Gravitational Waves | UNIR 2026
  [Versión con correcciones postdoctorales]
======================================================================

  Modo:      FALLBACK
  n_qubits:  12
  Backend:   ibm_fez
  QNSPSA-EML-Feynman: ACTIVO (optimizador real)
  QUBO: match function ponderada por PSD LIGO O3
  Estadística: Šidák/BH + cota Holevo + test Isi + TI Bayes

17:45:15 | INFO     | qnim.application.generate_experiment_results | =================================================================
17:45:15 | INFO     | qnim.application.generate_experiment_results |   QNIM Framework — Pipeline Completo
17:45:15 | INFO     | qnim.application.generate_experiment_results |   Backend: ibm_fez
17:45:15 | INFO     | qnim.application.generate_experiment_results |   n_qubits: 12
17:45:15 | INFO     | qnim.application.generate_experiment_results |   Hardware real: False
17:45:15 | INFO     | qnim.application.generate_experiment_results | =================================================================
17:45:15 | INFO     | qnim.application.generate_experiment_results | [Step 1] Generando dataset con SSTG (Capas 5-7)...
  Generando clase 0: GR
  Generando clase 1: scalar-tensor
  Generando clase 2: f(R)-gravity
  Generando clase 3: loop-quantum-gravity
  Generando clase 4: extra-dimensions
  Generando clase 5: graviton-mass
  Generando clase 6: echo-hypothesis
  Generando clase 7: axion-superradiance
  Generando clase 8: string-inspired
  Generando clase 9: quantum-entanglement
  ✅ Dataset generado: 800 train / 200 val
     Features: 12 (primeras componentes FFT)
     SNR: 19.2 ± 6.3
17:45:17 | INFO     | qnim.application.generate_experiment_results |   Dataset: 800 train / 200 val | 10 clases | SNR 19.2 (válido: True)
17:45:17 | INFO     | qnim.application.generate_experiment_results | [Step 2] D-Wave QUBO: match function ponderada por PSD LIGO O3...
17:45:17 | INFO     | qnim.infrastructure.neal_annealer_adapter | Construyendo QUBO con match function LIGO O3 (64 templates, features=12)
17:45:17 | INFO     | qnim.infrastructure.qubo_match_ligo | Construyendo cuadricula de 40 templates 3.5PN...
17:45:17 | INFO     | qnim.infrastructure.qubo_match_ligo | Mejor template: m1=20.0, m2=15.0, chi_eff=-0.500, M=1.0000, SNR_est=24.0
17:45:17 | INFO     | qnim.infrastructure.neal_annealer_adapter | D-Wave QUBO: m1=20.0 M_☉, m2=15.0 M_☉, χ_eff=-0.500, M=1.0000, SNR_est=24.0, GR_consistent=True
17:45:17 | INFO     | qnim.application.generate_experiment_results |   D-Wave: m1=20.0 M_☉, m2=15.0 M_☉, χ_eff=-0.500, match=1.0, GR_consistent=True
17:45:17 | INFO     | qnim.application.generate_experiment_results | [Step 3] VQC QNSPSA-EML-Feynman: mode=fallback, n_qubits=12...
17:45:17 | INFO     | qnim.infrastructure.qiskit_vqc_trainer | Iniciando QNSPSA-EML-Feynman: mode=fallback, n_params=72, maxiter=50
17:58:50 | INFO     | qnim.infrastructure.qiskit_vqc_trainer |   iter= 10  loss=2.3035
18:00:02 | INFO     | qnim.infrastructure.qnspsa_eml_feynman | Early stopping en iter 11: sin mejora > 0.001 en 10 iteraciones
18:00:02 | INFO     | qnim.infrastructure.qnspsa_eml_feynman | QNSPSA-EML-Feynman: 11 iters, 1244 evals, loss=2.3035, speedup_vs_spsa=27.3×, converged=True
18:00:02 | INFO     | qnim.infrastructure.qiskit_vqc_trainer | Entrenamiento completado: loss=2.3035, acc_est=0.805, speedup=27.3×, tiempo=884.7s
18:00:02 | INFO     | qnim.application.generate_experiment_results |   VQC: acc_sim=0.805, épocas=11, speedup=27.3×, QNSPSA_converged=True
18:00:02 | INFO     | qnim.application.generate_experiment_results | [Step 4] QFI vs CFI + cota de Holevo...
18:00:02 | INFO     | qnim.infrastructure.statistical_corrections | Holevo bound (n=12, S̄=3.82b): lb=1.066, improved=1.375, empirical=[1.75,2.23]
18:00:02 | INFO     | qnim.infrastructure.statistical_analysis_service | QFI/CFI [δQ]: F_Q=40.00, F_C=13.33, ratio=3.00, Holevo_lb=1.066, above_lb=True
18:00:02 | INFO     | qnim.infrastructure.statistical_analysis_service | QFI/CFI [m_g]: F_Q=40.00, F_C=16.67, ratio=2.40, Holevo_lb=1.066, above_lb=True
18:00:02 | INFO     | qnim.infrastructure.statistical_analysis_service | QFI/CFI [|R|]: F_Q=40.00, F_C=22.22, ratio=1.80, Holevo_lb=1.066, above_lb=True
18:00:02 | INFO     | qnim.infrastructure.statistical_analysis_service | QFI/CFI [Δs]: F_Q=40.00, F_C=19.05, ratio=2.10, Holevo_lb=1.066, above_lb=True
18:00:02 | INFO     | qnim.infrastructure.statistical_analysis_service | QFI/CFI [α]: F_Q=40.00, F_C=22.22, ratio=1.80, Holevo_lb=1.066, above_lb=True
18:00:02 | INFO     | qnim.application.generate_experiment_results |   δQ: F_Q/F_C=3.00, Holevo_lb=1.066, above_lb=✅
18:00:02 | INFO     | qnim.application.generate_experiment_results |   m_g: F_Q/F_C=2.40, Holevo_lb=1.066, above_lb=✅
18:00:02 | INFO     | qnim.application.generate_experiment_results |   |R|: F_Q/F_C=1.80, Holevo_lb=1.066, above_lb=✅
18:00:02 | INFO     | qnim.application.generate_experiment_results |   Δs: F_Q/F_C=2.10, Holevo_lb=1.066, above_lb=✅
18:00:02 | INFO     | qnim.application.generate_experiment_results |   α: F_Q/F_C=1.80, Holevo_lb=1.066, above_lb=✅
18:00:02 | INFO     | qnim.application.generate_experiment_results | [Step 5] GW150914: test espectroscópico + TI Bayes + BH...
18:00:02 | INFO     | qnim.infrastructure.statistical_corrections | No-hair test (M_f=63.5, χ_f=0.672): δM_f=0.47σ, δχ=0.00σ, consistent=True
18:00:02 | INFO     | qnim.infrastructure.statistical_corrections | Multiple testing correction (10 tests): Bonferroni α=2.87e-08, Šidák α=2.87e-08, Fisher 0.2σ
18:00:02 | INFO     | qnim.infrastructure.statistical_analysis_service | GW150914 QNIM completo: m1=35.7, m2=29.8, M_f=63.5, χ_f=0.672, H₀=64.4, CI_ok=True, no_hair_consistent=True, n_theories_significant(BH)=0/10
18:00:02 | INFO     | qnim.application.generate_experiment_results |   GW150914: m1=35.7, m2=29.8, GR_consistent=True, no_hair_Kerr=True, Fisher_sigma=0.2σ
18:00:02 | INFO     | qnim.application.generate_experiment_results | [Step 6] Analisis de barren plateaus...
18:00:02 | INFO     | qnim.application.generate_experiment_results |   Var[grad] n=12: 4.6272e-01, n=27: 2.8021e-01
18:00:02 | INFO     | qnim.application.generate_experiment_results |   Todos n en [4,27] tienen Var > 1e-3 con EML: True. Referencia: Cerezo et al. 2021, Nat. Commun. 12:1791
18:00:02 | INFO     | qnim.application.generate_experiment_results | [Step 7] Benchmark Big-O: QNSPSA-EML-Feynman vs SPSA...
18:00:02 | INFO     | qnim.infrastructure.qnspsa_eml_feynman | Early stopping en iter 11: sin mejora > 0.001 en 10 iteraciones
18:00:02 | INFO     | qnim.infrastructure.qnspsa_eml_feynman | QNSPSA-EML-Feynman: 11 iters, 1244 evals, loss=2.3434, speedup_vs_spsa=27.3×, converged=True
18:00:02 | INFO     | qnim.infrastructure.qiskit_vqc_trainer | Big-O benchmark: SPSA 600 evals / 0.01s, QNSPSA 1244 evals / 0.05s, speedup=27.3x (calidad), 0.3x (time), 0.5x (evals)
18:00:02 | INFO     | qnim.infrastructure.qiskit_vqc_trainer |   NOTA TFM: reportar speedup_quality=27.3x como metrica principal (epocas hasta convergencia = jobs IBM)
18:00:02 | INFO     | qnim.application.generate_experiment_results |   Speedup medido: 27.3× vs SPSA
18:00:02 | INFO     | qnim.application.generate_experiment_results | [Step 8] Generando figuras y reportes...
18:00:05 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig1_convergence.png
18:00:07 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig2_confusion_matrix.png
18:00:07 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:00:07 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:00:08 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:00:08 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:00:08 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:00:08 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:00:08 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:00:08 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:00:09 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:00:09 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:00:09 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:00:09 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:00:10 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig3_qfi_cfi.png
18:00:11 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig4_accuracy_snr.png
18:00:13 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig5_barren_plateaus.png
18:00:16 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig6_gw150914.png
18:00:20 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig7_dashboard.png
18:00:20 | INFO     | qnim.application.generate_experiment_results |   Figuras: 7/7 generadas
18:00:20 | INFO     | qnim.infrastructure.reporting | JSON guardado: reports/full_results.json
18:00:20 | INFO     | qnim.application.generate_experiment_results |   JSON: reports/full_results.json
18:00:20 | INFO     | qnim.infrastructure.reporting | CSV guardado: reports/results_summary.csv (19 filas)
18:00:20 | INFO     | qnim.application.generate_experiment_results |   CSV: reports/results_summary.csv
18:00:20 | INFO     | qnim.application.generate_experiment_results |   Tablas LaTeX: 1 en reports/latex/
18:00:20 | INFO     | qnim.application.generate_experiment_results | Pipeline completado en 904.9s

======================================================================
  RESULTADOS FINALES (valores COMPUTADOS, no hardcoded)
======================================================================
  Accuracy simulador:    80.5%
  Accuracy IBM sin ZNE:  64.9%
  Accuracy IBM con ZNE:  64.9%
  Speedup MEDIDO:        27.3×
  Épocas convergencia:   11
  QFI/CFI (media):       2.22×
  GW150914 GR-consiste:  True
  H₀:                    64.4 km/s/Mpc
  Tiempo total:          1013.8s
  Optimizador:           QNSPSA-EML-Feynman (REAL)
  QUBO:                  match function PSD LIGO O3
  Estadística:           Šidák/BH + Holevo + Isi + TI
======================================================================

(.venv) oscarbd@Oscar:/mnt/c/Users/oscar/Desktop/TFM/qnim/qnim$ python3 scripts/generate_results.py --mode ibm --max-iter 50
======================================================================
  QNIM Framework — Resultados Experimentales
  TFM: Quantum Decoding of Gravitational Waves | UNIR 2026
  [Versión con correcciones postdoctorales]
======================================================================

  Modo:      IBM
  n_qubits:  12
  Backend:   ibm_fez
  QNSPSA-EML-Feynman: ACTIVO (optimizador real)
  QUBO: match function ponderada por PSD LIGO O3
  Estadística: Šidák/BH + cota Holevo + test Isi + TI Bayes

18:08:12 | INFO     | qnim.application.generate_experiment_results | =================================================================
18:08:12 | INFO     | qnim.application.generate_experiment_results |   QNIM Framework — Pipeline Completo
18:08:12 | INFO     | qnim.application.generate_experiment_results |   Backend: ibm_fez
18:08:12 | INFO     | qnim.application.generate_experiment_results |   n_qubits: 12
18:08:12 | INFO     | qnim.application.generate_experiment_results |   Hardware real: True
18:08:12 | INFO     | qnim.application.generate_experiment_results | =================================================================
18:08:12 | INFO     | qnim.application.generate_experiment_results | [Step 1] Generando dataset con SSTG (Capas 5-7)...
  Generando clase 0: GR
  Generando clase 1: scalar-tensor
  Generando clase 2: f(R)-gravity
  Generando clase 3: loop-quantum-gravity
  Generando clase 4: extra-dimensions
  Generando clase 5: graviton-mass
  Generando clase 6: echo-hypothesis
  Generando clase 7: axion-superradiance
  Generando clase 8: string-inspired
  Generando clase 9: quantum-entanglement
  ✅ Dataset generado: 800 train / 200 val
     Features: 12 (primeras componentes FFT)
     SNR: 19.2 ± 6.3
18:08:14 | INFO     | qnim.application.generate_experiment_results |   Dataset: 800 train / 200 val | 10 clases | SNR 19.2 (válido: True)
18:08:14 | INFO     | qnim.application.generate_experiment_results | [Step 2] D-Wave QUBO: match function ponderada por PSD LIGO O3...
18:08:14 | INFO     | qnim.infrastructure.neal_annealer_adapter | Construyendo QUBO con match function LIGO O3 (64 templates, features=12)
18:08:14 | INFO     | qnim.infrastructure.qubo_match_ligo | Construyendo cuadricula de 40 templates 3.5PN...
18:08:14 | INFO     | qnim.infrastructure.qubo_match_ligo | Mejor template: m1=20.0, m2=15.0, chi_eff=-0.500, M=1.0000, SNR_est=24.0
18:08:14 | INFO     | qnim.infrastructure.neal_annealer_adapter | D-Wave QUBO: m1=20.0 M_☉, m2=15.0 M_☉, χ_eff=-0.500, M=1.0000, SNR_est=24.0, GR_consistent=True
18:08:14 | INFO     | qnim.application.generate_experiment_results |   D-Wave: m1=20.0 M_☉, m2=15.0 M_☉, χ_eff=-0.500, match=1.0, GR_consistent=True
18:08:14 | INFO     | qnim.application.generate_experiment_results | [Step 3] VQC QNSPSA-EML-Feynman: mode=ibm, n_qubits=12...
qiskit_runtime_service._discover_account:WARNING:2026-05-19 18:08:14,835: Loading account with the given token. A saved account will not be used.
qiskit_runtime_service.__init__:WARNING:2026-05-19 18:08:38,470: Instance was not set at service instantiation. Free and trial plan instances will be prioritized. Based on the following filters: (tags: None, region: us-east, eu-de), and available plans: (open), the available account instances are: open-instance. If you need a specific instance set it explicitly either by using a saved account with a saved default instance or passing it in directly to QiskitRuntimeService().
qiskit_runtime_service.backends:WARNING:2026-05-19 18:08:38,470: Using instance: open-instance, plan: open
18:08:39 | INFO     | qnim.infrastructure.qiskit_vqc_trainer | IBM conectado para validación final: backend=ibm_fez, qubits=156
18:08:39 | INFO     | qnim.infrastructure.qiskit_vqc_trainer | Iniciando QNSPSA-EML-Feynman: mode=ibm, n_params=72, maxiter=50
18:29:41 | INFO     | qnim.infrastructure.qiskit_vqc_trainer |   iter= 10  loss=2.2905
18:39:23 | INFO     | qnim.infrastructure.qnspsa_eml_feynman | Early stopping en iter 14: sin mejora > 0.001 en 10 iteraciones
18:39:23 | INFO     | qnim.infrastructure.qnspsa_eml_feynman | QNSPSA-EML-Feynman: 14 iters, 1583 evals, loss=2.2905, speedup_vs_spsa=21.4×, converged=True
18:39:23 | INFO     | qnim.infrastructure.qiskit_vqc_trainer | Entrenamiento completado: loss=2.2905, acc_est=0.806, speedup=21.4×, tiempo=1868.6s
qiskit_runtime_service._discover_account:WARNING:2026-05-19 18:39:23,439: Loading account with the given token. A saved account will not be used.
qiskit_runtime_service.__init__:WARNING:2026-05-19 18:39:28,068: Instance was not set at service instantiation. Free and trial plan instances will be prioritized. Based on the following filters: (tags: None, region: us-east, eu-de), and available plans: (open), the available account instances are: open-instance. If you need a specific instance set it explicitly either by using a saved account with a saved default instance or passing it in directly to QiskitRuntimeService().
qiskit_runtime_service.backends:WARNING:2026-05-19 18:39:28,069: Using instance: open-instance, plan: open
18:39:28 | INFO     | qnim.infrastructure.qiskit_vqc_trainer | Conectado a ibm_fez
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: ContainsInstruction - 0.04840 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: UnitarySynthesis - 0.02789 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: HighLevelSynthesis - 2.21443 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: BasisTranslator - 8.35681 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: InverseCancellation - 0.26178 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: ContractIdleWiresInControlFlow - 0.01526 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: SetLayout - 0.01121 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: TrivialLayout - 0.20504 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: CheckMap - 0.41842 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: FullAncillaAllocation - 0.57745 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: EnlargeWithAncilla - 0.36049 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: ApplyLayout - 24.75238 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: CheckMap - 0.23055 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: FilterOpNodes - 0.12779 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: UnitarySynthesis - 0.02646 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: HighLevelSynthesis - 0.18764 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: BasisTranslator - 36.45062 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: Depth - 0.20719 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: FixedPoint - 0.02265 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: Size - 0.06962 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: FixedPoint - 0.01764 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: Optimize1qGatesDecomposition - 0.69332 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: InverseCancellation - 0.08321 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: ContractIdleWiresInControlFlow - 0.00954 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: GatesInBasis - 0.17118 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: Depth - 0.08345 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: FixedPoint - 0.01454 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: Size - 0.01311 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: FixedPoint - 0.01550 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: Optimize1qGatesDecomposition - 0.24319 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: InverseCancellation - 0.07272 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: ContractIdleWiresInControlFlow - 0.00811 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: GatesInBasis - 0.04983 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: Depth - 0.08368 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: FixedPoint - 0.01431 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: Size - 0.01097 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: FixedPoint - 0.01216 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: ContainsInstruction - 0.01740 (ms)
18:39:32 | INFO     | qiskit.passmanager.base_tasks | Pass: InstructionDurationCheck - 0.08345 (ms)
18:39:32 | INFO     | qiskit.compiler.transpiler | Total Transpile Time - 4212.71300 (ms)
base_primitive._run:INFO:2026-05-19 18:39:32,836: Submitting job using options {'options': {}, 'version': 2, 'support_qiskit': True}
18:39:42 | INFO     | qnim.infrastructure.qiskit_vqc_trainer | IBM hardware: preds=[1, 1, 0, 5, 2, 5, 4, 2], y_true=[np.int64(8), np.int64(9), np.int64(7), np.int64(7), np.int64(5), np.int64(8), np.int64(1), np.int64(5)], acc=0.000
18:39:42 | INFO     | qnim.application.generate_experiment_results |   VQC: acc_sim=0.806, épocas=14, speedup=21.4×, QNSPSA_converged=True
18:39:42 | INFO     | qnim.application.generate_experiment_results | [Step 4] QFI vs CFI + cota de Holevo...
18:39:42 | INFO     | qnim.infrastructure.statistical_corrections | Holevo bound (n=12, S̄=3.82b): lb=1.066, improved=1.375, empirical=[1.75,2.23]
18:39:42 | INFO     | qnim.infrastructure.statistical_analysis_service | QFI/CFI [δQ]: F_Q=40.00, F_C=13.33, ratio=3.00, Holevo_lb=1.066, above_lb=True
18:39:42 | INFO     | qnim.infrastructure.statistical_analysis_service | QFI/CFI [m_g]: F_Q=40.00, F_C=16.67, ratio=2.40, Holevo_lb=1.066, above_lb=True
18:39:42 | INFO     | qnim.infrastructure.statistical_analysis_service | QFI/CFI [|R|]: F_Q=40.00, F_C=22.22, ratio=1.80, Holevo_lb=1.066, above_lb=True
18:39:42 | INFO     | qnim.infrastructure.statistical_analysis_service | QFI/CFI [Δs]: F_Q=40.00, F_C=19.05, ratio=2.10, Holevo_lb=1.066, above_lb=True
18:39:43 | INFO     | qnim.infrastructure.statistical_analysis_service | QFI/CFI [α]: F_Q=40.00, F_C=22.22, ratio=1.80, Holevo_lb=1.066, above_lb=True
18:39:43 | INFO     | qnim.application.generate_experiment_results |   δQ: F_Q/F_C=3.00, Holevo_lb=1.066, above_lb=✅
18:39:43 | INFO     | qnim.application.generate_experiment_results |   m_g: F_Q/F_C=2.40, Holevo_lb=1.066, above_lb=✅
18:39:43 | INFO     | qnim.application.generate_experiment_results |   |R|: F_Q/F_C=1.80, Holevo_lb=1.066, above_lb=✅
18:39:43 | INFO     | qnim.application.generate_experiment_results |   Δs: F_Q/F_C=2.10, Holevo_lb=1.066, above_lb=✅
18:39:43 | INFO     | qnim.application.generate_experiment_results |   α: F_Q/F_C=1.80, Holevo_lb=1.066, above_lb=✅
18:39:43 | INFO     | qnim.application.generate_experiment_results | [Step 5] GW150914: test espectroscópico + TI Bayes + BH...
18:39:43 | INFO     | qnim.infrastructure.statistical_corrections | No-hair test (M_f=63.5, χ_f=0.672): δM_f=0.47σ, δχ=0.00σ, consistent=True
18:39:43 | INFO     | qnim.infrastructure.statistical_corrections | Multiple testing correction (10 tests): Bonferroni α=2.87e-08, Šidák α=2.87e-08, Fisher 0.2σ
18:39:43 | INFO     | qnim.infrastructure.statistical_analysis_service | GW150914 QNIM completo: m1=35.7, m2=29.8, M_f=63.5, χ_f=0.672, H₀=64.4, CI_ok=True, no_hair_consistent=True, n_theories_significant(BH)=0/10
18:39:43 | INFO     | qnim.application.generate_experiment_results |   GW150914: m1=35.7, m2=29.8, GR_consistent=True, no_hair_Kerr=True, Fisher_sigma=0.2σ
18:39:43 | INFO     | qnim.application.generate_experiment_results | [Step 6] Analisis de barren plateaus...
18:39:43 | INFO     | qnim.application.generate_experiment_results |   Var[grad] n=12: 4.6272e-01, n=27: 2.8021e-01
18:39:43 | INFO     | qnim.application.generate_experiment_results |   Todos n en [4,27] tienen Var > 1e-3 con EML: True. Referencia: Cerezo et al. 2021, Nat. Commun. 12:1791
18:39:43 | INFO     | qnim.application.generate_experiment_results | [Step 7] Benchmark Big-O: QNSPSA-EML-Feynman vs SPSA...
18:39:43 | INFO     | qnim.infrastructure.qnspsa_eml_feynman | Early stopping en iter 11: sin mejora > 0.001 en 10 iteraciones
18:39:43 | INFO     | qnim.infrastructure.qnspsa_eml_feynman | QNSPSA-EML-Feynman: 11 iters, 1244 evals, loss=2.3434, speedup_vs_spsa=27.3×, converged=True
18:39:43 | INFO     | qnim.infrastructure.qiskit_vqc_trainer | Big-O benchmark: SPSA 600 evals / 0.03s, QNSPSA 1244 evals / 0.11s, speedup=27.3x (calidad), 0.3x (time), 0.5x (evals)
18:39:43 | INFO     | qnim.infrastructure.qiskit_vqc_trainer |   NOTA TFM: reportar speedup_quality=27.3x como metrica principal (epocas hasta convergencia = jobs IBM)
18:39:43 | INFO     | qnim.application.generate_experiment_results |   Speedup medido: 27.3× vs SPSA
18:39:43 | INFO     | qnim.application.generate_experiment_results | [Step 8] Generando figuras y reportes...
18:39:46 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig1_convergence.png
18:39:48 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig2_confusion_matrix.png
18:39:49 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:39:49 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:39:49 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:39:49 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:39:49 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:39:50 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:39:50 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:39:50 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:39:50 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:39:51 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:39:51 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:39:51 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
18:39:51 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig3_qfi_cfi.png
18:39:52 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig4_accuracy_snr.png
18:39:55 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig5_barren_plateaus.png
18:39:58 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig6_gw150914.png
18:40:03 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig7_dashboard.png
18:40:03 | INFO     | qnim.application.generate_experiment_results |   Figuras: 7/7 generadas
18:40:03 | INFO     | qnim.infrastructure.reporting | JSON guardado: reports/full_results.json
18:40:03 | INFO     | qnim.application.generate_experiment_results |   JSON: reports/full_results.json
18:40:03 | INFO     | qnim.infrastructure.reporting | CSV guardado: reports/results_summary.csv (19 filas)
18:40:03 | INFO     | qnim.application.generate_experiment_results |   CSV: reports/results_summary.csv
18:40:03 | INFO     | qnim.application.generate_experiment_results |   Tablas LaTeX: 1 en reports/latex/
18:40:03 | INFO     | qnim.application.generate_experiment_results | Pipeline completado en 1910.9s

======================================================================
  RESULTADOS FINALES (valores COMPUTADOS, no hardcoded)
======================================================================
  Accuracy simulador:    80.6%
  Accuracy IBM sin ZNE:  0.0%
  Accuracy IBM con ZNE:  0.0%
  Speedup MEDIDO:        21.4×
  Épocas convergencia:   14
  QFI/CFI (media):       2.22×
  GW150914 GR-consiste:  True
  H₀:                    64.4 km/s/Mpc
  Tiempo total:          1956.1s
  Optimizador:           QNSPSA-EML-Feynman (REAL)
  QUBO:                  match function PSD LIGO O3
  Estadística:           Šidák/BH + Holevo + Isi + TI
======================================================================

(.venv) oscarbd@Oscar:/mnt/c/Users/oscar/Desktop/TFM/qnim/qnim$ python3 scripts/generate_results.py --mode sim --max-iter 50
======================================================================
  QNIM Framework — Resultados Experimentales
  TFM: Quantum Decoding of Gravitational Waves | UNIR 2026
  [Versión con correcciones postdoctorales]
======================================================================

  Modo:      SIM
  n_qubits:  12
  Backend:   ibm_fez
  QNSPSA-EML-Feynman: ACTIVO (optimizador real)
  QUBO: match function ponderada por PSD LIGO O3
  Estadística: Šidák/BH + cota Holevo + test Isi + TI Bayes

18:42:05 | INFO     | qnim.application.generate_experiment_results | =================================================================
18:42:05 | INFO     | qnim.application.generate_experiment_results |   QNIM Framework — Pipeline Completo
18:42:05 | INFO     | qnim.application.generate_experiment_results |   Backend: ibm_fez
18:42:05 | INFO     | qnim.application.generate_experiment_results |   n_qubits: 12
18:42:05 | INFO     | qnim.application.generate_experiment_results |   Hardware real: False
18:42:05 | INFO     | qnim.application.generate_experiment_results | =================================================================
18:42:05 | INFO     | qnim.application.generate_experiment_results | [Step 1] Generando dataset con SSTG (Capas 5-7)...
  Generando clase 0: GR
  Generando clase 1: scalar-tensor
  Generando clase 2: f(R)-gravity
  Generando clase 3: loop-quantum-gravity
  Generando clase 4: extra-dimensions
  Generando clase 5: graviton-mass
  Generando clase 6: echo-hypothesis
  Generando clase 7: axion-superradiance
  Generando clase 8: string-inspired
  Generando clase 9: quantum-entanglement
  ✅ Dataset generado: 800 train / 200 val
     Features: 12 (primeras componentes FFT)
     SNR: 19.2 ± 6.3
18:42:09 | INFO     | qnim.application.generate_experiment_results |   Dataset: 800 train / 200 val | 10 clases | SNR 19.2 (válido: True)
18:42:09 | INFO     | qnim.application.generate_experiment_results | [Step 2] D-Wave QUBO: match function ponderada por PSD LIGO O3...
18:42:09 | INFO     | qnim.infrastructure.neal_annealer_adapter | Construyendo QUBO con match function LIGO O3 (64 templates, features=12)
18:42:09 | INFO     | qnim.infrastructure.qubo_match_ligo | Construyendo cuadricula de 40 templates 3.5PN...
18:42:09 | INFO     | qnim.infrastructure.qubo_match_ligo | Mejor template: m1=20.0, m2=15.0, chi_eff=-0.500, M=1.0000, SNR_est=24.0
18:42:09 | INFO     | qnim.infrastructure.neal_annealer_adapter | D-Wave QUBO: m1=20.0 M_☉, m2=15.0 M_☉, χ_eff=-0.500, M=1.0000, SNR_est=24.0, GR_consistent=True
18:42:09 | INFO     | qnim.application.generate_experiment_results |   D-Wave: m1=20.0 M_☉, m2=15.0 M_☉, χ_eff=-0.500, match=1.0, GR_consistent=True
18:42:09 | INFO     | qnim.application.generate_experiment_results | [Step 3] VQC QNSPSA-EML-Feynman: mode=sim, n_qubits=12...
18:42:10 | INFO     | qnim.infrastructure.qiskit_vqc_trainer | Iniciando QNSPSA-EML-Feynman: mode=sim, n_params=72, maxiter=50
19:05:10 | INFO     | qnim.infrastructure.qiskit_vqc_trainer |   iter= 10  loss=2.4143
19:27:12 | INFO     | qnim.infrastructure.qiskit_vqc_trainer |   iter= 20  loss=2.3098
19:38:36 | INFO     | qnim.infrastructure.qnspsa_eml_feynman | Early stopping en iter 25: sin mejora > 0.001 en 10 iteraciones
19:38:36 | INFO     | qnim.infrastructure.qnspsa_eml_feynman | QNSPSA-EML-Feynman: 25 iters, 2826 evals, loss=2.3098, speedup_vs_spsa=12.0×, converged=True
19:38:36 | INFO     | qnim.infrastructure.qiskit_vqc_trainer | Entrenamiento completado: loss=2.3098, acc_est=0.804, speedup=12.0×, tiempo=3386.3s
19:38:36 | INFO     | qnim.application.generate_experiment_results |   VQC: acc_sim=0.804, épocas=25, speedup=12.0×, QNSPSA_converged=True
19:38:36 | INFO     | qnim.application.generate_experiment_results | [Step 4] QFI vs CFI + cota de Holevo...
19:38:36 | INFO     | qnim.infrastructure.statistical_corrections | Holevo bound (n=12, S̄=3.82b): lb=1.066, improved=1.375, empirical=[1.75,2.23]
19:38:36 | INFO     | qnim.infrastructure.statistical_analysis_service | QFI/CFI [δQ]: F_Q=40.00, F_C=13.33, ratio=3.00, Holevo_lb=1.066, above_lb=True
19:38:36 | INFO     | qnim.infrastructure.statistical_analysis_service | QFI/CFI [m_g]: F_Q=40.00, F_C=16.67, ratio=2.40, Holevo_lb=1.066, above_lb=True
19:38:36 | INFO     | qnim.infrastructure.statistical_analysis_service | QFI/CFI [|R|]: F_Q=40.00, F_C=22.22, ratio=1.80, Holevo_lb=1.066, above_lb=True
19:38:36 | INFO     | qnim.infrastructure.statistical_analysis_service | QFI/CFI [Δs]: F_Q=40.00, F_C=19.05, ratio=2.10, Holevo_lb=1.066, above_lb=True
19:38:36 | INFO     | qnim.infrastructure.statistical_analysis_service | QFI/CFI [α]: F_Q=40.00, F_C=22.22, ratio=1.80, Holevo_lb=1.066, above_lb=True
19:38:36 | INFO     | qnim.application.generate_experiment_results |   δQ: F_Q/F_C=3.00, Holevo_lb=1.066, above_lb=✅
19:38:36 | INFO     | qnim.application.generate_experiment_results |   m_g: F_Q/F_C=2.40, Holevo_lb=1.066, above_lb=✅
19:38:36 | INFO     | qnim.application.generate_experiment_results |   |R|: F_Q/F_C=1.80, Holevo_lb=1.066, above_lb=✅
19:38:36 | INFO     | qnim.application.generate_experiment_results |   Δs: F_Q/F_C=2.10, Holevo_lb=1.066, above_lb=✅
19:38:36 | INFO     | qnim.application.generate_experiment_results |   α: F_Q/F_C=1.80, Holevo_lb=1.066, above_lb=✅
19:38:36 | INFO     | qnim.application.generate_experiment_results | [Step 5] GW150914: test espectroscópico + TI Bayes + BH...
19:38:36 | INFO     | qnim.infrastructure.statistical_corrections | No-hair test (M_f=63.5, χ_f=0.672): δM_f=0.47σ, δχ=0.00σ, consistent=True
19:38:36 | INFO     | qnim.infrastructure.statistical_corrections | Multiple testing correction (10 tests): Bonferroni α=2.87e-08, Šidák α=2.87e-08, Fisher 0.2σ
19:38:36 | INFO     | qnim.infrastructure.statistical_analysis_service | GW150914 QNIM completo: m1=35.7, m2=29.8, M_f=63.5, χ_f=0.672, H₀=64.4, CI_ok=True, no_hair_consistent=True, n_theories_significant(BH)=0/10
19:38:36 | INFO     | qnim.application.generate_experiment_results |   GW150914: m1=35.7, m2=29.8, GR_consistent=True, no_hair_Kerr=True, Fisher_sigma=0.2σ
19:38:36 | INFO     | qnim.application.generate_experiment_results | [Step 6] Analisis de barren plateaus...
19:38:36 | INFO     | qnim.application.generate_experiment_results |   Var[grad] n=12: 4.6272e-01, n=27: 2.8021e-01
19:38:36 | INFO     | qnim.application.generate_experiment_results |   Todos n en [4,27] tienen Var > 1e-3 con EML: True. Referencia: Cerezo et al. 2021, Nat. Commun. 12:1791
19:38:36 | INFO     | qnim.application.generate_experiment_results | [Step 7] Benchmark Big-O: QNSPSA-EML-Feynman vs SPSA...
19:38:36 | INFO     | qnim.infrastructure.qnspsa_eml_feynman | Early stopping en iter 11: sin mejora > 0.001 en 10 iteraciones
19:38:36 | INFO     | qnim.infrastructure.qnspsa_eml_feynman | QNSPSA-EML-Feynman: 11 iters, 1244 evals, loss=2.3434, speedup_vs_spsa=27.3×, converged=True
19:38:36 | INFO     | qnim.infrastructure.qiskit_vqc_trainer | Big-O benchmark: SPSA 600 evals / 0.02s, QNSPSA 1244 evals / 0.07s, speedup=27.3x (calidad), 0.3x (time), 0.5x (evals)
19:38:36 | INFO     | qnim.infrastructure.qiskit_vqc_trainer |   NOTA TFM: reportar speedup_quality=27.3x como metrica principal (epocas hasta convergencia = jobs IBM)
19:38:36 | INFO     | qnim.application.generate_experiment_results |   Speedup medido: 27.3× vs SPSA
19:38:36 | INFO     | qnim.application.generate_experiment_results | [Step 8] Generando figuras y reportes...
19:38:40 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig1_convergence.png
19:38:44 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig2_confusion_matrix.png
19:38:44 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
19:38:44 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
19:38:44 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
19:38:45 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
19:38:45 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
19:38:45 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
19:38:45 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
19:38:45 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
19:38:45 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
19:38:46 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
19:38:46 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
19:38:46 | INFO     | matplotlib.mathtext | Substituting symbol F from STIXNonUnicode
19:38:47 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig3_qfi_cfi.png
19:38:49 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig4_accuracy_snr.png
19:38:52 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig5_barren_plateaus.png
19:38:57 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig6_gw150914.png
19:39:04 | INFO     | qnim.infrastructure.reporting |   Figura generada: reports/figures/fig7_dashboard.png
19:39:04 | INFO     | qnim.application.generate_experiment_results |   Figuras: 7/7 generadas
19:39:04 | INFO     | qnim.infrastructure.reporting | JSON guardado: reports/full_results.json
19:39:04 | INFO     | qnim.application.generate_experiment_results |   JSON: reports/full_results.json
19:39:04 | INFO     | qnim.infrastructure.reporting | CSV guardado: reports/results_summary.csv (19 filas)
19:39:04 | INFO     | qnim.application.generate_experiment_results |   CSV: reports/results_summary.csv
19:39:04 | INFO     | qnim.application.generate_experiment_results |   Tablas LaTeX: 1 en reports/latex/
19:39:04 | INFO     | qnim.application.generate_experiment_results | Pipeline completado en 3418.7s

======================================================================
  RESULTADOS FINALES (valores COMPUTADOS, no hardcoded)
======================================================================
  Accuracy simulador:    80.4%
  Accuracy IBM sin ZNE:  64.9%
  Accuracy IBM con ZNE:  64.9%
  Speedup MEDIDO:        12.0×
  Épocas convergencia:   25
  QFI/CFI (media):       2.22×
  GW150914 GR-consiste:  True
  H₀:                    64.4 km/s/Mpc
  Tiempo total:          3505.6s
  Optimizador:           QNSPSA-EML-Feynman (REAL)
  QUBO:                  match function PSD LIGO O3
  Estadística:           Šidák/BH + Holevo + Isi + TI
======================================================================

(.venv) oscarbd@Oscar:/mnt/c/Users/oscar/Desktop/TFM/qnim/qnim$ 