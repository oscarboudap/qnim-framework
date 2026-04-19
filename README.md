# QNIM - Quantum Neuro-Inspired Manifold
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
