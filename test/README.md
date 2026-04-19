# QNIM Physics Layers - Real Testing Suite

## Structure Overview

```
test/
├── __init__.py                    # Test package
├── conftest.py                    # PyTest fixtures & configuration
├── sanity_check.py                # Quick validation script (no pytest)
├── pytest.ini                     # PyTest settings
├── unit/
│   ├── __init__.py
│   └── test_physics_layers.py     # Unit tests for Layers 5, 6, 7
└── integration/
    ├── __init__.py
    └── test_generator_integration.py # End-to-end generator tests
```

## Running Tests

### Option 1: Quick Sanity Check (No PyTest Required)
```bash
# Run fast validation of all 3 layers
cd qnim
python3 test/sanity_check.py
```

Output shows:
- Layer 5 (Brans-Dicke, ADD) ✅
- Layer 6 (ECO, LQG) ✅  
- Layer 7 (Hawking, AdS/CFT) ✅
- Parameter signatures verified

### Option 2: Full Unit Test Suite
```bash
# Run all unit tests with pytest
pytest test/unit/ -v

# Run specific test class
pytest test/unit/test_physics_layers.py::TestLayer5BeyondGR -v

# Run with markers
pytest test/unit/ -m "not slow" -v
```

### Option 3: Integration Tests (Full Pipeline)
```bash
# Run generator integration tests
pytest test/integration/ -v

# Test all theories in generator
pytest test/integration/test_generator_integration.py::TestGeneratorIntegration::test_all_theories_produce_valid_output -v
```

### Option 4: Run All Tests
```bash
# Full test suite
pytest test/ -v

# With coverage report
pytest test/ --cov=src --cov-report=html
```

## Test Structure Details

### Unit Tests (`test/unit/test_physics_layers.py`)

**TestLayer5BeyondGR**
- ✅ `test_brans_dicke_injection()` - Scalar-tensor with dipolar radiation
- ✅ `test_extra_dimensions_injection()` - ADD energy leakage
- ✅ `test_lorentz_violation_injection()` - Birefringence effects

**TestLayer6HorizonTopology**
- ✅ `test_eco_echoes_injection()` - Exotic compact object echoes
- ✅ `test_lqg_discrete_spectrum()` - Loop quantum gravity modes
- ✅ `test_gravitational_memory_injection()` - Permanent displacements

**TestLayer7QuantumCorrections**
- ✅ `test_hawking_evaporation()` - Radiation spectrum
- ✅ `test_ads_cft_viscosity()` - Dissipation via duality
- ✅ `test_firewall_correction()` - Singularity distortions

**TestPhysicsSignatureDifferences**
- ✅ Verifies Layer 5 theories produce distinguishable outputs
- ✅ Verifies Layer 6 theories produce different signatures
- ✅ Cross-correlation < 0.99 for different theories

**TestOutputConsistency**
- ✅ All layers return `h_plus` and `h_cross`
- ✅ Output shapes preserved
- ✅ No NaN/Inf values

### Integration Tests (`test/integration/test_generator_integration.py`)

**TestGeneratorIntegration**
- ✅ `test_generator_kerr_vacuum()` - GR baseline
- ✅ `test_generator_scalar_tensor()` - Beyond-GR Layer 5
- ✅ `test_generator_eco_theory()` - Horizon effects Layer 6
- ✅ `test_generator_hawking_radiation()` - Quantum Layer 7
- ✅ `test_all_theories_produce_valid_output()` - All 11+ theories

**TestSignatureDistinguishability**
- ✅ Layer 5 theories differ (correlation < 0.99)
- ✅ Layer 6 theories differ (RMS difference > 0)
- ✅ Consistency within theory

**TestEdgeCases**
- ✅ NSBH mass ratios (q=0.01)
- ✅ Equal mass mergers (q=1.0)
- ✅ High redshift sources (z~0.5)

## Parameter Signatures - CRITICAL

### Layer 5: Beyond-GR Physics
```python
output = Layer5BeyondGRInjector.apply_beyond_gr_physics(
    h_plus: ndarray,
    h_cross: ndarray,
    params: BeyondGRParams,
    total_mass_msun: float,    # ✓ NOT 'mass'
    distance_mpc: float,        # ✓ Required parameter
    fs: int                      # ✓ Convert to int
)
# Returns: {"h_plus", "h_cross", "h_scalar"?, "n_extra_dims"?}
```

### Layer 6: Horizon Topology
```python
output = Layer6HorizonTopologyInjector.apply_horizon_topology(
    h_plus: ndarray,
    h_cross: ndarray,
    params: HorizonTopologyParams,
    mass: float,                 # ✓ 'mass' NOT 'total_mass_msun'
    fs: float                    # ✓ Keep as float
)
# Returns: {"h_plus", "h_cross"}
```

### Layer 7: Quantum Corrections
```python
output = Layer7QuantumCorrectionsInjector.apply_quantum_corrections(
    h_plus: ndarray,
    h_cross: ndarray,
    params: QuantumCorrectionParams,
    mass: float,                 # ✓ 'mass' NOT 'total_mass_msun'
    fs: float                    # ✓ Keep as float
)
# Returns: {"h_plus", "h_cross", "physics_applied", "metadata"}
```

## Expected Test Results

### Passing Criteria
- ✅ All 8 unit test class methods PASS
- ✅ All 3 integration test classes PASS
- ✅ No NaN/Inf in outputs
- ✅ Signatures distinguishable (correlations < 0.99)
- ✅ Energy conservation (no 0 signals)

### Common Issues Fixed
1. ❌→✅ `total_mass_msun` parameter for Layer 6/7 (use `mass` instead)
2. ❌→✅ Dataclass field ordering in layers.py
3. ❌→✅ Output dictionary key expectations

## Fixtures Available

From `conftest.py`:

```python
@pytest.fixture
def sample_strain():
    """1024-sample GW strain at 16384 Hz"""
    return {
        "h_plus": ndarray(1024,),
        "h_cross": ndarray(1024,),
        "fs": 16384.0,
        ...
    }

@pytest.fixture
def mass_parameters():
    """GW150914-like parameters"""
    return {
        "total_mass_msun": 65.0,
        "mass_ratio": 1.0,
        "distance_mpc": 410.0,
        ...
    }
```

## Next Steps

After tests pass:
1. ✅ Generate synthetic data with all 11+ theories
2. ✅ Train classifier on diverse signatures
3. ✅ Create thesis plots and figures
4. ✅ Final validation before defense (Apr 22)

## Documentation

- **Implementation details**: `IMPLEMENTATION_LAYERS_5_6_7.md`
- **Physics references**: In-code docstrings with citations
- **Generator integration**: `generator.py` dispatcher logic
