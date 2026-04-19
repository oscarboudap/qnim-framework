# Infrastructure Integration Guide

**Purpose:** Show how application layer use cases inject and use infrastructure adapters through hexagonal ports.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Dependency Injection Patterns](#dependency-injection-patterns)
3. [Port Implementations](#port-implementations)
4. [Integration Examples](#integration-examples)
5. [Testing Strategies](#testing-strategies)

---

## Architecture Overview

### Hexagonal Architecture Design

```
┌─────────────────────────────────────────────────┐
│         APPLICATION LAYER (use cases)            │
│                                                 │
│  HybridOrchestrator, ModelTrainingUseCase, etc │
└─────────────────────────────────────────────────┘
              ↓ Depends on (injects)
┌─────────────────────────────────────────────────┐
│        APPLICATION PORTS (abstract)              │
│    IQuantumOptimizerPort, IQuantumMLTrainerPort │
│    IPreprocessingPort, IMetricsReporterPort,    │
│    IDataLoaderPort, IStoragePort, etc.          │
└─────────────────────────────────────────────────┘
              ↓ Implemented by
┌─────────────────────────────────────────────────┐
│     INFRASTRUCTURE ADAPTERS (concrete)           │
│    NealSimulatedAnnealerAdapter, QiskitVQC      │
│    Trainer, SklearnPreprocessor, etc.           │
└─────────────────────────────────────────────────┘
              ↓ Wraps
┌─────────────────────────────────────────────────┐
│       EXTERNAL FRAMEWORKS & LIBRARIES            │
│  Qiskit, D-Wave Neal, sklearn, matplotlib, etc. │
└─────────────────────────────────────────────────┘
```

### Key Principles

1. **No Framework Imports in Application:** Application never imports Qiskit, sklearn, matplotlib
2. **All Dependencies Flow Inward:** Domain ← App ← Infrastructure ← Frameworks
3. **Type Safety at Boundaries:** All port methods return typed DTOs/VOs, not dicts
4. **Single Adapter Per Framework:** All Qiskit code lives in QiskitVQCTrainer bucket

---

## Dependency Injection Patterns

### Pattern 1: Constructor Injection

```python
# ❌ BEFORE (coupled to infrastructure)
from qiskit.circuit.library import ZZFeatureMap
class ModelTrainingService:
    def train(self, X, y):
        feature_map = ZZFeatureMap(12)  # Qiskit import here!
        ...

# ✅ AFTER (decoupled via port)
from src.application.ports import IQuantumMLTrainerPort

class ModelTrainingUseCase:
    def __init__(self, trainer_port: IQuantumMLTrainerPort):
        self.trainer = trainer_port  # Abstract port
    
    def train(self, X, y) -> TrainingMetrics:
        return self.trainer.train_vqc(X, y, 12, 100)
        # Infrastructure adapter (QiskitVQCTrainer) provides impl
```

### Pattern 2: Setter Injection (Method Parameter)

```python
# For one-off operations
class DecodeGravitationalWaveUseCase:
    def execute(self,
                event: QuantumDecodedEvent,
                dataloader_port: IDataLoaderPort,
                optimizer_port: IQuantumOptimizerPort) -> InferenceResult:
        
        # Use ports without knowing implementation
        strain = dataloader_port.prepare_for_quantum(event.file_path)
        parameters = optimizer_port.solve_qubo(Q_matrix, num_reads=1000)
        ...
        return InferenceResult(...)
```

### Pattern 3: Factory Injection

```python
# For complex initialization
from src.infrastructure import (
    QiskitVQCTrainer,
    SklearnPreprocessor,
    MatplotlibMetricsReporter,
)

class PortFactory:
    @staticmethod
    def create_training_pipeline():
        return (
            QiskitVQCTrainer(),
            SklearnPreprocessor(),
            NealSimulatedAnnealerAdapter()
        )

# Usage in entry point (main.py)
trainer, preprocessor, optimizer = PortFactory.create_training_pipeline()
use_case = ModelTrainingUseCase(trainer, preprocessor, optimizer)
```

---

## Port Implementations

### 1. IQuantumMLTrainerPort → QiskitVQCTrainer

**Contract:**
```python
class IQuantumMLTrainerPort(ABC):
    @abstractmethod
    def train_vqc(self,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  num_qubits: int,
                  max_iterations: int = 100,
                  optimizer_name: str = "SPSA") -> Dict[str, any]:
        """Returns dict with: weights, training_loss, validation_accuracy, iterations, execution_time_seconds"""
```

**Implementation:**
```python
class QiskitVQCTrainer(IQuantumMLTrainerPort):
    def train_vqc(self, X_train, y_train, num_qubits, max_iterations="SPSA") → Dict:
        # All Qiskit logic here (encapsulated)
        feature_map = ZZFeatureMap(num_qubits, reps=2)
        ansatz = RealAmplitudes(num_qubits, reps=3)
        optimizer = SPSA(maxiter=max_iterations, learning_rate=0.05)
        vqc = VQC(feature_map, ansatz, optimizer, ...)
        vqc.fit(X_train, y_train)
        return {
            "weights": vqc.weights,
            "training_loss": ...,
            ...
        }
```

**Usage:**
```python
# In application layer (no Qiskit imported)
class ModelTrainingUseCase:
    def __init__(self, trainer_port: IQuantumMLTrainerPort):
        self.trainer = trainer_port
    
    def execute(self, training_data: Dataset) -> TrainingMetrics:
        result = self.trainer.train_vqc(
            X_train=training_data.X,
            y_train=training_data.y,
            num_qubits=12,
            max_iterations=100
        )
        
        return TrainingMetrics(
            loss=result["training_loss"],
            accuracy=result["validation_accuracy"],
            iterations=result["iterations"],
            execution_time=result["execution_time_seconds"]
        )
```

### 2. IPreprocessingPort → SklearnPreprocessor

**Contract:**
```python
class IPreprocessingPort(ABC):
    @abstractmethod
    def fit_transform(self, X: np.ndarray) → np.ndarray:
        """Fit pipeline on X and return transformed X"""
    
    @abstractmethod
    def transform(self, X: np.ndarray) → np.ndarray:
        """Apply fitted pipeline to X"""
    
    @abstractmethod
    def save(self, path: str) → None:
        """Save fitted pipeline to disk"""
    
    @abstractmethod
    def load(self, path: str) → None:
        """Load fitted pipeline from disk"""
```

**Implementation:**
```python
class SklearnPreprocessor(IPreprocessingPort):
    N_COMPONENTS = 12
    MIN_VALUE = -np.pi
    MAX_VALUE = np.pi
    
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=self.N_COMPONENTS)),
            ('minmax', MinMaxScaler(feature_range=(self.MIN_VALUE, self.MAX_VALUE)))
        ])
        self.is_fitted = False
    
    def fit_transform(self, X: np.ndarray) → np.ndarray:
        # StandardScaler + PCA + MinMaxScaler
        X_transformed = self.pipeline.fit_transform(X)
        self.is_fitted = True
        return X_transformed
    
    def transform(self, X: np.ndarray) → np.ndarray:
        return self.pipeline.transform(X)
    
    def save(self, path: str) → None:
        joblib.dump(self.pipeline, path)
    
    def load(self, path: str) → None:
        self.pipeline = joblib.load(path)
        self.is_fitted = True
```

**Usage:**
```python
# In ModelTrainingUseCase
def execute(self, data: RawDataset) -> TrainingMetrics:
    # Preprocess via port (don't know it's sklearn internally)
    X_clean = self.preprocessor.fit_transform(data.X)
    
    # Train VQC
    train_result = self.trainer.train_vqc(X_clean, data.y, 12, 100)
    
    # Save preprocessor for later inference
    self.preprocessor.save("models/preprocessing_pipeline.pkl")
    
    return TrainingMetrics(...)
```

### 3. IMetricsReporterPort → MatplotlibMetricsReporter

**Contract:**
```python
class IMetricsReporterPort(ABC):
    @abstractmethod
    def report_confusion_matrix(self,
                               cm_data: ConfusionMatrixData,
                               output_filename: str = None) → str:
        """Generate confusion matrix heatmap, return path to .png"""
    
    @abstractmethod
    def report_inference_trace(self,
                              event_id: str,
                              classic_parameters: dict,
                              quantum_results: dict,
                              execution_time_seconds: float,
                              output_filename: str = None) → str:
        """Generate inference report, return path to .md"""
```

**Implementation:**
```python
class MatplotlibMetricsReporter(IMetricsReporterPort):
    def report_confusion_matrix(self, cm_data, output_filename=None) → str:
        # All matplotlib code here
        matrix = np.array([[cm_data.tp, cm_data.fp],
                          [cm_data.fn, cm_data.tn]])
        fig, ax = plt.subplots()
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        
        path = self.output_dir / output_filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return str(path)
    
    def report_inference_trace(self, event_id, classic_params, quantum_results, time) → str:
        # Generate markdown report
        markdown = f"""
# Inference Report
## Event {event_id}
**Time:** {time:.3f}s

**Classical Parameters:** {classic_params}
**Quantum Results:** {quantum_results}
"""
        path = self.output_dir / f"trace_{event_id}.md"
        path.write_text(markdown)
        return str(path)
```

**Usage:**
```python
# In ModelValidationUseCase
class ModelValidationUseCase:
    def __init__(self, reporter_port: IMetricsReporterPort):
        self.reporter = reporter_port
    
    def execute(self, validation_data: Dataset) -> ConfusionMatrixData:
        # Get predictions
        predictions = self.model.predict(validation_data.X)
        
        # Build confusion matrix data
        cm_data = ConfusionMatrixData(
            tp=int(np.sum((predictions == 1) & (validation_data.y == 1))),
            tn=int(np.sum((predictions == 0) & (validation_data.y == 0))),
            fp=int(np.sum((predictions == 1) & (validation_data.y == 0))),
            fn=int(np.sum((predictions == 0) & (validation_data.y == 1)))
        )
        
        # Generate report (matplotlib encapsulated)
        png_path = self.reporter.report_confusion_matrix(cm_data)
        print(f"📊 Confusion matrix: {png_path}")
        
        return cm_data
```

### 4. IDataLoaderPort → QuantumDatasetLoader

**Contract:**
```python
class IDataLoaderPort(ABC):
    @abstractmethod
    def prepare_for_quantum(self, file_path: str, is_real_data: bool = False) → np.ndarray:
        """Load + preprocess data, return [16384] array for VQC"""
```

**Implementation:**
```python
class QuantumDatasetLoader(IDataLoaderPort):
    def prepare_for_quantum(self, file_path: str, is_real_data=False) → np.ndarray:
        with h5py.File(file_path, 'r') as f:
            strain = f['strain'][:]
        
        if is_real_data:
            strain = self._apply_ligo_whitening(strain)
            strain = self._bandpass_filter(strain, 35, 350)
        
        strain = self._align_merger(strain)
        strain = self._normalize(strain)
        
        return strain  # [16384] for VQC
    
    def _apply_ligo_whitening(self, data) → np.ndarray:
        # LIGO whitening algorithm
        ...
    
    def _bandpass_filter(self, data, low, high) → np.ndarray:
        # Bandpass filtering
        ...
```

**Usage:**
```python
class DecodeGravitationalWaveUseCase:
    def __init__(self, dataloader_port: IDataLoaderPort):
        self.dataloader = dataloader_port
    
    def execute(self, event: QuantumDecodedEvent) → InferenceResult:
        # Load data via port
        strain = self.dataloader.prepare_for_quantum(
            event.file_path,
            is_real_data=event.is_real_ligo_data
        )
        
        # Use strain for inference
        ...
        return InferenceResult(...)
```

---

## Integration Examples

### Full Training Pipeline

```python
# main.py (Entry point)
from src.infrastructure import (
    QiskitVQCTrainer,
    SklearnPreprocessor,
    NealSimulatedAnnealerAdapter,
    QuantumDatasetLoader,
)
from src.application import ModelTrainingUseCase

# 1. Instantiate adapters
trainer = QiskitVQCTrainer()
preprocessor = SklearnPreprocessor()
dataloader = QuantumDatasetLoader()

# 2. Inject into use case (all dependencies via ports)
training_use_case = ModelTrainingUseCase(
    trainer_port=trainer,
    preprocessor_port=preprocessor,
    dataloader_port=dataloader
)

# 3. Execute (no knowledge of frameworks)
training_data = Dataset.load("data/training.csv")
metrics = training_use_case.execute(training_data)

print(f"✅ Training complete: {metrics.accuracy}")
```

### Inference Pipeline

```python
# scripts/inference.py
from src.infrastructure import (
    QuantumDatasetLoader,
    QiskitVQCTrainer,
    MatplotlibMetricsReporter,
    NealSimulatedAnnealerAdapter,
)
from src.application import (
    HybridOrchestrator,
    DecodeGravitationalWaveUseCase,
)

# Setup adapters
dataloader = QuantumDatasetLoader()
trainer = QiskitVQCTrainer()
trainer.load_weights("models/weights.npy")
optimizer = NealSimulatedAnnealerAdapter()
reporter = MatplotlibMetricsReporter()

# Setup use cases
hybrid_orchestrator = HybridOrchestrator(
    dwave_optimizer=optimizer,
    vqc_trainer=trainer,
    dataloader=dataloader,
    reporter=reporter
)

# Inference
event = QuantumDecodedEvent(file_path="data/event_001.h5")
result = hybrid_orchestrator.execute(event)

# Report
print(f"🎯 Theory: {result.classification}")
print(f"📊 Confidence: {result.confidence}")
```

---

## Testing Strategies

### 1. Mock Adapters for Application Tests

```python
# tests/test_model_training.py
from unittest.mock import Mock
from src.application import ModelTrainingUseCase
from src.application.dto import TrainingMetrics

class MockQiskitTrainer:
    def train_vqc(self, X, y, num_qubits, max_iterations):
        return {
            "weights": np.zeros(10),
            "training_loss": 0.01,
            "validation_accuracy": 0.99,
            "iterations": 100,
            "execution_time_seconds": 5.0
        }

class MockSklearnPreprocessor:
    def fit_transform(self, X):
        return np.random.randn(X.shape[0], 12)
    def save(self, path):
        pass

# Test without any real framework
def test_model_training():
    use_case = ModelTrainingUseCase(
        trainer_port=MockQiskitTrainer(),
        preprocessor_port=MockSklearnPreprocessor()
    )
    
    dataset = Dataset(X=np.random.randn(100, 16384), y=np.random.binomial(1, 0.5, 100))
    result = use_case.execute(dataset)
    
    assert result.accuracy == 0.99
    assert result.iterations == 100
```

### 2. Integration Tests with Real Adapters

```python
# tests/test_infrastructure_integration.py
import pytest
from src.infrastructure import QiskitVQCTrainer, SklearnPreprocessor

@pytest.fixture
def small_dataset():
    X = np.random.randn(10, 16384)
    y = np.random.binomial(1, 0.5, 10)
    return X, y

def test_sklearn_preprocessing_pipeline(small_dataset):
    X, y = small_dataset
    
    preprocessor = SklearnPreprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    # Verify output shape
    assert X_transformed.shape == (10, 12)  # N_COMPONENTS=12
    
    # Verify range [-π, π]
    assert np.min(X_transformed) >= -np.pi
    assert np.max(X_transformed) <= np.pi
    
    # Verify transform() works
    X_test = np.random.randn(5, 16384)
    X_test_transformed = preprocessor.transform(X_test)
    assert X_test_transformed.shape == (5, 12)

def test_trainer_save_load():
    trainer = QiskitVQCTrainer()
    
    weights = np.random.randn(10, 5)
    trainer.save_weights(weights, "test_weights.npy")
    
    loaded = trainer.load_weights("test_weights.npy")
    
    assert np.allclose(weights, loaded)
```

### 3. Contract Testing (Port Compliance)

```python
# tests/test_port_contracts.py
from abc import ABC, abstractmethod
from src.application.ports import IQuantumMLTrainerPort
from src.infrastructure import QiskitVQCTrainer

class PortContractTest:
    """Verify adapter implements port contract correctly"""
    
    def test_quantum_ml_trainer_port_contract(self):
        # Adapter must implement port
        assert isinstance(QiskitVQCTrainer(), IQuantumMLTrainerPort)
        
        # Methods must exist and be callable
        trainer = QiskitVQCTrainer()
        assert hasattr(trainer, 'train_vqc')
        assert callable(trainer.train_vqc)
        assert hasattr(trainer, 'save_weights')
        assert hasattr(trainer, 'load_weights')
        
        # Return types must be correct
        result = trainer.train_vqc(
            X_train=np.random.randn(5, 16384),
            y_train=np.random.binomial(1, 0.5, 5),
            num_qubits=12
        )
        assert isinstance(result, dict)
        assert 'weights' in result
        assert 'training_loss' in result
        assert 'validation_accuracy' in result
```

---

## Error Handling Strategy

### Exception Flow

```
User Input Error → ValueError/TypeError (validation)
                  ↓
Application Layer → ApplicationException (business logic)
                  ↓
Infrastructure Layer → (Type-specific)
                   ├─ TrainingException (QiskitVQCTrainer)
                   ├─ PreprocessingException (SklearnPreprocessor)
                   ├─ DataLoaderException (QuantumDatasetLoader)
                   ├─ StorageException (HDF5Exporter)
                   └─ QuantumComputeException (Neal/IBM adapters)
                  ↓
External Frameworks → (Qiskit, sklearn, h5py, etc.)
```

### Usage

```python
from src.infrastructure.exceptions import TrainingException, PreprocessingException

class ModelTrainingUseCase:
    def execute(self, data):
        try:
            X_clean = self.preprocessor.fit_transform(data.X)
        except PreprocessingException as e:
            raise ApplicationException(f"Preprocessing failed: {str(e)}")
        
        try:
            result = self.trainer.train_vqc(X_clean, data.y, 12)
        except TrainingException as e:
            raise ApplicationException(f"Training failed: {str(e)}")
        
        return TrainingMetrics(...)
```

---

## Performance Considerations

### Caching Preprocessed Data
```python
class ModelTrainingUseCase:
    def __init__(self, preprocessor_port, trainer_port):
        self.preprocessor = preprocessor_port
        self.trainer = trainer_port
        self._preproc_cache = {}
    
    def execute(self, dataset_id, data):
        # Avoid re-preprocessing
        if dataset_id not in self._preproc_cache:
            X_clean = self.preprocessor.fit_transform(data.X)
            self._preproc_cache[dataset_id] = X_clean
        else:
            X_clean = self._preproc_cache[dataset_id]
        
        return self.trainer.train_vqc(X_clean, data.y, 12)
```

### Lazy Loading Infrastructure
```python
class SSTGAdapter:
    def __init__(self):
        self._engine = None  # Lazy-loaded
    
    def _get_engine(self):
        if self._engine is None:
            from domain.astrophysics.sstg import QuantumUniverseEngine
            self._engine = QuantumUniverseEngine()
        return self._engine
    
    def synthesize_event(self, m1, m2, distance, theory):
        engine = self._get_engine()
        return engine.generate_signal(m1, m2, distance, theory)
```

---

## Deployment Checklist

- [ ] All ports properly injected into use cases
- [ ] No framework imports in application layer
- [ ] All adapters 100% type hints
- [ ] All adapters use custom exceptions
- [ ] Mock adapters created for unit tests
- [ ] Integration tests verify real adapters
- [ ] Contract tests verify port compliance
- [ ] Performance benchmarks for preprocessing + training
- [ ] Error handling tested (happy + sad paths)
- [ ] Documentation updated with new patterns

---

*This guide provides foundation for extending infrastructure layer with new adapters and ensuring application layer maintains hexagonal architecture.*
