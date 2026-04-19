"""
REPOSITORIOS Y PERSISTENCIA DE DOMINIO.

Define las interfaces para persistencia de eventos GW decodificados,
respetando los principios de Domain-Driven Design (agregados, especificaciones).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Set
from .entities import QuantumDecodedEvent, GWEventSpecification
from .value_objects import TheoryFamily


class GravitationalWaveRepository(ABC):
    """
    Repositorio de eventos GW decodificados.
    
    Abstrae la persistencia y recuperación de QuantumDecodedEvent,
    permitiendo consultas complejas respetando invariantes de dominio.
    """
    
    @abstractmethod
    def add(self, event: QuantumDecodedEvent) -> None:
        """
        Añade un nuevo evento al repositorio.
        
        Raises:
            ValueError: si el event_id ya existe o violenta invariantes
        """
        pass
    
    @abstractmethod
    def remove(self, event_id: str) -> None:
        """Elimina un evento por su ID."""
        pass
    
    @abstractmethod
    def find_by_id(self, event_id: str) -> Optional[QuantumDecodedEvent]:
        """Busca un evento por ID exacto."""
        pass
    
    @abstractmethod
    def find_all_matching(self, spec: GWEventSpecification) -> List[QuantumDecodedEvent]:
        """
        Busca eventos que cumplen la especificación.
        
        Usa patrón Specification para consultas type-safe.
        """
        pass
    
    @abstractmethod
    def count_by_theory(self, theory: TheoryFamily) -> int:
        """Cuenta eventos cuya teoría preferida es la especificada."""
        pass
    
    @abstractmethod
    def get_all_events(self) -> List[QuantumDecodedEvent]:
        """Retorna todos los eventos (use sparingly)."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, float]:
        """
        Retorna estadísticas agregadas del catálogo.
        
        Returns:
            {
                "total_events": int,
                "mean_snr": float,
                "mean_chirp_mass": float,
                "fraction_beyond_gr": float,
                ...
            }
        """
        pass


class UnitOfWork(ABC):
    """
    UnitOfWork pattern para manejar transaccionalidad.
    
    Coordina cambios en múltiples agregados y garantiza
    consistencia al persistir.
    """
    
    @abstractmethod
    def __enter__(self):
        pass
    
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    @abstractmethod
    def commit(self) -> None:
        """Persiste todos los cambios pendientes."""
        pass
    
    @abstractmethod
    def rollback(self) -> None:
        """Revierte cambios pendientes."""
        pass
    
    @property
    @abstractmethod
    def gw_events(self) -> GravitationalWaveRepository:
        """Acceso al repositorio de eventos."""
        pass


class LayerRepository(ABC):
    """
    Repositorio por capa (abstracción adicional para escalabilidad).
    """
    
    layer_number: int  # 1-7
    
    @abstractmethod
    def find_events_with_populated_layer(self) -> List[QuantumDecodedEvent]:
        """Busca eventos que tienen esta capa poblada y significativa."""
        pass
    
    @abstractmethod
    def get_layer_statistics(self) -> Dict[str, float]:
        """Estadísticas de qué tan bien se infieren los observables de la capa."""
        pass


# ============================================================================
# ESPECIFICACIONES DE CONSULTA (Specification Pattern)
# ============================================================================

class EventSpecification(ABC):
    """
    Especificación abstracta para filtrado de eventos.
    """
    
    @abstractmethod
    def is_satisfied_by(self, event: QuantumDecodedEvent) -> bool:
        pass


class MinimumSNRSpecification(EventSpecification):
    """Especificación: SNR >= threshold."""
    
    def __init__(self, min_snr: float):
        self.min_snr = min_snr
    
    def is_satisfied_by(self, event: QuantumDecodedEvent) -> bool:
        return event.snr_total.signal_to_noise() >= self.min_snr


class BeyondGRConfidenceSpecification(EventSpecification):
    """Especificación: probabilidad de Beyond-GR >= threshold."""
    
    def __init__(self, min_confidence: float):
        self.min_confidence = min_confidence
    
    def is_satisfied_by(self, event: QuantumDecodedEvent) -> bool:
        if not event.beyond_gr:
            return False
        return event.beyond_gr.beyond_gr_confidence.value >= self.min_confidence


class HasEchoesSpecification(EventSpecification):
    """Especificación: tiene ecos detectados (firma de ECO)."""
    
    def is_satisfied_by(self, event: QuantumDecodedEvent) -> bool:
        if not event.horizon_topology or not event.horizon_topology.echo_spectroscopy:
            return False
        return len(event.horizon_topology.echo_spectroscopy.echo_pattern) > 0


class TheoryFamilySpecification(EventSpecification):
    """Especificación: teoría preferida está en conjunto."""
    
    def __init__(self, theories: Set[TheoryFamily]):
        self.theories = theories
    
    def is_satisfied_by(self, event: QuantumDecodedEvent) -> bool:
        return event.inferred_theory in self.theories


class CompositeSpecification(EventSpecification):
    """
    Especificación compuesta con lógica AND/OR.
    """
    
    def __init__(self, specs: List[EventSpecification], logic: str = "AND"):
        self.specs = specs
        self.logic = logic.upper()
    
    def is_satisfied_by(self, event: QuantumDecodedEvent) -> bool:
        if self.logic == "AND":
            return all(spec.is_satisfied_by(event) for spec in self.specs)
        elif self.logic == "OR":
            return any(spec.is_satisfied_by(event) for spec in self.specs)
        else:
            raise ValueError(f"Lógica desconocida: {self.logic}")


# ============================================================================
# FACTORY PARA REPOSITORIOS
# ============================================================================

class RepositoryFactory:
    """
    Factory para crear instancias apropiadas de repositorios.
    Permite cambiar backend (SQL, MongoDB, JSON) sin afectar la lógica.
    """
    
    _backends: Dict[str, type] = {}
    
    @classmethod
    def register_backend(cls, name: str, backend_class: type) -> None:
        """Registra un nuevo tipo de backend."""
        cls._backends[name] = backend_class
    
    @classmethod
    def create_repository(cls, backend: str = "memory") -> GravitationalWaveRepository:
        """
        Crea un repositorio del tipo especificado.
        
        Args:
            backend: "memory", "sql", "mongodb", "json", etc.
        """
        if backend not in cls._backends:
            raise ValueError(f"Backend desconocido: {backend}. Registrados: {list(cls._backends.keys())}")
        
        return cls._backends[backend]()
    
    @classmethod
    def create_unit_of_work(cls, backend: str = "memory") -> UnitOfWork:
        """Crea un UnitOfWork con el backend especificado."""
        # Placeholder: en infraestructura se implementan backends concretos
        pass
