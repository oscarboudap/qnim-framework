# src/domain/astrophysics/layers.py
from dataclasses import dataclass, field
from typing import Optional
from .value_objects import TheoryFamily, SolarMass

# --- CAPA 2: GEOMETRÍA INTRÍNSECA ---
@dataclass
class IntrinsicGeometry:
    m1: SolarMass
    m2: SolarMass
    chirp_mass: SolarMass
    effective_spin_chi: float
    orbital_eccentricity: float = 0.0
    tidal_deformability_lambda: Optional[float] = None # Solo si hay Estrellas de Neutrones

# --- CAPA 4: COSMOLOGÍA ---
@dataclass
class CosmologicalFootprint:
    luminosity_distance_mpc: float
    redshift_z: float
    hubble_constant_h0: Optional[float] = None # Inferido si actúa como Sirena Estándar

# --- CAPA 5: MÁS ALLÁ DEL GR (ANOMALÍAS DE PROPAGACIÓN) ---
@dataclass
class BeyondGRDeviations:
    graviton_mass_limit_ev: Optional[float] = None # Dispersion límite
    extra_dimensions_leakage: float = 0.0 # Atenuación anómala por dimensiones Kaluza-Klein
    scalar_tensor_polarization: bool = False # Detección de modo respiración/longitudinal

# --- CAPA 6 & 7: ESTRUCTURA DEL HORIZONTE Y CUÁNTICA ---
@dataclass
class QuantumHorizonTopology:
    no_hair_delta_q: float # Desviación del Momento Cuadrupolar (-J^2/M)
    horizon_reflectivity: float # 0.0 en GR pura. >0 en Fuzzballs/ECOs
    echo_delay_time_ms: Optional[float] = None
    detected_theory: TheoryFamily = TheoryFamily.GENERAL_RELATIVITY
    quantum_confidence_sigma: float = 0.0 # Significancia estadística (Ej. 5.0)