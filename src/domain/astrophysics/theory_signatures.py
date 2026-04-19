"""
FIRMAS TEÓRICAS Y DISCRIMINADORES.

Implementaciones concretas de calculadores de evidencia bayesiana
y discriminadores de teorías para cada una de las 5 clases de BSR:
- Escalar-tensorial (Brans-Dicke, Horndeski)
- Gravedad modificada (f(R), dRGT)
- Dimensiones extra (Kaluza-Klein, Randall-Sundrum)  
- ECOs y objetos exóticos
- Superradiancia de axiones
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Set
import numpy as np
from abc import ABC, abstractmethod

from .entities import (
    QuantumDecodedEvent, BayesianEvidenceCalculator, 
    TheoryDiscriminator, LayerAnalyzer
)
from .value_objects import TheoryFamily, Measurement


# ============================================================================
# CAPA 5: ANALIZADORES ESPECÍFICOS POR TEORÍA
# ============================================================================

class ScalarTensorAnalyzer(LayerAnalyzer):
    """
    Analizador de firmas de teorías escalar-tensorial.
    
    Busca:
    - Emisión dipolar (extra radiación)
    - Desviación de velocidad GW respecto a c
    - Modo escalar de respiración
    - Cambios en la evolución de fase respecto a GR
    """
    
    def extract_observables(self, event: QuantumDecodedEvent) -> Dict[str, Measurement]:
        observables = {}
        
        if not event.beyond_gr or not event.beyond_gr.scalar_tensor:
            return observables
        
        st = event.beyond_gr.scalar_tensor
        
        if st.dipolar_emission_flux_phi:
            observables["dipolar_flux"] = st.dipolar_emission_flux_phi
        if st.gw_speed_deviation:
            observables["c_gw_deviation"] = st.gw_speed_deviation
        if st.breathing_mode_amplitude:
            observables["breathing_mode"] = st.breathing_mode_amplitude
        
        return observables
    
    def compute_layer_significance(self, event: QuantumDecodedEvent) -> float:
        """
        Retorna SNR de la evidencia de teoría escalar-tensorial.
        """
        observables = self.extract_observables(event)
        if not observables:
            return 0.0
        
        # Combina SNRs de todos los observables
        snrs = []
        for obs in observables.values():
            snrs.append(obs.signal_to_noise())
        
        return float(np.mean(snrs) / 8.0)  # Normaliza a [0,1] asumiendo SNR_max ~ 8


class ModifiedGravityAnalyzer(LayerAnalyzer):
    """
    Analizador de firmas de gravedad modificada.
    
    Busca:
    - Dispersión de grupo (chirp invertido)
    - Acumulación de desfase de fase proporcional a distancia
    - Evolución no-monotónica de amplitud
    """
    
    def extract_observables(self, event: QuantumDecodedEvent) -> Dict[str, Measurement]:
        observables = {}
        
        if not event.beyond_gr or not event.beyond_gr.modified_gravity_dispersion:
            return observables
        
        mgd = event.beyond_gr.modified_gravity_dispersion
        
        if mgd.massive_graviton_mass:
            observables["m_graviton"] = mgd.massive_graviton_mass
        if mgd.group_delay_phase_shift:
            observables["group_delay"] = mgd.group_delay_phase_shift
        
        return observables
    
    def compute_layer_significance(self, event: QuantumDecodedEvent) -> float:
        observables = self.extract_observables(event)
        if not observables:
            return 0.0
        
        snrs = [obs.signal_to_noise() for obs in observables.values()]
        return float(np.mean(snrs) / 8.0)


class ExtraDimensionAnalyzer(LayerAnalyzer):
    """
    Analizador de firmas de dimensiones extra.
    
    Busca:
    - Amplitud anómala en función de distancia: h ~ d_L^{-(1+n/2)}
    - Resonancias de KK en el ringdown
    - Escape de energía GW al bulk
    """
    
    def extract_observables(self, event: QuantumDecodedEvent) -> Dict[str, Measurement]:
        observables = {}
        
        if not event.beyond_gr or not event.beyond_gr.extra_dimensions:
            return observables
        
        ed = event.beyond_gr.extra_dimensions
        
        if ed.bulk_amplitude_reduction:
            observables["bulk_reduction"] = ed.bulk_amplitude_reduction
        if ed.kk_resonance_frequencies:
            # Contar resonancias detectadas
            num_resonances = len(ed.kk_resonance_frequencies)
            observables["num_kk_resonances"] = Measurement(float(num_resonances), np.sqrt(num_resonances))
        
        return observables
    
    def compute_layer_significance(self, event: QuantumDecodedEvent) -> float:
        observables = self.extract_observables(event)
        if not observables:
            return 0.0
        
        snrs = [obs.signal_to_noise() for obs in observables.values()]
        return float(np.mean(snrs) / 8.0)


class ECOAnalyzer(LayerAnalyzer):
    """
    Analizador de firmas de Objetos Compactos Exóticos.
    
    Busca:
    - Ecos espacio-temporales después del ringdown
    - Timing y reflectividad (identifica tipo de ECO)
    - Ringdown no-monotónico
    """
    
    def extract_observables(self, event: QuantumDecodedEvent) -> Dict[str, Measurement]:
        observables = {}
        
        if not event.beyond_gr or not event.beyond_gr.exotic_objects:
            return observables
        
        eco = event.beyond_gr.exotic_objects
        
        if eco.echo_time_delay_dt:
            observables["echo_delay"] = eco.echo_time_delay_dt
        if eco.echo_reflectivity_R:
            observables["reflectivity"] = eco.echo_reflectivity_R
        if eco.num_detectable_echoes:
            observables["num_echoes"] = Measurement(float(eco.num_detectable_echoes), 0.5)
        
        return observables
    
    def compute_layer_significance(self, event: QuantumDecodedEvent) -> float:
        observables = self.extract_observables(event)
        if not observables:
            return 0.0
        
        snrs = [obs.signal_to_noise() for obs in observables.values()]
        return float(np.mean(snrs) / 8.0)


class AxionSuperradianceAnalyzer(LayerAnalyzer):
    """
    Analizador de firmas de superradiancia de axiones.
    
    Busca:
    - Pérdida de masa/espín del BH predecible
    - Nube de axiones emitiendo GW monocromáticas a f ~ m_a*c²/h
    - Shift en frecuencias QNM
    """
    
    def extract_observables(self, event: QuantumDecodedEvent) -> Dict[str, Measurement]:
        observables = {}
        
        if not event.beyond_gr or not event.beyond_gr.axion_superradiance:
            return observables
        
        axion = event.beyond_gr.axion_superradiance
        
        if axion.axion_mass_constraint:
            observables["axion_mass"] = axion.axion_mass_constraint
        if axion.qnm_frequency_shift:
            observables["qnm_shift"] = axion.qnm_frequency_shift
        if axion.monochromatic_cw_signal:
            observables["cw_amplitude"] = axion.monochromatic_cw_signal
        
        return observables
    
    def compute_layer_significance(self, event: QuantumDecodedEvent) -> float:
        observables = self.extract_observables(event)
        if not observables:
            return 0.0
        
        snrs = [obs.signal_to_noise() for obs in observables.values()]
        return float(np.mean(snrs) / 8.0)


# ============================================================================
# CALCULADOR BAYESIANO: Evidencia para cada teoría
# ============================================================================

@dataclass
class BayesianMultiModelCalculator(BayesianEvidenceCalculator):
    """
    Calculador de evidencia bayesiana Z = P(data|theory) × prior.
    
    Implementa el cálculo de likelihood para cada familia teórica
    basándose en los observables de la Capa 5.
    """
    
    def compute_log_evidence(
        self, 
        event: QuantumDecodedEvent,
        theory: TheoryFamily
    ) -> Tuple[float, float]:
        """
        Calcula log(Z) usando la Capa 5 (Beyond-GR signatures).
        
        Estrategia:
        1. Extrae SNR de los observables relevantes a la teoría
        2. Propaga incertidumbres
        3. Calcula likelihood integrado
        4. Retorna (log_Z, sigma_log_Z)
        """
        
        if not event.beyond_gr:
            # Si no hay análisis Beyond-GR, asumir GR
            if theory == TheoryFamily.KERR_VACUUM:
                return (0.0, 0.1)  # log(Z_Kerr) = referencia
            else:
                return (-10.0, 5.0)  # Penalización por falta de datos
        
        # Selecciona analizador según teoría
        if theory in [TheoryFamily.BRANS_DICKE, TheoryFamily.HORNDESKI]:
            analyzer = ScalarTensorAnalyzer()
        elif theory in [TheoryFamily.F_R_GRAVITY, TheoryFamily.MASSIVE_GRAVITY]:
            analyzer = ModifiedGravityAnalyzer()
        elif theory in [TheoryFamily.KALUZA_KLEIN, TheoryFamily.RANDALL_SUNDRUM]:
            analyzer = ExtraDimensionAnalyzer()
        elif theory in [TheoryFamily.GRAVASTAR, TheoryFamily.BOSON_STAR, 
                       TheoryFamily.STRING_FUZZBALL, TheoryFamily.PLANCK_STAR]:
            analyzer = ECOAnalyzer()
        elif theory == TheoryFamily.AXION_CLOUD:
            analyzer = AxionSuperradianceAnalyzer()
        elif theory == TheoryFamily.KERR_VACUUM:
            # GR: penaliza si hay firmas Beyond-GR significativas
            log_z = -event.beyond_gr.beyond_gr_confidence.signal_to_noise() ** 2 / 2.0
            return (log_z, 0.5)
        else:
            return (-10.0, 5.0)
        
        # Calcula SNR de la teoría usando sus observables
        snr = analyzer.compute_layer_significance(event)
        observables = analyzer.extract_observables(event)
        
        if not observables:
            # Sin observables relevantes, no hay evidencia
            return (-5.0, 2.0)
        
        # Likelihood proporcional a SNR²/2 (asume gaussiana)
        # log(Z) ~ -SNR²/2 (para hipótesis nula H0: teoría es falsa)
        # log(Z) ~ SNR²/2 (para hipótesis alt H1: teoría es verdadera)
        
        # Calcula SNR combinado de todos los observables
        snrs = [obs.signal_to_noise() for obs in observables.values()]
        combined_snr = np.sqrt(np.sum(np.array(snrs)**2))
        
        # Bayes factor respecto a GR (referencia log_Z = 0)
        log_z = combined_snr ** 2 / 2.0 - 5.0  # Sustrae prior débil
        
        # Incertidumbre en log_Z ~ 1 / SNR
        sigma_log_z = 1.0 / max(combined_snr, 1.0)
        
        return (log_z, sigma_log_z)
    
    def compute_likelihood(
        self, 
        event: QuantumDecodedEvent,
        theory: TheoryFamily
    ) -> float:
        """
        Retorna L(data|theory) normalizado a [0,1].
        """
        log_z, _ = self.compute_log_evidence(event, theory)
        # Normaliza exponencial
        likelihood = np.exp(log_z) / (1.0 + np.exp(log_z))  # Sigmoide
        return float(likelihood)


# ============================================================================
# DISCRIMINADOR: Comparación teórica por pares
# ============================================================================

@dataclass
class BayesianTheoryDiscriminator(TheoryDiscriminator):
    """
    Discrimina entre pares de teorías mediante su Bayes factor.
    """
    
    evidence_calculator: BayesianEvidenceCalculator
    
    def bayes_factor(
        self, 
        event: QuantumDecodedEvent,
        theory_a: TheoryFamily,
        theory_b: TheoryFamily
    ) -> float:
        """
        Retorna B_AB = Z_A / Z_B.
        
        Interpretación:
        - B > 10: Fuerte evidencia para A
        - B > 100: Muy fuerte evidencia para A
        - 1/10 < B < 10: Evidencia indecisa
        """
        log_z_a, _ = self.evidence_calculator.compute_log_evidence(event, theory_a)
        log_z_b, _ = self.evidence_calculator.compute_log_evidence(event, theory_b)
        
        log_bayes = log_z_a - log_z_b
        return float(np.exp(log_bayes))


# ============================================================================
# TABLA DE FIRMAS: Identificación rápida de teorías
# ============================================================================

class TheorySignatureLibrary:
    """
    Tabla de referencia de firmas características de cada teoría.
    Permite identificación rápida basada en observables detectados.
    """
    
    SIGNATURES = {
        TheoryFamily.KERR_VACUUM: {
            "description": "Relatividad General pura",
            "key_observables": [],
            "forbidden_observables": ["dipolar_flux", "echo_dt", "massive_graviton", "breathing_mode"],
            "typical_snr": 0.0,
        },
        TheoryFamily.BRANS_DICKE: {
            "description": "Emisión dipolar, respira escalar",
            "key_observables": ["dipolar_flux", "breathing_mode", "c_gw_deviation"],
            "forbidden_observables": ["echo_dt"],
            "typical_snr": 5.0,
        },
        TheoryFamily.HORNDESKI: {
            "description": "Generalización Horndeski/DHOST",
            "key_observables": ["gw_speed_deviation", "breathing_mode"],
            "forbidden_observables": [],
            "typical_snr": 4.0,
        },
        TheoryFamily.MASSIVE_GRAVITY: {
            "description": "Gravitón masivo dRGT",
            "key_observables": ["m_graviton", "group_delay"],
            "forbidden_observables": ["echo_dt"],
            "typical_snr": 6.0,
        },
        TheoryFamily.KALUZA_KLEIN: {
            "description": "Dimensiones extra, fugas de energía",
            "key_observables": ["bulk_reduction", "num_kk_resonances"],
            "forbidden_observables": ["echo_dt"],
            "typical_snr": 4.0,
        },
        TheoryFamily.GRAVASTAR: {
            "description": "ECO gravastar, ecos fuertes",
            "key_observables": ["echo_delay", "reflectivity", "num_echoes"],
            "forbidden_observables": [],
            "typical_snr": 8.0,
        },
        TheoryFamily.AXION_CLOUD: {
            "description": "Superradiancia de axiones",
            "key_observables": ["axion_mass", "qnm_shift", "cw_amplitude"],
            "forbidden_observables": ["echo_dt"],
            "typical_snr": 3.0,
        },
    }
    
    @classmethod
    def quick_identify_theory(cls, observables_detected: Set[str]) -> TheoryFamily:
        """
        Identificación rápida: busca la teoría con mejor coincidencia
        de observables detectados.
        """
        best_theory = TheoryFamily.KERR_VACUUM
        best_score = 0.0
        
        for theory, sig in cls.SIGNATURES.items():
            # Puntuación: #observables detectados - #observables prohibidos
            score = len(set(sig["key_observables"]) & observables_detected)
            penalty = len(set(sig["forbidden_observables"]) & observables_detected)
            
            theory_score = score - penalty
            if theory_score > best_score:
                best_score = theory_score
                best_theory = theory
        
        return best_theory
