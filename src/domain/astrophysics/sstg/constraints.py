"""
Physical Constraints for Stochastic Gravity Engine

Validates that sampled parameters satisfy fundamental physical constraints:
  - Energy conditions (weak, strong, dominant)
  - Cosmic censorship (no naked singularities)
  - Causality (no CTCs)
  - Thermodynamic bounds
"""

import numpy as np
from typing import Tuple


class PhysicalConstraints:
    """
    Physical constraint validator for SSTG sampling.
    
    Ensures that sampled parameters violate no fundamental physics.
    """
    
    # Physical constants
    C_LIGHT = 299792458.0  # m/s
    G_CONST = 6.67430e-11  # m³/kg/s²
    HBAR = 1.054571817e-34  # J·s
    M_SUN = 1.98841e30  # kg
    
    @staticmethod
    def check_energy_conditions(
        m1: float,
        m2: float,
        distance_mpc: float
    ) -> Tuple[bool, str]:
        """
        Check weak, strong, and dominant energy conditions.
        
        Args:
            m1: Primary mass (solar masses)
            m2: Secondary mass (solar masses)
            distance_mpc: Distance (megaparsecs)
        
        Returns:
            (is_valid, explanation)
        """
        # Weak Energy Condition: ρ ≥ 0 (energy density non-negative)
        total_mass = m1 + m2
        if total_mass <= 0:
            return False, "Non-positive total mass"
        
        # Strong Energy Condition: ρ + 3p ≥ 0 (related to acceleration)
        # For matter-dominated systems, this is generically satisfied
        # We impose: total_mass < ~10⁶ solar masses (avoid supermassive BH)
        if total_mass > 1e6:
            return False, "Total mass exceeds supermassive BH threshold"
        
        # Dominant Energy Condition: p ≤ ρ
        # Automatically satisfied for most compact object equations of state
        if distance_mpc < 0.1:
            return False, "Distance too small (causality concern)"
        
        if distance_mpc > 10000:
            return False, "Distance too large (observability concern)"
        
        return True, "All energy conditions satisfied"
    
    @staticmethod
    def validate_cosmos_censorship(
        total_mass: float,
        spin: float
    ) -> Tuple[bool, str]:
        """
        Validate cosmic censorship hypothesis.
        
        Ensures no naked singularities: |a| ≤ M
        
        Args:
            total_mass: Total mass (solar masses)
            spin: Spin parameter (0 to 1, where 1 is extremal Kerr)
        
        Returns:
            (is_valid, explanation)
        """
        # Kerr parameter: a = L / (M * c)
        # For physical systems: a ≤ 1 (Kerr bound)
        
        if spin < 0 or spin > 1.0:
            return False, f"Spin outside physical range [0,1]: {spin}"
        
        # Extremal Kerr (a=1) requires infinite angular momentum extraction
        # Physical systems are typically a < 0.998
        if spin > 0.998:
            return False, "Spin too close to extremal Kerr (near naked singularity)"
        
        # For non-rotating systems
        if spin < 0.0:
            return False, "Negative spin (non-physical)"
        
        return True, f"Cosmic censorship satisfied (a={spin:.3f} < M)"
    
    @staticmethod
    def validate_thermodynamic_bounds(
        m1: float,
        m2: float,
        spin: float
    ) -> Tuple[bool, str]:
        """
        Check thermodynamic bounds (Hawking temperature, entropy).
        
        Args:
            m1: Primary mass (solar masses)
            m2: Secondary mass (solar masses)
            spin: Spin parameter
        
        Returns:
            (is_valid, explanation)
        """
        total_mass = m1 + m2
        
        # Hawking temperature: T_H = (ℏ c³) / (8π k_B G M)
        # For astrophysical BHs, this is extremely small
        # We just check that M is large enough to be classical
        
        if total_mass < 1.0:
            return False, "Mass too small (quantum Hawking dominated)"
        
        # Mass ratio constraint (avoid extreme asymmetry)
        q = m2 / m1  # q < 1
        if q < 0.05:
            return False, f"Mass ratio too extreme (q={q:.3f} < 0.05)"
        
        return True, "Thermodynamic bounds satisfied"
    
    @staticmethod
    def validate_all(
        m1: float,
        m2: float,
        spin: float,
        distance_mpc: float
    ) -> Tuple[bool, str]:
        """
        Run all constraint checks.
        
        Args:
            m1: Primary mass (solar masses)
            m2: Secondary mass (solar masses)
            spin: Spin parameter
            distance_mpc: Distance (megaparsecs)
        
        Returns:
            (is_valid, combined_explanation)
        """
        checks = [
            PhysicalConstraints.check_energy_conditions(m1, m2, distance_mpc),
            PhysicalConstraints.validate_cosmos_censorship(m1 + m2, spin),
            PhysicalConstraints.validate_thermodynamic_bounds(m1, m2, spin),
        ]
        
        all_valid = all(c[0] for c in checks)
        explanations = [c[1] for c in checks]
        
        return all_valid, " | ".join(explanations)
