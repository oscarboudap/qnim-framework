"""Calculadora de la Constante de Hubble desde ondas gravitacionales.

Nivel Postdoctoral: Usa eventos de ondas gravitacionales como "sirenas
 estándar" para medir la expansión cósmica (H0).

Ecuación fundamental (ley de Hubble-Lemaître):
    v = H₀ d_L
    c z = H₀ d_L  →  H₀ = (c z) / d_L

Donde:
- c: velocidad de la luz (299792.458 km/s)
- z: redshift cosmológico (espectroscópico o fotométrico)
- d_L: distancia de luminosidad (de la onda GW)
- H₀: Constante de Hubble (km/s/Mpc)

Tensión Hubble ("Hubble Tension"):
- Planck T_CMB: H₀ = 67.4 ± 0.5 km/s/Mpc
- GW + host galaxy: H₀ = 73-75 km/s/Mpc
- Discrepancia: ~3-sigma (posible nueva física z ~ 0.1)
"""

from dataclasses import dataclass
from .value_objects import HubbleConstant


class HubbleCosmologyCalculator:
    """
    Servicio de dominio stateless para cosmología de ondas gravitacionales.
    
    Infiere parámetros cosmológicos (H₀, Ω_m, Ω_Λ) desde distancias de
    luminosidad de eventos GW.
    """
    
    # Constantes de la naturaleza
    SPEED_OF_LIGHT_KM_S = 299792.458  # km/s (exacto por definición SI)
    
    @staticmethod
    def infer_hubble_constant(
        luminosity_distance_mpc: float,
        redshift_z: float,
        luminosity_distance_uncertainty_mpc: float = None
    ) -> HubbleConstant:
        """
        Calcula H₀ desde distancia de luminosidad y redshift.
        
        Asume ley de Hubble-Lemaître: v = H₀ d_L
        Válido para z < 0.2 (régimen de Hubble lineal)
        
        Args:
            luminosity_distance_mpc: d_L desde onda GW (Mpc)
            redshift_z: Redshift del host galaxy (z > 0)
            luminosity_distance_uncertainty_mpc: Incertidumbre en d_L
            
        Returns:
            HubbleConstant value object con valor ± intervalo
            
        Raises:
            ValueError: Si inputs fuera de rango físico
        """
        if luminosity_distance_mpc <= 0:
            raise ValueError("d_L debe ser positiva")
        if redshift_z <= 0:
            raise ValueError("Redshift z debe ser positivo (z > 0)")
        if redshift_z > 0.3:
            raise ValueError("z > 0.3: régimen no-lineal, necesitar Ω_m, Ω_Λ")
        
        # Cálculo central
        recession_velocity = HubbleCosmologyCalculator.SPEED_OF_LIGHT_KM_S * redshift_z
        h0_value = recession_velocity / luminosity_distance_mpc
        
        # Propagación de incertidumbre
        if luminosity_distance_uncertainty_mpc:
            # dH0/dL = -H0/L = -(c*z)/L²
            h0_uncertainty = (HubbleCosmologyCalculator.SPEED_OF_LIGHT_KM_S * redshift_z * 
                            luminosity_distance_uncertainty_mpc) / (luminosity_distance_mpc ** 2)
        else:
            h0_uncertainty = 0.05 * h0_value  # 5% por defecto
        
        return HubbleConstant(
            value_km_s_mpc=h0_value,
            redshift=redshift_z,
            luminosity_distance_mpc=luminosity_distance_mpc,
            upper_bound_km_s_mpc=h0_value + h0_uncertainty,
            lower_bound_km_s_mpc=h0_value - h0_uncertainty
        )