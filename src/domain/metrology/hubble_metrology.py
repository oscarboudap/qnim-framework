# src/domain/metrology/hubble_metrology.py

class HubbleMetrology:
    """
    Validador de la Capa 4 (Cosmología).
    Infiere la constante de Hubble (H0) usando la onda gravitacional como sirena estándar.
    """
    C_KM_S = 299792.458 # Velocidad de la luz en km/s

    @classmethod
    def infer_hubble_constant(cls, luminosity_distance_mpc: float, redshift_z: float) -> float:
        """
        Calcula H0 asumiendo la ley de Hubble-Lemaître para redshifts bajos/medios:
        v = H0 * d_L  =>  c * z = H0 * d_L
        """
        if luminosity_distance_mpc <= 0 or redshift_z <= 0:
            return 0.0
            
        recession_velocity = cls.C_KM_S * redshift_z
        h0_estimate = recession_velocity / luminosity_distance_mpc
        return round(h0_estimate, 2)