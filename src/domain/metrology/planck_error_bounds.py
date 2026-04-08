class PlanckErrorBounds:
    """
    Define los límites de exclusión para Gravedad Cuántica (LQG y Cuerdas).
    """
    def get_exclusion_limit(self, snr: float):
        # A mayor SNR, podemos detectar granos de espacio (R) más pequeños
        # R_planck ~ 1e-35 metros. 
        # Tu software acota el 'R' efectivo.
        detectable_r = 1.0 / (snr**2) 
        return {
            "min_detectable_r": detectable_r,
            "unit": "Planck Length Units"
        }