class MultipoleValidator:
    """
    Estudia el Teorema de No-Cabello analizando la estructura multipolar.
    M2 = -a^2 * M^3 (Relatividad General)
    """
    def check_no_hair_theorem(self, mass: float, spin: float, observed_m2: float):
        # Valor teórico según la Relatividad General de Einstein
        theoretical_m2 = -(spin**2) * (mass**3)
        
        # Desviación (El "Cabello" cuántico)
        delta_q = abs(observed_m2 - theoretical_m2) / abs(theoretical_m2)
        
        is_kerr = delta_q < 0.05 # Umbral de tolerancia de la RG
        return {
            "delta_q": delta_q,
            "is_pure_kerr": is_kerr,
            "object_type": "Kerr Black Hole" if is_kerr else "Exotic Compact Object (ECO)"
        }