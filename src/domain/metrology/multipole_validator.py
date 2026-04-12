class MultipoleValidator:
    """
    Estudia el Teorema de No-Cabello y clasifica el origen del evento
    según el formalismo de Christensen & Meyer (2022).
    """
    def check_no_hair_theorem(self, mass_1: float, mass_2: float, observed_m2: float):
        total_mass = mass_1 + mass_2
        # Asumimos un espín efectivo promediado para la métrica de Kerr
        spin_eff = 0.7 
        
        # Valor teórico según la Relatividad General (Kerr)
        # M2 = -a^2 * M^3
        theoretical_m2 = -(spin_eff**2) * (total_mass**3)
        
        # Desviación (El "Cabello" cuántico)
        delta_q = abs(observed_m2 - theoretical_m2) / abs(theoretical_m2) if theoretical_m2 != 0 else 0
        
        # Identificación positiva del objeto
        if total_mass > 50 and delta_q < 0.05:
            obj_type = "Binary Black Hole (BBH)"
        elif 2.0 < total_mass < 5.0:
            obj_type = "Binary Neutron Star (BNS)"
        elif delta_q >= 0.05:
            obj_type = "Exotic Compact Object (ECO) / Fuzzball"
        else:
            obj_type = "Compact Binary Candidate"

        return {
            "delta_q": delta_q,
            "is_pure_kerr": delta_q < 0.05,
            "object_type": obj_type,
            "total_mass": total_mass
        }