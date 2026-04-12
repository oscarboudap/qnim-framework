# src/domain/astrophysics/sstg/injectors/exotic_effects.py

class MultiverseInjector:
    @staticmethod
    def inject_wormhole_transfer(strain, params):
        """
        Simula una señal que atraviesa una garganta hacia otro universo.
        Produce una inversión de quiralidad y un eco de 'salida'.
        """
        if params.get('is_wormhole', False):
            # Inversión de fase (propiedad de algunas gargantas de Schwarzschild)
            strain = -1 * strain 
            # Añadir eco de salida (la señal que 'rebota' en la otra boca)
            exit_echo = np.roll(strain, int(4096 * 0.2)) * 0.3
            return strain + exit_echo
        return strain