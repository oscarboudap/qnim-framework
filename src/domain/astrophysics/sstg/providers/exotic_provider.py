import numpy as np

class ExoticProvider:
    """Generador de formas de onda para objetos compactos exóticos (ECOs)."""

    def generate_white_hole(self, params, fs=4096):
        """
        Simula la inversión temporal de una fusión. 
        En lugar de un chirp ascendente, la frecuencia decae.
        """
        t = np.linspace(0, params['duration'], int(fs * params['duration']))
        # f(t) inverso: la energía sale del horizonte
        f_inst = params['f_min'] * (1 + (t/params['duration']))**-2
        strain = np.sin(2 * np.pi * f_inst * t)
        return t, strain

    def generate_boson_star(self, params, fs=4096):
        """
        Las estrellas de bosones no tienen horizonte de sucesos. 
        El Ringdown es mucho más largo y con frecuencias distintas.
        """
        t, base_strain = self._get_base_inspiral(params, fs)
        # Modificación: eliminamos el corte abrupto del horizonte
        ringdown_mod = np.exp(-t * 0.1) * np.cos(2 * np.pi * 500 * t)
        return t, base_strain + ringdown_mod