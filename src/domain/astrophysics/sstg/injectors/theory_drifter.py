# src/domain/astrophysics/sstg/injectors/theory_drifter.py

class TheoryDrifter:
    @staticmethod
    def inject_lqg_graininess(strain, length_scale):
        """
        Modifica el strain basado en la discretización del área de LQG.
        Añade una cuantización a la amplitud de la onda.
        """
        # El espacio no es continuo, la amplitud solo puede tomar valores discretos
        quantum_step = np.max(strain) * length_scale
        return np.round(strain / quantum_step) * quantum_step

    @staticmethod
    def inject_fuzzball_echoes(strain, merger_index, params):
        """
        Añade los ecos característicos de la Teoría de Cuerdas.
        La señal rebota en la superficie de la 'bola de cuerdas'.
        """
        echo_amplitude = 0.4
        delay_samples = int(4096 * 0.01) # 10ms de delay
        # Creamos una serie de ecos post-merger
        for i in range(1, 4):
            echo = np.roll(strain, delay_samples * i) * (echo_amplitude ** i)
            strain[merger_index:] += echo[merger_index:]
        return strain