import yaml
import numpy as np
import h5py
import os
import sys
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
# Buscamos la raíz del proyecto 'qnim' para que las importaciones de 'src' funcionen
current_path = Path(__file__).resolve()
try:
    project_root = next(p for p in current_path.parents if p.name == 'qnim')
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except StopIteration:
    print("❌ Error: No se encontró la raíz 'qnim'. Ejecuta desde dentro del proyecto.")
    sys.exit(1)

# Importaciones de dominio sincronizadas con tu archivo value_objects.py
from pycbc.waveform import get_td_waveform
from src.domain.astrophysics.value_objects import TheoryFamily

# Importar inyectores de capas 4, 5, 6, 7
from src.domain.astrophysics.sstg.injectors.layer4_quantum_foam_complete import (
    Layer4QuantumFoamInjector, QuantumFoamParams
)
from src.domain.astrophysics.sstg.injectors.layer5_beyond_gr_complete import (
    Layer5BeyondGRInjector, BeyondGRParams
)
from src.domain.astrophysics.sstg.injectors.layer6_horizon_topology_complete import (
    Layer6HorizonTopologyInjector, HorizonTopologyParams
)
from src.domain.astrophysics.sstg.injectors.layer7_quantum_corrections_complete import (
    Layer7QuantumCorrectionsInjector, QuantumCorrectionParams
)

class StochasticSignalGenerator:
    def __init__(self, config_path="config/universe_params.yaml"):
        config_full_path = project_root / config_path
        with open(config_full_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = project_root / "data" / "synthetic" / "massive_dataset"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_event(self, theory_type: TheoryFamily):
        """Genera un evento basado en la probabilidad física y parámetros del config."""
        conf = self.config['rangos_geometria']
        anom = self.config['anomalias_cuanticas']
        instr = self.config['instrumentacion']
        
        # 1. Muestreo de parámetros base (Geometría de Kerr)
        m1 = np.random.uniform(conf['masa_1'][0], conf['masa_1'][1])
        m2 = np.random.uniform(conf['masa_2'][0], m1)  # Masa 2 siempre menor o igual
        chi = np.random.uniform(conf['spin_chi'][0], conf['spin_chi'][1])
        dist = np.random.uniform(conf['distancia_mpc'][0], conf['distancia_mpc'][1])
        
        # 2. Generación de la onda base (Relatividad General - PyCBC)
        dt = 1.0 / instr['sample_rate']
        
        hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                                 mass1=m1, mass2=m2,
                                 spin1z=chi, spin2z=chi,
                                 delta_t=dt,
                                 f_lower=20,
                                 distance=dist)

        # Convertir TimeSeries de PyCBC a numpy arrays
        hp_array = hp.numpy()
        hc_array = hc.numpy()

        # 3. Inyección de anomalías (Capas 5, 6, 7 según la teoría seleccionada)
        total_mass = (m1 + m2) / 2  # Masa característica
        
        if theory_type == TheoryFamily.KERR_VACUUM:
            # Relatividad General pura - sin inyección
            result_h_plus = hp_array
            result_h_cross = hc_array
            physics_layers = "GR baseline"
            
        elif theory_type == TheoryFamily.BRANS_DICKE:
            # Capa 5: Brans-Dicke (escalar-tensorial)
            params_bd = BeyondGRParams(
                theory="Brans-Dicke",
                omega_bd=np.random.uniform(50, 200),  # Constrains observacionales
                mode="injection"
            )
            result = Layer5BeyondGRInjector.apply_beyond_gr_physics(
                hp_array, hc_array, params_bd,
                total_mass_msun=total_mass, distance_mpc=dist,
                fs=int(instr['sample_rate'])
            )
            result_h_plus = result["h_plus"]
            result_h_cross = result["h_cross"]
            physics_layers = "Capa 5: Brans-Dicke dipolar"
            
        elif theory_type == TheoryFamily.GRAVASTAR:
            # Capa 6: Ecos de Objetos Compactos Efectivos
            delay_eco = 0.0001 * total_mass  # Delay proporcional a masa
            params_eco = HorizonTopologyParams(
                theory="ECO",
                echo_delay=delay_eco,
                echo_amplitude=np.random.uniform(0.10, 0.25),
                n_echoes=np.random.randint(2, 4),
                mode="injection"
            )
            result = Layer6HorizonTopologyInjector.apply_horizon_topology(
                hp_array, hc_array, params_eco,
                mass=total_mass,
                fs=instr['sample_rate']
            )
            result_h_plus = result["h_plus"]
            result_h_cross = result["h_cross"]
            physics_layers = "Capa 6: ECO echoes"
            
        elif theory_type == TheoryFamily.STRING_FUZZBALL:
            # Capa 6: Fuzzball (String theory)
            params_fuzz = HorizonTopologyParams(
                theory="Fuzzball",
                echo_amplitude=np.random.uniform(0.12, 0.20),
                mode="injection"
            )
            result = Layer6HorizonTopologyInjector.apply_horizon_topology(
                hp_array, hc_array, params_fuzz,
                mass=total_mass,
                fs=instr['sample_rate']
            )
            result_h_plus = result["h_plus"]
            result_h_cross = result["h_cross"]
            physics_layers = "Capa 6: Fuzzball echoes"
            
        elif theory_type == TheoryFamily.PLANCK_STAR:
            # Capa 6: LQG con cuantización de área
            params_lqg = HorizonTopologyParams(
                theory="LQG",
                lqg_area_quantum=np.random.uniform(0.1, 0.5),
                mode="injection"
            )
            result = Layer6HorizonTopologyInjector.apply_horizon_topology(
                hp_array, hc_array, params_lqg,
                mass=total_mass,
                fs=instr['sample_rate']
            )
            result_h_plus = result["h_plus"]
            result_h_cross = result["h_cross"]
            physics_layers = "Capa 6: LQG quantization"
        
        elif theory_type == TheoryFamily.QUANTUM_FOAM:
            # Capa 4: Wheeler Quantum Foam
            # Modified Dispersion Relations + stochastic phase diffusion
            from scipy.fft import rfftfreq
            params_foam = QuantumFoamParams(
                mdr_strength=np.random.uniform(0.005, 0.02),  # 0.5-2% correction
                mdr_exponent=2.0,
                diffusion_coefficient=np.random.uniform(5e-4, 2e-3),
                coherence_timescale=max(0.05, 0.2 / total_mass)
            )
            
            # Compute frequency array for MDR
            dt = 1.0 / instr['sample_rate']
            freq_array = rfftfreq(len(hp_array), dt)
            freq_array = np.abs(freq_array)
            
            result = Layer4QuantumFoamInjector.inject_quantum_foam(
                hp_array, hc_array, params_foam,
                freq_array, int(instr['sample_rate'])
            )
            result_h_plus = result["h_plus"]
            result_h_cross = result["h_cross"]
            physics_layers = "Capa 4: Wheeler Quantum Foam (MDR + phase diffusion)"
            
        else:
            # Default: GR
            result_h_plus = hp_array
            result_h_cross = hc_array
            physics_layers = "GR baseline (default)"

        return result_h_plus, result_h_cross, {
            "m1": round(m1, 2), 
            "m2": round(m2, 2), 
            "chi": round(chi, 3), 
            "distance": round(dist, 1),
            "theory": theory_type.value,
            "physics_layers": physics_layers
        }

    def _apply_quantum_perturbation(self, waveform, delta_q):
        """
        [DEPRECATED] Método antiguo de perturbación simple.
        Ahora usamos inyectores rigurosos de Capas 5, 6, 7.
        
        Conservado solo para compatibilidad backwards.
        """
        # Convertimos el objeto TimeSeries de PyCBC a un array de numpy
        data = waveform.numpy()
        
        t = np.linspace(0, 1, len(data))
        # Perturbación antigua (muy simplificada)
        phase_shift = np.exp(1j * delta_q * np.pi * t**2)
        
        # Devolvemos el array modificado (solo la parte real para el strain)
        return (data * phase_shift.real)

    def save_to_hdf5(self, filename, strain, metadata):
        """Almacena la señal y su 'etiqueta física' en HDF5."""
        path = self.output_dir / filename
        with h5py.File(path, "w") as f:
            f.create_dataset("strain", data=strain)
            for k, v in metadata.items():
                f.attrs[k] = v
        print(f"✅ Evento guardado: {filename} [{metadata['theory']}]")

# --- LANZADOR DE GENERACIÓN MASIVA ---
if __name__ == "__main__":
    gen = StochasticSignalGenerator()
    
    TOTAL_EVENTOS = 500  # ¡Vamos a por el dataset de verdad!
    eventos_creados = 0
    intentos = 0

    print(f"🚀 Iniciando generación masiva de {TOTAL_EVENTOS} eventos...")

    while eventos_creados < TOTAL_EVENTOS:
        intentos += 1
        dice = np.random.random()
        
        # Selección de teoría (Priors)
        if dice < 0.80:
            theory = TheoryFamily.KERR_VACUUM
        elif 0.80 <= dice < 0.87:
            theory = TheoryFamily.PLANCK_STAR
        elif 0.87 <= dice < 0.92:
            theory = TheoryFamily.STRING_FUZZBALL
        elif 0.92 <= dice < 0.96:
            theory = TheoryFamily.BRANS_DICKE
        elif 0.96 <= dice < 0.99:
            theory = TheoryFamily.GRAVASTAR
        else:  # 0.99 - 1.00 (1% of events)
            theory = TheoryFamily.QUANTUM_FOAM
            
        try:
            # Intentamos generar la física
            hp, _, meta = gen.generate_event(theory)
            nombre_archivo = f"qnim_evt_{eventos_creados:04d}.h5"
            gen.save_to_hdf5(nombre_archivo, hp, meta)
            eventos_creados += 1
        except RuntimeError as e:
            # Si chocamos con el límite de Nyquist, ignoramos y seguimos
            print(f"⚠️ Salto en intento {intentos}: Parámetros extremos (Nyquist Limit). Reintentando...")
            continue

    print(f"\n✨ Dataset completado: {eventos_creados} realidades listas en data/synthetic/massive_dataset")