import yaml
import numpy as np
import h5py
import os
import sys
import logging
from pathlib import Path

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("SSTG.Generator")
logger.setLevel(logging.DEBUG)

# --- CONFIGURACIÓN DE RUTAS ---
# Buscamos la raíz del proyecto 'qnim' para que las importaciones de 'src' funcionen
current_path = Path(__file__).resolve()
try:
    project_root = next(p for p in current_path.parents if p.name == 'qnim')
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    logger.debug(f"✓ Project root found: {project_root}")
except StopIteration:
    print("❌ Error: No se encontró la raíz 'qnim'. Ejecuta desde dentro del proyecto.")
    sys.exit(1)

# Importaciones de dominio sincronizadas con tu archivo value_objects.py
try:
    logger.debug("Importing pycbc.waveform...")
    from pycbc.waveform import get_td_waveform
    logger.debug("✓ pycbc.waveform imported")
except ImportError as e:
    logger.error(f"❌ Failed to import pycbc: {e}")
    raise

try:
    logger.debug("Importing TheoryFamily...")
    from src.domain.astrophysics.value_objects import TheoryFamily
    logger.debug("✓ TheoryFamily imported")
except ImportError as e:
    logger.error(f"❌ Failed to import TheoryFamily: {e}")
    raise

# Importar inyectores de capas 4, 5, 6, 7
try:
    logger.debug("Importing Layer4 (Quantum Foam)...")
    from src.domain.astrophysics.sstg.injectors.layer4_quantum_foam_complete import (
        Layer4QuantumFoamInjector, QuantumFoamParams
    )
    logger.debug("✓ Layer4 imported")
except ImportError as e:
    logger.warning(f"⚠ Could not import Layer4: {e}")
    Layer4QuantumFoamInjector = None
    QuantumFoamParams = None

try:
    logger.debug("Importing Layer5 (Beyond GR)...")
    from src.domain.astrophysics.sstg.injectors.layer5_beyond_gr_complete import (
        Layer5BeyondGRInjector, BeyondGRParams
    )
    logger.debug("✓ Layer5 imported")
except ImportError as e:
    logger.warning(f"⚠ Could not import Layer5: {e}")
    Layer5BeyondGRInjector = None
    BeyondGRParams = None

try:
    logger.debug("Importing Layer6 (Horizon Topology)...")
    from src.domain.astrophysics.sstg.injectors.layer6_horizon_topology_complete import (
        Layer6HorizonTopologyInjector, HorizonTopologyParams
    )
    logger.debug("✓ Layer6 imported")
except ImportError as e:
    logger.warning(f"⚠ Could not import Layer6: {e}")
    Layer6HorizonTopologyInjector = None
    HorizonTopologyParams = None

try:
    logger.debug("Importing Layer7 (Quantum Corrections)...")
    from src.domain.astrophysics.sstg.injectors.layer7_quantum_corrections_complete import (
        Layer7QuantumCorrectionsInjector, QuantumCorrectionParams
    )
    logger.debug("✓ Layer7 imported")
except ImportError as e:
    logger.warning(f"⚠ Could not import Layer7: {e}")
    Layer7QuantumCorrectionsInjector = None
    QuantumCorrectionParams = None

class StochasticSignalGenerator:
    def __init__(self, config_path="config/universe_params.yaml"):
        config_full_path = project_root / config_path
        with open(config_full_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = project_root / "data" / "synthetic" / "massive_dataset"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_event(self, theory_type: TheoryFamily):
        """Genera un evento basado en la probabilidad física y parámetros del config."""
        logger.debug(f"┌─ START generate_event(theory={theory_type})")
        
        try:
            conf = self.config['rangos_geometria']
            anom = self.config['anomalias_cuanticas']
            instr = self.config['instrumentacion']
            logger.debug(f"  ✓ Config loaded")
            
        except KeyError as e:
            logger.error(f"  ❌ Missing config key: {e}")
            raise
        
        # 1. Muestreo de parámetros base (Geometría de Kerr)
        try:
            logger.debug(f"  → Sampling masses...")
            m1 = np.random.uniform(conf['masa_1'][0], conf['masa_1'][1])
            m2 = np.random.uniform(conf['masa_2'][0], m1)
            chi = np.random.uniform(conf['spin_chi'][0], conf['spin_chi'][1])
            dist = np.random.uniform(conf['distancia_mpc'][0], conf['distancia_mpc'][1])
            logger.debug(f"    m1={m1:.2f}, m2={m2:.2f}, χ={chi:.3f}, d={dist:.1f} Mpc")
        except Exception as e:
            logger.error(f"  ❌ Failed to sample parameters: {e}")
            raise
        
        # 2. Generación de la onda base (Relatividad General - PyCBC)
        try:
            logger.debug(f"  → Generating GR baseline with PyCBC (EOB)...")
            dt = 1.0 / instr['sample_rate']
            logger.debug(f"    dt={dt:.6f}s, f_lower=20Hz")
            
            hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                                     mass1=m1, mass2=m2,
                                     spin1z=chi, spin2z=chi,
                                     delta_t=dt,
                                     f_lower=20,
                                     distance=dist)
            logger.debug(f"    ✓ Waveform generated: {len(hp)} samples")

            # Convertir TimeSeries de PyCBC a numpy arrays
            hp_array = hp.numpy()
            hc_array = hc.numpy()
            logger.debug(f"    ✓ Converted to numpy: shape={hp_array.shape}")
            
        except Exception as e:
            logger.error(f"  ❌ Waveform generation failed: {e}")
            raise

        # 3. Inyección de anomalías (Capas 5, 6, 7 según la teoría seleccionada)
        total_mass = (m1 + m2) / 2
        
        try:
            logger.debug(f"  → Applying theory layers...")
            
            if theory_type == TheoryFamily.KERR_VACUUM:
                logger.debug(f"    Theory: KERR_VACUUM (no anomalies)")
                result_h_plus = hp_array
                result_h_cross = hc_array
                physics_layers = "GR baseline"
                
            elif theory_type == TheoryFamily.BRANS_DICKE:
                logger.debug(f"    Theory: BRANS_DICKE (Layer 5)")
                if Layer5BeyondGRInjector is None:
                    logger.warning(f"      ⚠ Layer5 not available, using baseline")
                    result_h_plus = hp_array
                    result_h_cross = hc_array
                    physics_layers = "GR baseline (Layer5 unavailable)"
                else:
                    try:
                        params_bd = BeyondGRParams(
                            theory="Brans-Dicke",
                            omega_bd=np.random.uniform(50, 200),
                            mode="injection"
                        )
                        logger.debug(f"      ω_BD={params_bd.omega_bd:.1f}")
                        result = Layer5BeyondGRInjector.apply_beyond_gr_physics(
                            hp_array, hc_array, params_bd,
                            total_mass_msun=total_mass, distance_mpc=dist,
                            fs=int(instr['sample_rate'])
                        )
                        result_h_plus = result["h_plus"]
                        result_h_cross = result["h_cross"]
                        physics_layers = "Capa 5: Brans-Dicke dipolar"
                        logger.debug(f"      ✓ Layer 5 applied")
                    except Exception as e:
                        logger.warning(f"      ⚠ Layer 5 failed: {e}, using baseline")
                        result_h_plus = hp_array
                        result_h_cross = hc_array
                        physics_layers = "GR baseline (Layer5 error)"
                
            elif theory_type == TheoryFamily.GRAVASTAR:
                logger.debug(f"    Theory: GRAVASTAR (Layer 6 ECO)")
                if Layer6HorizonTopologyInjector is None:
                    logger.warning(f"      ⚠ Layer6 not available, using baseline")
                    result_h_plus = hp_array
                    result_h_cross = hc_array
                    physics_layers = "GR baseline (Layer6 unavailable)"
                else:
                    try:
                        delay_eco = 0.0001 * total_mass
                        params_eco = HorizonTopologyParams(
                            theory="ECO",
                            echo_delay=delay_eco,
                            echo_amplitude=np.random.uniform(0.10, 0.25),
                            n_echoes=np.random.randint(2, 4),
                            mode="injection"
                        )
                        logger.debug(f"      delay={delay_eco:.6f}s, amp={params_eco.echo_amplitude:.3f}")
                        result = Layer6HorizonTopologyInjector.apply_horizon_topology(
                            hp_array, hc_array, params_eco,
                            mass=total_mass,
                            fs=instr['sample_rate']
                        )
                        result_h_plus = result["h_plus"]
                        result_h_cross = result["h_cross"]
                        physics_layers = "Capa 6: ECO echoes"
                        logger.debug(f"      ✓ Layer 6 applied")
                    except Exception as e:
                        logger.warning(f"      ⚠ Layer 6 failed: {e}, using baseline")
                        result_h_plus = hp_array
                        result_h_cross = hc_array
                        physics_layers = "GR baseline (Layer6 error)"
                
            else:
                logger.warning(f"    ⚠ Unknown theory type: {theory_type}, using baseline")
                result_h_plus = hp_array
                result_h_cross = hc_array
                physics_layers = "GR baseline (unknown theory)"
            
            logger.debug(f"    ✓ Physics layers applied: {physics_layers}")
            
        except Exception as e:
            logger.error(f"  ❌ Physics layer application failed: {e}")
            raise
        
        logger.debug(f"└─ END generate_event (success)")
        
        return {
            "h_plus": result_h_plus,
            "h_cross": result_h_cross,
            "mass_1": m1,
            "mass_2": m2,
            "spin": chi,
            "distance_mpc": dist,
            "theory": str(theory_type),
            "physics_layers": physics_layers,
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
        """Almacena la señal y su 'etiqueta física' en HDF5 con logging."""
        try:
            logger.debug(f"  → Saving to HDF5: {filename}")
            path = self.output_dir / filename
            
            with h5py.File(path, "w") as f:
                logger.debug(f"    Saving strain data (shape={strain.shape}, dtype={strain.dtype})")
                f.create_dataset("strain", data=strain)
                
                logger.debug(f"    Saving metadata: {metadata}")
                for k, v in metadata.items():
                    f.attrs[k] = str(v) if not isinstance(v, (int, float, str)) else v
            
            logger.debug(f"  ✓ HDF5 file saved: {path}")
            
        except Exception as e:
            logger.error(f"  ❌ Failed to save HDF5: {e}")
            raise

# --- LANZADOR DE GENERACIÓN MASIVA ---
if __name__ == "__main__":
    logger.info("="*80)
    logger.info("Starting StochasticSignalGenerator.__main__")
    logger.info("="*80)
    
    try:
        logger.info("Initializing StochasticSignalGenerator...")
        gen = StochasticSignalGenerator()
        logger.info("✓ Generator initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize generator: {e}")
        sys.exit(1)
    
    TOTAL_EVENTOS = 500
    eventos_creados = 0
    intentos = 0
    failed_theories = {}

    logger.info(f"Starting generation of {TOTAL_EVENTOS} synthetic GW events...")

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
            logger.debug(f"[Attempt {intentos}] Generating event {eventos_creados} with theory {theory}")
            
            # Intentamos generar la física
            event = gen.generate_event(theory)
            hp = event["h_plus"]
            meta = {
                "m1": event["mass_1"],
                "m2": event["mass_2"],
                "chi": event["spin"],
                "distance": event["distance_mpc"],
                "theory": event["theory"],
                "physics_layers": event["physics_layers"]
            }
            
            nombre_archivo = f"qnim_evt_{eventos_creados:04d}.h5"
            gen.save_to_hdf5(nombre_archivo, hp, meta)
            eventos_creados += 1
            
            if eventos_creados % 50 == 0:
                logger.info(f"✓ {eventos_creados}/{TOTAL_EVENTOS} events generated (success rate: {100*eventos_creados/intentos:.1f}%)")
            
        except Exception as e:
            logger.debug(f"  ⚠ Failed (attempt {intentos}): {type(e).__name__}: {str(e)[:100]}")
            failed_theories[str(theory)] = failed_theories.get(str(theory), 0) + 1
            continue

    logger.info("="*80)
    logger.info(f"✨ Generation complete!")
    logger.info(f"   Total events created: {eventos_creados}/{TOTAL_EVENTOS}")
    logger.info(f"   Total attempts: {intentos}")
    logger.info(f"   Success rate: {100*eventos_creados/intentos:.1f}%")
    if failed_theories:
        logger.info(f"   Failed theories: {failed_theories}")
    logger.info(f"   Output: data/synthetic/massive_dataset/")
    logger.info("="*80)