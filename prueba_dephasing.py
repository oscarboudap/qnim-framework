# prueba_dephasing.py  — pégalo en la raíz y ejecútalo
import numpy as np
import sys
sys.path.insert(0, '.')
from src.infrastructure.physics_sstg_adapter import PhysicsSSTGAdapter, _generate_base_waveform
from src.infrastructure.stochastic_signal_generator import _inject_chern_simons, _inject_pn_deformation
from src.infrastructure.dephasing_features import DephasingFeatureExtractor

rng = np.random.default_rng(42)
sample_rate = 4096
duration = 4.0
dt = 1.0 / sample_rate

# Generar 3 señales GR base
print("Generando señales base...")
gr_strains = []
for _ in range(3):
    m1 = float(rng.uniform(10, 80))
    m2 = float(rng.uniform(10, m1))
    d  = float(rng.uniform(100, 1000))
    s, _ = _generate_base_waveform(m1, m2, d, duration, sample_rate, 20.0)
    gr_strains.append(s)
    print(f"  GR: m1={m1:.1f} m2={m2:.1f} d={d:.0f}Mpc  |strain|_max={np.abs(s).max():.3e}")

# Generar 1 señal beyond-GR (Chern-Simons, señal más fuerte)
m1, m2, d = 40.0, 35.0, 400.0
s_gr, _ = _generate_base_waveform(m1, m2, d, duration, sample_rate, 20.0)
htilde = np.fft.rfft(s_gr)
freqs  = np.fft.rfftfreq(len(s_gr), d=dt)
htilde_cs = _inject_chern_simons(htilde, freqs, d, kappa_gpc=0.83)
s_cs = np.fft.irfft(htilde_cs, n=len(s_gr))

print(f"\n  GR  |strain|_max = {np.abs(s_gr).max():.3e}")
print(f"  CS  |strain|_max = {np.abs(s_cs).max():.3e}")
print(f"  Diferencia relativa: {np.abs(s_cs - s_gr).max() / np.abs(s_gr).max():.4f}")

# Ajustar extractor y calcular features
extractor = DephasingFeatureExtractor(n_bands=6, f_low=20.0, f_high=512.0)
extractor.fit(gr_strains, dt=dt)

f_gr = extractor.transform(s_gr[np.newaxis], dt=dt)[0]
f_cs = extractor.transform(s_cs[np.newaxis], dt=dt)[0]

print(f"\nFeatures GR (raw, sin normalizar): {np.round(f_gr, 6)}")
print(f"Features CS (raw, sin normalizar): {np.round(f_cs, 6)}")
print(f"Diferencia absoluta max:           {np.abs(f_cs - f_gr).max():.6f}")