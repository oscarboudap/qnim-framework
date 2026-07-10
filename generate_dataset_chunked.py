"""
generate_dataset_chunked.py
============================
Genera el dataset completo de 54.600 eventos en chunks para evitar
OOM en WSL. Guarda X_train, y_train, X_val, y_val en disco como .npy
y ajusta el PCA incrementalmente con sklearn IncrementalPCA.

USO:
    python generate_dataset_chunked.py [--n-per-class 4200] [--approximant IMRPhenomPv2]

Salida:
    data/dataset_full/X_train.npy   (54600, 12)
    data/dataset_full/y_train.npy   (54600,)
    data/dataset_full/X_val.npy     (7800, 12)
    data/dataset_full/y_val.npy     (7800,)
    data/dataset_full/pca.pkl
    data/dataset_full/scaler.pkl
"""

import argparse, sys, time, pickle, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler

from src.infrastructure.ligo_pca_adapter import (
    LIGOPCAAdapter, _generate_base_waveform, _whiten_strain
)
from src.infrastructure.stochastic_signal_generator import (
    _inject_standard_siren, _inject_qnm_subdominant,
    _inject_pn_deformation, _inject_scalar_tensor,
    _inject_graviton_mass, _inject_chern_simons,
    _inject_liv, _inject_gup,
)

THEORY_CLASSES = [
    "GR", "standard-siren", "qnm-21", "qnm-33", "pn-deformation",
    "extra-dimensions", "scalar-tensor", "graviton-mass", "chern-simons",
    "liv-alpha2", "liv-alpha4", "loop-quantum-gravity", "gup",
]


def generate_one_white(theory_class, rng, approximant, sample_rate=4096,
                        duration=4.0, f_lower=20.0, f_low_w=20.0, f_high_w=1024.0):
    """Genera un strain blanqueado para una clase dada."""
    m1 = float(rng.uniform(10, 80))
    m2 = float(rng.uniform(10, m1))
    d  = float(rng.uniform(100, 1000))
    M  = m1 + m2
    dt = 1.0 / sample_rate

    strain, _ = _generate_base_waveform(m1, m2, d, duration, sample_rate,
                                         f_lower, approximant)
    htilde = np.fft.rfft(strain)
    freqs  = np.fft.rfftfreq(len(strain), d=dt)

    if theory_class == 0:
        pass
    elif theory_class == 1:
        htilde = _inject_standard_siren(htilde, freqs, H0_true=float(rng.uniform(60,80)))
    elif theory_class == 2:
        htilde = _inject_qnm_subdominant(htilde, freqs, mode="21", rng=rng)
    elif theory_class == 3:
        htilde = _inject_qnm_subdominant(htilde, freqs, mode="33", rng=rng)
    elif theory_class == 4:
        htilde = _inject_pn_deformation(htilde, freqs, M, delta_phi_hat=float(rng.uniform(0.03,0.10)))
    elif theory_class == 5:
        htilde = _inject_liv(htilde, freqs, d, alpha=3.0, A_alpha_m=float(rng.uniform(1e8,1e9)))
    elif theory_class == 6:
        htilde = _inject_scalar_tensor(htilde, freqs, M, omega_BD_eff=float(rng.uniform(40,80)))
    elif theory_class == 7:
        htilde = _inject_graviton_mass(htilde, freqs, d, mg_eV=1.27e-23*7e21)
    elif theory_class == 8:
        htilde = _inject_chern_simons(htilde, freqs, d, kappa_gpc=float(rng.uniform(0.5,1.2)))
    elif theory_class == 9:
        htilde = _inject_liv(htilde, freqs, d, alpha=2.0, A_alpha_m=float(rng.uniform(5e9,2e10)))
    elif theory_class == 10:
        htilde = _inject_liv(htilde, freqs, d, alpha=4.0, A_alpha_m=float(rng.uniform(5e9,2e10)))
    elif theory_class == 11:
        htilde = _inject_qnm_subdominant(htilde, freqs, mode="33", rng=rng)
        htilde *= 1.3
    elif theory_class == 12:
        htilde = _inject_gup(htilde, freqs, beta_gup=float(rng.uniform(5e-3,2e-2)))

    strain_mod = np.fft.irfft(htilde, n=len(strain))
    return _whiten_strain(strain_mod, dt, f_low=f_low_w, f_high=f_high_w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class",    type=int, default=4200)
    p.add_argument("--n-val-per-class",type=int, default=600)
    p.add_argument("--approximant",    type=str, default="IMRPhenomPv2")
    p.add_argument("--n-components",   type=int, default=12)
    p.add_argument("--chunk-size",     type=int, default=100,
                   help="Eventos por chunk (controla uso de RAM)")
    p.add_argument("--out-dir",        type=str, default="data/dataset_full")
    p.add_argument("--seed",           type=int, default=42)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_classes   = len(THEORY_CLASSES)
    n_train     = args.n_per_class * n_classes
    n_val       = args.n_val_per_class * n_classes
    n_total     = args.n_per_class + args.n_val_per_class
    rng         = np.random.default_rng(args.seed)

    print(f"Generando {n_train} train + {n_val} val eventos")
    print(f"Aproximante: {args.approximant}")
    print(f"Chunk size:  {args.chunk_size} eventos")

    # ---- Fase 1: generar strains blanqueados en chunks, ajustar PCA ----
    ipca = IncrementalPCA(n_components=args.n_components)

    # Guardar strains blanqueados temporalmente por clase
    tmp_dir = out / "tmp_white"
    tmp_dir.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    for cls in range(n_classes):
        print(f"  Generando clase {cls}: {THEORY_CLASSES[cls]}", flush=True)
        n_cls_chunks = (n_total + args.chunk_size - 1) // args.chunk_size
        cls_whites = []

        for chunk_i in range(n_cls_chunks):
            n_this = min(args.chunk_size, n_total - chunk_i * args.chunk_size)
            chunk  = np.zeros((n_this, 4 * 4096))  # 4s × 4096 Hz = 16384
            for j in range(n_this):
                chunk[j] = generate_one_white(cls, rng, args.approximant)
            cls_whites.append(chunk)

            # Ajuste incremental del PCA (solo con train)
            if chunk_i * args.chunk_size < args.n_per_class:
                n_train_in_chunk = min(n_this,
                    args.n_per_class - chunk_i * args.chunk_size)
                if n_train_in_chunk >= args.n_components:
                    ipca.partial_fit(chunk[:n_train_in_chunk])

        # Guardar strains de esta clase
        np.save(tmp_dir / f"cls_{cls:02d}.npy",
                np.concatenate(cls_whites, axis=0))
        elapsed = time.perf_counter() - t0
        rate    = (cls + 1) * n_total / elapsed
        eta     = (n_classes - cls - 1) * n_total / rate / 60
        print(f"    {n_total} eventos en {elapsed:.0f}s  ETA: {eta:.0f}min", flush=True)

    # ---- Fase 2: proyectar con PCA y construir splits ----
    print("\nProyectando con PCA...", flush=True)

    X_train_list, y_train_list = [], []
    X_val_list,   y_val_list   = [], []

    for cls in range(n_classes):
        whites = np.load(tmp_dir / f"cls_{cls:02d}.npy")
        scores = ipca.transform(whites)[:, :args.n_components]
        X_train_list.append(scores[:args.n_per_class])
        y_train_list.append(np.full(args.n_per_class, cls, dtype=int))
        X_val_list.append(scores[args.n_per_class:args.n_per_class + args.n_val_per_class])
        y_val_list.append(np.full(args.n_val_per_class, cls, dtype=int))
        # Liberar memoria
        (tmp_dir / f"cls_{cls:02d}.npy").unlink()

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_val   = np.concatenate(X_val_list)
    y_val   = np.concatenate(y_val_list)

    # Escalar a [-1, 1]
    scaler  = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_train = np.clip(scaler.fit_transform(X_train), -1.0, 1.0)
    X_val   = np.clip(scaler.transform(X_val),       -1.0, 1.0)

    # ---- Guardar ----
    np.save(out / "X_train.npy", X_train)
    np.save(out / "y_train.npy", y_train)
    np.save(out / "X_val.npy",   X_val)
    np.save(out / "y_val.npy",   y_val)

    with open(out / "pca.pkl",    "wb") as f: pickle.dump(ipca,   f)
    with open(out / "scaler.pkl", "wb") as f: pickle.dump(scaler, f)

    # Limpiar tmp
    tmp_dir.rmdir()

    total = time.perf_counter() - t0
    var_exp = ipca.explained_variance_ratio_[:args.n_components].sum()
    print(f"\n✅ Dataset guardado en {out}/")
    print(f"   X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"   X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"   Varianza explicada PCA: {var_exp*100:.1f}%")
    print(f"   Tiempo total: {total/3600:.1f}h")


if __name__ == "__main__":
    main()