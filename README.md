# QNIM

Proyecto mínimo para ejecutar la pipeline experimental de QNIM con el entrypoint principal.

## Uso rápido

```bash
python scripts/generate_results.py --mode sim
python scripts/generate_results.py --mode ibm
```

## Perfil 27 qubits

```bash
python scripts/generate_results.py --from-disk --mode sim --profile ibm_27q_proposal --max-iter 1
```

Perfiles disponibles en [config/experiment_profiles.json](config/experiment_profiles.json).

## Estructura mínima

- [scripts/generate_results.py](scripts/generate_results.py): entrypoint principal.
- [src](src): lógica de aplicación e infraestructura.
- [config](config): configuración y perfiles.
- [requirements.txt](requirements.txt): dependencias mínimas.
