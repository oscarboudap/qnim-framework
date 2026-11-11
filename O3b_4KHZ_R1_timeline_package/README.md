# O3b_4KHZ_R1 — paquete de preparación de línea temporal

Este paquete prepara la selección de segmentos de 4 s para train/validation/test.
No contiene strain HDF5: contiene el descargador reproducible de los metadatos oficiales
de GWOSC. Al ejecutarlo con conexión a Internet descargará la línea temporal real de O3b.

## Rango O3b
- GPS inicial: 1256655618
- GPS final: 1269363618 (intervalo semiabierto)
- Duración: 12 708 000 s
- Dataset de strain previsto: O3b_4KHZ_R1 (4096 Hz)

## Objetivos de datos (no son disponibilidad confirmada)
- Train: 7 h 13 min = 6 495 ventanas de 4 s
- Validation: 43 min = 645 ventanas de 4 s
- Test: 1 h 27 min = 1 305 ventanas de 4 s
- Total: 9 h 23 min = 8 445 ventanas de 4 s

## Uso
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_o3b_timeline.py --detectors H1 L1 --event-margin 2
```

El script:
1. Descarga el JSON de todas las líneas temporales O3b desde GWOSC.
2. Guarda JSON completo y ficheros ASCII por flag.
3. Construye intervalos candidatos coincidentes H1+L1 usando DATA,
   vetos CBC CAT1/CAT2/CAT3 y exclusión de hardware injections CBC.
4. Descarga el catálogo GWTC-3-confident e identifica ventanas cercanas a eventos
   (margen configurable, por defecto 2 s).
5. Escribe CSV de ventanas de 4 s y un resumen de duraciones.

## Importante sobre calidad y eventos
- Las categorías CAT son flags de problemas; CAT1 es la severidad más alta.
- El script aplica de forma conservadora los vetos CAT1, CAT2 y CAT3 para crear
  candidatos. Para el análisis definitivo hay que justificar el esquema de veto
  según la definición de calidad CBC de O3b y la metodología del proyecto.
- La exclusión temporal alrededor de eventos GWTC-3 es una máscara de análisis,
  no una regla oficial universal. Revisa el margen y la lista de eventos.
- Los segmentos son semiabiertos [inicio, fin), no se solapan y se alinean a
  múltiplos de 4 s dentro de los intervalos válidos.
- No se garantiza que existan 8 445 ventanas válidas. El script informa la
  disponibilidad real; no inventa ni rellena datos.

## Fuentes oficiales
- GWOSC O3b: https://gwosc.org/O3/O3b/
- Timeline: https://gwosc.org/timeline/
- API documentación: https://gwosc.org/api/v1/docs/
- Dataset strain: https://gwosc.org/api/v2/datasets/O3b_4KHZ_R1/strain-files?format=api
- GWTC-3: https://gwosc.org/GWTC-3/

Licencia de los datos: consulta los términos y pautas de atribución de GWOSC.
