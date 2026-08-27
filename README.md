# tesspy

Pipeline de procesamiento de fotometría TESS. Descarga Target Pixel Files (TPF), genera aperturas y curvas de luz de forma interactiva, corrige el fondo de cielo y normaliza las curvas resultantes.

## Flujo de trabajo

```
download.py  →  aperture.py  →  background.py  →  normalize.py
```

## Scripts

### `download.py`
Descarga TPFs de TESS para un objeto dado usando TESScut (vía lightkurve). Guarda los archivos FITS en `tpf/`.

```bash
python download.py "HD 209458"
python download.py "TIC 261136679" --cutout-size 15
python download.py "TOI-700" --cutout-size 10 20
```

### `aperture.py`
Visualizador interactivo de TPF. Permite seleccionar píxeles de apertura manualmente (con soporte de catálogo Gaia DR3) y genera la curva de luz cruda. Guarda la apertura en `apers/` y la curva en `lcs/`.

```bash
python aperture.py tpf/tess-tpf_HD209458_10.fits
```

### `background.py`
Corrector de fondo de cielo para TPFs TESS. Carga una apertura existente, estima y sustrae el fondo mediante regresión, y guarda la curva corregida en `lcs/`.

```bash
python background.py tpf/tess-tpf_HD209458_10.fits
```

### `normalize.py`
Normalizador interactivo de curvas de luz. Permite seleccionar regiones de continuo gráficamente, ajusta un polinomio de Legendre con sigma-clipping y guarda la curva normalizada en `lcs/`.

```bash
python normalize.py lcs/tess-corrected_HD209458_10.csv
```

### `pixels.py`
Genera imágenes JPEG de los píxeles de uno o varios TPFs sin abrir ventanas gráficas.

```bash
python pixels.py tpf/*.fits
```

## Estructura de directorios

```
tesspy/
├── tpf/      # TPFs descargados (.fits)
├── apers/    # Máscaras de apertura (.csv)
└── lcs/      # Curvas de luz (.csv): crudas, corregidas y normalizadas
```

## Dependencias

- [lightkurve](https://lightkurve.github.io/)
- astropy
- astroquery
- numpy
- matplotlib
- tqdm
