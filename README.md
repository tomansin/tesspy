# tesspy

Pipeline de procesamiento de fotometría TESS. Descarga Target Pixel Files (TPF), genera aperturas y curvas de luz de forma interactiva, corrige el fondo de cielo y normaliza las curvas resultantes.

## Flujo de trabajo

```
download_tpfs.py  →  make_lc.py  →  correct_lc.py  →  norm_lc.py
```

## Scripts

### `download_tpfs.py`
Descarga TPFs de TESS para un objeto dado usando TESScut (vía lightkurve). Guarda los archivos FITS en `tpf/`.

```bash
python download_tpfs.py "HD 209458"
python download_tpfs.py "TIC 261136679" --cutout-size 15
python download_tpfs.py "TOI-700" --cutout-size 10 20
```

### `make_lc.py`
Visualizador interactivo de TPF. Permite seleccionar píxeles de apertura manualmente (con soporte de catálogo Gaia DR3) y genera la curva de luz cruda. Guarda la apertura en `apers/` y la curva en `lcs/`.

```bash
python make_lc.py tpf/tess-tpf_HD209458_10.fits
```

### `correct_lc.py`
Corrector de fondo de cielo para TPFs TESS. Carga una apertura existente, estima y sustrae el fondo mediante regresión, y guarda la curva corregida en `lcs/`.

```bash
python correct_lc.py tpf/tess-tpf_HD209458_10.fits
```

### `norm_lc.py`
Normalizador interactivo de curvas de luz. Permite seleccionar regiones de continuo gráficamente, ajusta un polinomio de Legendre con sigma-clipping y guarda la curva normalizada en `lcs/`.

```bash
python norm_lc.py lcs/tess-corrected_HD209458_10.csv
```

### `plot_pixels.py`
Genera imágenes JPEG de los píxeles de uno o varios TPFs sin abrir ventanas gráficas.

```bash
python plot_pixels.py tpf/*.fits
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
