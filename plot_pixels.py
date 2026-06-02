#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_pixels.py — Genera imágenes JPEG pixel-by-pixel de archivos TPF (Target Pixel Files)
de misiones TESS/Kepler/K2 usando lightkurve. Solo guarda las imágenes, sin ventanas gráficas.

Uso:
    python plot_pixels.py <patron_o_archivo> [...]

Ejemplos:
    python plot_pixels.py archivo.fits
    python plot_pixels.py tpfs/*.fits
    python plot_pixels.py "tpfs/*.fits"
    python plot_pixels.py sector1/*.fits sector2/*.fits
"""

import sys
import os
import glob
import matplotlib
matplotlib.use("Agg")  # backend no interactivo: nunca abre ventanas
import matplotlib.pyplot as plt
import lightkurve as lk
from tqdm import tqdm


def process_tpf_file(fits_filename: str):
    """
    Carga un TPF, genera la imagen de píxeles y la guarda como JPEG.

    Args:
        fits_filename: Ruta al archivo FITS con datos TPF.

    Returns:
        Tupla (éxito, resultado, shape):
            - éxito (bool): True si el proceso fue exitoso.
            - resultado (str): Ruta al JPEG generado, o mensaje de error.
            - shape (tuple | None): (n_cuadros, filas, cols) del TPF, o None si hubo error.
    """
    base_name = os.path.splitext(os.path.basename(fits_filename))[0]
    jpg_filename = f"tpf_{base_name}.jpg"

    try:
        tpf = lk.read(fits_filename)
    except FileNotFoundError:
        return False, f"Archivo no encontrado: {fits_filename}", None
    except OSError as e:
        return False, f"Error de lectura FITS: {e}", None
    except Exception as e:
        return False, f"Error inesperado al leer el archivo: {e}", None

    if not hasattr(tpf, "plot_pixels"):
        return False, "El archivo no contiene datos de píxeles TPF (¿es una curva de luz?)", None

    try:
        tpf.plot_pixels(show_flux=True, yscale="log")
        fig = plt.gcf()
        fig.savefig(jpg_filename, dpi=100, bbox_inches="tight", format="jpg")
        plt.close(fig)
    except OSError as e:
        plt.close("all")
        return False, f"Error al guardar '{jpg_filename}': {e}", None
    except Exception as e:
        plt.close("all")
        return False, f"Error al generar o guardar la figura: {e}", None

    return True, jpg_filename, tpf.shape


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Expandir patrones glob (maneja wildcards entre comillas y rutas directas)
    input_files = []
    for pattern in sys.argv[1:]:
        expanded = glob.glob(pattern)
        input_files.extend(expanded if expanded else [pattern])

    # Validar archivos: deben existir y ser FITS
    valid_files, skipped = [], []
    for f in input_files:
        if not os.path.exists(f):
            skipped.append((f, "no encontrado"))
        elif not f.lower().endswith((".fits", ".fit")):
            skipped.append((f, "no es un archivo FITS"))
        else:
            valid_files.append(f)

    for f, reason in skipped:
        print(f"Ignorado ({reason}): {os.path.basename(f)}")

    if not valid_files:
        print("\nNo hay archivos FITS válidos para procesar.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("GENERANDO IMÁGENES TPF")
    print(f"{'='*60}")
    print(f"Archivos a procesar: {len(valid_files)}")
    print(f"{'='*60}\n")

    successful, failed = [], []

    for fits_filename in tqdm(valid_files, desc="Progreso", unit="archivo"):
        success, result, shape = process_tpf_file(fits_filename)
        if success:
            successful.append((fits_filename, result, shape))
            tqdm.write(f"  ✓ {os.path.basename(result)}")
        else:
            failed.append((fits_filename, result))
            tqdm.write(f"  ✗ {os.path.basename(fits_filename)}: {result}")

    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")
    print(f"Generadas: {len(successful)}/{len(valid_files)}")

    if successful:
        print("\nImágenes creadas:")
        for i, (_, img, shape) in enumerate(successful[:10], 1):
            print(f"  {i:2d}. {os.path.basename(img)} ({shape[1]}x{shape[2]} píxeles)")
        if len(successful) > 10:
            print(f"  ... y {len(successful) - 10} más")

    if failed:
        print(f"\nFallos ({len(failed)}/{len(valid_files)}):")
        for i, (orig, error) in enumerate(failed[:5], 1):
            print(f"  {i:2d}. {os.path.basename(orig)}: {error}")
        if len(failed) > 5:
            print(f"  ... y {len(failed) - 5} más")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
