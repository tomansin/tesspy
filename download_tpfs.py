#!/home/tansin/.conda/envs/lkurve/bin/python
# -*- coding: utf-8 -*-
"""
download_tpfs.py — Descarga Target Pixel Files (TPF) de TESS para un objeto dado
usando el servicio TESScut a través de lightkurve.

Los archivos FITS se guardan en el subdirectorio 'tpfs/' con el formato:
    tess-tpf_<target>_<sector>.fits

Uso:
    python download_tpfs.py <target> [--cutout-size N [M]]
    python download_tpfs.py            (solicita el target interactivamente)

Ejemplos:
    python download_tpfs.py "TIC 261136679"
    python download_tpfs.py "HD 209458" --cutout-size 15
    python download_tpfs.py "TOI-700" --cutout-size 10 20
"""

import os
import sys
import time
import argparse
from tqdm import tqdm
import lightkurve as lk


def _safe_name(target: str) -> str:
    """Convierte un nombre de target en una cadena segura para usar como nombre de archivo."""
    return "".join(c for c in target if c.isalnum() or c in ("-", "_")).rstrip()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Descarga TPFs de TESS usando TESScut.",
        add_help=True,
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Nombre del objeto (ej: 'TIC 261136679', 'TOI-700'). "
             "Si se omite, se solicita de forma interactiva.",
    )
    parser.add_argument(
        "--cutout-size",
        nargs="+",
        type=int,
        default=[21],
        metavar="N",
        help="Tamaño del recorte en píxeles. Un valor para cuadrado (ej: 15) "
             "o dos para rectangular (ej: 15 20). Por defecto: 21.",
    )
    args = parser.parse_args()

    # Validar cutout_size
    if len(args.cutout_size) == 1:
        cutout_size = (args.cutout_size[0], args.cutout_size[0])
    elif len(args.cutout_size) == 2:
        cutout_size = tuple(args.cutout_size)
    else:
        parser.error("--cutout-size acepta 1 o 2 valores enteros.")

    if any(s < 1 or s > 99 for s in cutout_size):
        parser.error("El tamaño del cutout debe estar entre 1 y 99 píxeles.")

    return args.target, cutout_size


def main():
    raw_target, cutout_size = _parse_args()

    # Obtener el target desde argumentos o de forma interactiva
    if raw_target:
        target = raw_target.strip()
        print(f"Target: {target}")
    else:
        target = input("Ingrese nombre del target: ").strip()

    if not target:
        print("Error: No se proporcionó un target válido.")
        sys.exit(1)

    # Buscar el target en el archivo TESScut (MAST)
    try:
        print(f"Buscando '{target}' en TESScut...")
        search_result = lk.search_tesscut(target)
    except Exception as e:
        print(f"Error de conexión o búsqueda en MAST: {e}")
        sys.exit(1)

    if len(search_result) == 0:
        print(f"No se encontró '{target}' en TESScut. Verifica el nombre del objeto.")
        sys.exit(1)

    # Mostrar lo que se encontró y pedir confirmación
    print(f"\n{'='*60}")
    print("OBSERVACIONES ENCONTRADAS")
    print(f"{'='*60}")
    print(search_result)
    print(f"\nConfiguración de descarga:")
    print(f"  Cutout size: {cutout_size[0]}x{cutout_size[1]} píxeles")
    print(f"  Directorio:  {os.path.abspath('tpfs')}")
    print(f"{'='*60}")

    try:
        confirm = input("\n¿Desea continuar con la descarga? [s/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelado.")
        sys.exit(0)

    if confirm not in ("s", "si", "sí", "y", "yes"):
        print("Descarga cancelada.")
        sys.exit(0)

    # Crear directorio de salida
    try:
        os.makedirs("tpfs", exist_ok=True)
    except OSError as e:
        print(f"Error al crear el directorio 'tpfs': {e}")
        sys.exit(1)

    # Descargar cada TPF individualmente con barra de progreso
    safe_target = _safe_name(target)
    sector_list = []
    failed_count = 0

    print("Iniciando descarga de TPFs...")
    try:
        with tqdm(total=len(search_result), desc="Descargando TPFs", unit="archivo") as pbar:
            for i, search_item in enumerate(search_result):
                try:
                    tpf = search_item.download(cutout_size=cutout_size, quality_bitmask="default")

                    sector = tpf.get_keyword("SECTOR")
                    filename = f"tpfs/tess-tpf_{safe_target}_{sector}.fits"
                    tpf.to_fits(output_fn=filename, overwrite=True)

                    sector_list.append(sector)
                    pbar.set_postfix(sector=sector, refresh=True)

                    time.sleep(0.5)  # evitar saturar el servidor MAST

                except OSError as e:
                    failed_count += 1
                    tqdm.write(f"  ✗ Observación {i+1}: error al guardar el archivo — {e}")
                except Exception as e:
                    failed_count += 1
                    tqdm.write(f"  ✗ Observación {i+1}: error al descargar — {e}")
                finally:
                    pbar.update(1)

    except KeyboardInterrupt:
        print("\nDescarga interrumpida por el usuario.")
        sys.exit(1)

    # Resumen final
    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")

    if sector_list:
        print(f"Descargados: {len(sector_list)}/{len(search_result)} archivos")
        print(f"Sectores:    {sorted(sector_list)}")
        print(f"Directorio:  {os.path.abspath('tpfs')}")
    else:
        print("No se pudo descargar ningún archivo TPF.")

    if failed_count:
        print(f"Fallos:      {failed_count}/{len(search_result)}")

    print(f"{'='*60}")

    if not sector_list:
        sys.exit(1)


if __name__ == "__main__":
    main()
