#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_lc.py - Visualizador interactivo de TPF TESS en formato FITS para generar apertura y curva de luz.

Uso:
    make_lc.py <archivo.fits>
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse
from matplotlib.gridspec import GridSpec
from matplotlib import patches
import lightkurve as lk
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord, Angle


def load_tpf(filename):
    try:
        tpf = lk.read(filename)
        print(f"LOADED TPF FROM {filename}")
        print(f"  Target: {tpf.targetid}")
        print(f"  Frames: {len(tpf.time)}")
        print(f"  Shape: {tpf.flux.shape[1]} x {tpf.flux.shape[2]} pixels")
        return tpf
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None
    except Exception as e:
        print(f"Error loading TPF: {e}")
        return None


def query_gaia(tpf):
    """
    Consulta Gaia DR3 en la region del TPF.
    Retorna la tabla de fuentes con Gmag < 14, o None si falla.
    """
    print("  Consultando Gaia DR3...", flush=True)
    try:
        Vizier.ROW_LIMIT = -1
        Vizier.COLUMNS = ['RA_ICRS', 'DE_ICRS', 'Gmag', 'Source']

        center = SkyCoord(tpf.ra, tpf.dec, frame='icrs', unit='deg')
        radius_arcsec = (np.max(tpf.flux.shape[1:]) - 2) * 21  # ~21"/pixel TESS
        result = Vizier.query_region(center,
                                     radius=Angle(radius_arcsec, 'arcsec'),
                                     catalog='I/355/gaiadr3')
        if not result or len(result) == 0:
            print("  Gaia: ninguna fuente encontrada en esta region.")
            return None

        sources = result[0]
        bright = sources[sources['Gmag'] < 14]
        if len(bright) == 0:
            print("  Gaia: ninguna fuente con Gmag < 14.")
            return None

        print(f"  Gaia: {len(bright)} fuente(s) con Gmag < 14.")
        return bright
    except Exception as e:
        print(f"  Error consultando Gaia: {e}")
        return None


def plot_tpf_viewer(tpf, filename):
    """Visualizador interactivo: TPF a la izquierda, curva de luz a la derecha."""

    # Liberar teclas ocupadas por matplotlib
    plt.rcParams['keymap.yscale'] = [k for k in plt.rcParams['keymap.yscale'] if k != 'l']
    # 'b' no está en ningún keymap de matplotlib, nada que liberar

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, width_ratios=[1, 2], wspace=0.3)
    ax_tpf = fig.add_subplot(gs[0], projection=tpf.wcs)
    ax_lc  = fig.add_subplot(gs[1])

    # Estado mutable
    current_frame    = [0]
    aperture_mask    = [np.zeros_like(tpf.flux[0].value, dtype=bool)]
    lc_cache         = [None]
    selection_active = [False]
    gaia_sources     = [None]   # tabla Gaia cargada (None = no cargada aun)
    show_gaia        = [False]  # True = mostrar circulos en el TPF
    gaia_pix_coords  = [None]   # lista de (px, py, source_id, gmag) en coordenadas pixel
    hover_annot      = [None]   # anotacion de hover
    tpf_im           = [None]   # handle del imshow del TPF
    tpf_patches      = []       # patches activos (apertura + Gaia)
    pending          = [None]   # accion pendiente al cerrar

    def compute_lc():
        try:
            lc_cache[0] = tpf.to_lightcurve(aperture_mask=aperture_mask[0])
        except Exception as e:
            print(f"  Error calculando light curve: {e}")
            lc_cache[0] = None

    def draw_tpf():
        # Actualizar imagen sin limpiar el eje (preserva la proyeccion WCS)
        frame_data = tpf.flux[current_frame[0]].value
        if tpf_im[0] is None:
            tpf_im[0] = ax_tpf.imshow(frame_data, cmap='viridis', origin='lower',
                                       aspect='equal')
            ax_tpf.set_xlabel('RA')
            ax_tpf.set_ylabel('Dec')
            hover_annot[0] = ax_tpf.annotate(
                '', xy=(0, 0), xytext=(8, 8), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.85),
                fontsize=8, visible=False, zorder=10
            )
        else:
            tpf_im[0].set_data(frame_data)
            tpf_im[0].set_clim(np.nanpercentile(frame_data, 5),
                               np.nanpercentile(frame_data, 99))

        # Eliminar patches anteriores
        for p in tpf_patches:
            p.remove()
        tpf_patches.clear()

        # Apertura
        alpha = 0.7 if selection_active[0] else 0.35
        ny, nx = aperture_mask[0].shape
        for y in range(ny):
            for x in range(nx):
                if aperture_mask[0][y, x]:
                    rect = patches.Rectangle(
                        (x - 0.5, y - 0.5), 1, 1,
                        linewidth=2, edgecolor='red', facecolor='none', hatch='//', alpha=alpha
                    )
                    ax_tpf.add_patch(rect)
                    tpf_patches.append(rect)

        # Fuentes Gaia
        gaia_pix_coords[0] = []
        if show_gaia[0] and gaia_sources[0] is not None:
            for source in gaia_sources[0]:
                try:
                    px, py = tpf.wcs.all_world2pix(
                        [[source['RA_ICRS'], source['DE_ICRS']]], 0)[0]
                    size = 20 / (1.5 ** source['Gmag'])
                    circ = patches.Circle(
                        (px, py), radius=size,
                        edgecolor='C3', facecolor='r', alpha=0.6, linewidth=1.5
                    )
                    ax_tpf.add_patch(circ)
                    tpf_patches.append(circ)
                    gaia_pix_coords[0].append((px, py, str(source['Source']), float(source['Gmag'])))
                except Exception:
                    pass

        mode_tag = '  [SELECCION]' if selection_active[0] else ''
        gaia_tag = '  [GAIA]' if show_gaia[0] else ''
        ax_tpf.set_title(
            f'TPF  frame {current_frame[0]}/{len(tpf.time)-1}'
            f'{mode_tag}{gaia_tag}\n'
            f't = {tpf.time[current_frame[0]].value:.4f}',
            fontsize=10
        )

    def draw_lc():
        ax_lc.clear()
        lc = lc_cache[0]
        if lc is None:
            ax_lc.text(0.5, 0.5, 'No hay curva de luz',
                       ha='center', va='center', transform=ax_lc.transAxes)
        else:
            ax_lc.plot(lc.time.value, lc.flux.value, 'k-', linewidth=0.8,
                       label='Apertura actual')

            t_cur = tpf.time[current_frame[0]].value
            idx = np.argmin(np.abs(lc.time.value - t_cur))
            ax_lc.axvline(t_cur, color='red', linestyle='--', alpha=0.6, linewidth=1)
            ax_lc.plot(lc.time.value[idx], lc.flux.value[idx],
                       'ro', markersize=5, zorder=5)

            n_pix = int(np.sum(aperture_mask[0]))
            ax_lc.set_title(f'Curva de luz  ({n_pix} pixel(s) en apertura)', fontsize=10)
            ax_lc.set_xlabel('Tiempo (TBJD)')
            ax_lc.set_ylabel('Flujo (e-/s)')
            ax_lc.grid(True, alpha=0.3, linestyle='--')
            ax_lc.legend(loc='best', fontsize=8)

    def refresh():
        draw_tpf()
        draw_lc()
        fig.canvas.draw_idle()

    # Calcular curva inicial y dibujar
    compute_lc()
    refresh()

    # ── Helpers de guardado ───────────────────────────────────────────────────

    def _target_part():
        base = os.path.splitext(os.path.basename(filename))[0]
        return base.split('tess-tpf_')[-1] if 'tess-tpf_' in base else base

    def do_save_aperture():
        if aperture_mask[0] is None or not np.any(aperture_mask[0]):
            print("  Apertura vacía, nada que guardar.")
            return
        try:
            os.makedirs('apers', exist_ok=True)
            out = f'apers/tess-aperture_{_target_part()}.csv'
            coords = np.array([(x, y)
                               for y, x in zip(*np.where(aperture_mask[0]))],
                              dtype=int)
            np.savetxt(out, coords, delimiter=',', header='x,y', fmt='%d', comments='')
            print(f"  Apertura guardada: {os.path.abspath(out)}")
        except Exception as e:
            print(f"  Error guardando apertura: {e}")

    def do_save_lc():
        if lc_cache[0] is None:
            print("  No hay curva de luz que guardar.")
            return
        try:
            os.makedirs('lcs', exist_ok=True)
            out = f'lcs/tess-uncorrected_{_target_part()}.csv'
            lc = lc_cache[0].copy()
            lc['sector'] = tpf.sector
            lc.to_csv(path_or_buf=out, overwrite=True)
            print(f"  Curva de luz guardada: {os.path.abspath(out)}")
        except Exception as e:
            print(f"  Error guardando curva de luz: {e}")

    # ── Clic en TPF ───────────────────────────────────────────────────────────
    def on_tpf_click(event):
        if not selection_active[0] or event.inaxes != ax_tpf:
            return
        if event.button != 1:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        ny, nx = aperture_mask[0].shape
        if not (0 <= x < nx and 0 <= y < ny):
            return

        aperture_mask[0][y, x] = not aperture_mask[0][y, x]
        n_pix = int(np.sum(aperture_mask[0]))
        print(f"  Pixel ({x}, {y}) {'agregado' if aperture_mask[0][y, x] else 'eliminado'}"
              f"  ({n_pix} pixel(s) en apertura)")

        compute_lc()
        refresh()

    # ── Teclado ───────────────────────────────────────────────────────────────
    def on_key(event):
        if event.key == 'q':
            pending[0] = 'quit'
            plt.close(fig)

        elif event.key == 'a':
            selection_active[0] = not selection_active[0]
            state = 'ACTIVADA' if selection_active[0] else 'DESACTIVADA'
            print(f"  Seleccion {state}")
            draw_tpf()
            fig.canvas.draw_idle()

        elif event.key == 'b':
            if gaia_sources[0] is None:
                # Primera vez: consultar Gaia (bloquea brevemente)
                gaia_sources[0] = query_gaia(tpf)
                if gaia_sources[0] is not None:
                    show_gaia[0] = True
            else:
                # Alternar visibilidad
                show_gaia[0] = not show_gaia[0]
                state = 'VISIBLE' if show_gaia[0] else 'OCULTA'
                print(f"  Gaia {state}")
            draw_tpf()
            fig.canvas.draw_idle()

        elif event.key == 'l':
            if not selection_active[0] and current_frame[0] < len(tpf.time) - 1:
                current_frame[0] += 1
                refresh()

        elif event.key == 'j':
            if not selection_active[0] and current_frame[0] > 0:
                current_frame[0] -= 1
                refresh()

        elif event.key == 'x':
            do_save_aperture()

        elif event.key == 'z':
            do_save_lc()

    # ── Hover sobre fuentes Gaia ──────────────────────────────────────────────
    def on_hover(event):
        annot = hover_annot[0]
        if annot is None:
            return
        if event.inaxes != ax_tpf or not show_gaia[0] or not gaia_pix_coords[0]:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        mx, my = event.xdata, event.ydata
        best_dist, best = np.inf, None
        for px, py, sid, gmag in gaia_pix_coords[0]:
            d = np.hypot(mx - px, my - py)
            if d < best_dist:
                best_dist, best = d, (px, py, sid, gmag)

        if best is not None and best_dist < 1.0:
            px, py, sid, gmag = best
            annot.xy = (px, py)
            annot.set_text(f'DR3 {sid}\nGmag={gmag:.2f}')
            annot.set_visible(True)
        else:
            annot.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_tpf_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('motion_notify_event', on_hover)

    fig.suptitle(os.path.basename(filename), fontsize=11, fontweight='bold')

    print("\n" + "="*50)
    print("TESS TPF VIEWER")
    print("="*50)
    print("  j / l     frame anterior / siguiente")
    print("  a         activar/desactivar modo seleccion de pixeles")
    print("            (en modo seleccion: click para agregar/quitar pixel)")
    print("  b         cargar/mostrar/ocultar fuentes Gaia DR3 (Gmag<14)")
    print("            (hover sobre una fuente para ver su DR3 ID y Gmag)")
    print("  x         guardar apertura  ->  apers/tess-aperture_*.csv")
    print("  z         guardar curva de luz  ->  lcs/tess-uncorrected_*.csv")
    print("  q         cerrar (pregunta si guardar)")
    print("="*50)

    plt.show()

    # Figura cerrada: preguntar sobre guardado
    if pending[0] == 'quit':
        if np.any(aperture_mask[0]):
            resp = input("\n  Guardar apertura? [S/n]: ").strip().lower()
            if resp not in ('n', 'no'):
                do_save_aperture()
        if lc_cache[0] is not None:
            resp = input("  Guardar curva de luz? [S/n]: ").strip().lower()
            if resp not in ('n', 'no'):
                do_save_lc()


def main():
    parser = argparse.ArgumentParser(description='TESS TPF interactive viewer')
    parser.add_argument('filename', help='FITS TPF file')
    args = parser.parse_args()

    tpf = load_tpf(args.filename)
    if tpf is None:
        sys.exit(1)

    plot_tpf_viewer(tpf, args.filename)


if __name__ == "__main__":
    main()
