#!/home/tansin/.conda/envs/lkurve/bin/python
# -*- coding: utf-8 -*-
"""
correct_lc.py - Corrector de fondo de cielo para TPF TESS.

Uso:
    correct_lc.py <archivo.fits>
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse
from matplotlib.gridspec import GridSpec
from matplotlib import patches
import lightkurve as lk
from lightkurve.correctors import RegressionCorrector, DesignMatrix


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


def _target_part(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    return base.split('tess-tpf_')[-1] if 'tess-tpf_' in base else base


def load_aperture_file(aper_file, tpf_shape):
    """Carga apertura desde CSV. Retorna máscara booleana o None."""
    try:
        data = np.loadtxt(aper_file, delimiter=',', skiprows=1)
        mask = np.zeros((tpf_shape[1], tpf_shape[2]), dtype=bool)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        for x, y in data.astype(int):
            if 0 <= x < tpf_shape[2] and 0 <= y < tpf_shape[1]:
                mask[y, x] = True
        print(f"  Apertura cargada: {int(np.sum(mask))} pixel(s) desde {aper_file}")
        return mask
    except Exception as e:
        print(f"  Error cargando apertura '{aper_file}': {e}")
        return None


def compute_sky_mask(tpf, threshold=0.1):
    target_mask = tpf.create_threshold_mask(threshold=threshold, reference_pixel='center')
    sky_mask = (~target_mask).copy()
    print(f"  Máscara de cielo: {int(np.sum(sky_mask))} pixel(s)  (threshold={threshold:.2f})")
    return sky_mask


def compute_correction(tpf, sky_mask, aperture_mask, pca_number=3):
    """Retorna (dm, lc_raw, lc_corrected) o (None, None, None) si falla."""
    try:
        dm = DesignMatrix(tpf.flux[:, sky_mask], name='regressors').pca(pca_number)
        lc_raw = tpf.to_lightcurve(aperture_mask=aperture_mask)
        lc_clean = lc_raw.remove_nans()

        valid = np.isfinite(lc_raw.flux.value)
        if np.sum(valid) < len(lc_raw.flux):
            dm_clean = DesignMatrix(dm.values[valid], name='regressors')
        else:
            dm_clean = dm

        corrector = RegressionCorrector(lc_clean)
        _ = corrector.correct(dm_clean.append_constant())
        model = corrector.model_lc
        model -= np.percentile(model.flux, 5)
        lc_corrected = lc_clean - model

        return dm, lc_clean, lc_corrected
    except Exception as e:
        print(f"  Error en PCA/corrección: {e}")
        return None, None, None


def compute_correction_median(tpf, sky_mask, aperture_mask):
    """Retorna (bkg, lc_raw, lc_corrected) usando mediana de regresores."""
    try:
        regressors = tpf.flux[:, sky_mask].value
        bkg = np.median(regressors, axis=1)
        bkg -= np.percentile(bkg, 5)

        lc_raw = tpf.to_lightcurve(aperture_mask=aperture_mask)
        lc_clean = lc_raw.remove_nans()

        valid = np.isfinite(lc_raw.flux.value)
        lc_corrected = lc_clean.copy()
        lc_corrected.flux = (lc_clean.flux.value - bkg[valid]) * lc_clean.flux.unit

        return bkg, lc_clean, lc_corrected
    except Exception as e:
        print(f"  Error en corrección por mediana: {e}")
        return None, None, None


def plot_sky_viewer(tpf, filename):
    """Visualizador: TPF izquierda, PCA derecha arriba, LC derecha abajo."""

    plt.rcParams['keymap.yscale'] = [k for k in plt.rcParams['keymap.yscale'] if k != 'l']

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 2, width_ratios=[1, 2], hspace=0.4, wspace=0.3)
    ax_tpf = fig.add_subplot(gs[:, 0], projection=tpf.wcs)
    ax_pca = fig.add_subplot(gs[0, 1])
    ax_lc  = fig.add_subplot(gs[1, 1])

    # Estado mutable
    sky_mask         = [None]
    aperture_mask    = [None]
    dm_state         = [None]
    lc_raw_state     = [None]
    lc_cor_state     = [None]
    tpf_im           = [None]
    tpf_patches      = []
    pending          = [None]
    pca_number       = [3]
    threshold        = [0.1]
    selection_active = [False]
    corr_mode        = ['pca']
    bkg_state        = [None]

    tgt = _target_part(filename)
    aper_file = f'apers/tess-aperture_{tgt}.csv'

    # Inicialización
    sky_mask[0] = compute_sky_mask(tpf, threshold[0])

    if os.path.exists(aper_file):
        aperture_mask[0] = load_aperture_file(aper_file, tpf.flux.shape)
    else:
        print(f"  Apertura no encontrada: {aper_file}")

    if sky_mask[0] is not None and aperture_mask[0] is not None:
        dm_state[0], lc_raw_state[0], lc_cor_state[0] = compute_correction(
            tpf, sky_mask[0], aperture_mask[0], pca_number[0])

    # ── Funciones de dibujo ───────────────────────────────────────────────────

    def draw_tpf():
        frame_data = tpf.flux[0].value
        if tpf_im[0] is None:
            tpf_im[0] = ax_tpf.imshow(frame_data, cmap='viridis', origin='lower',
                                       aspect='equal')
            ax_tpf.set_xlabel('RA')
            ax_tpf.set_ylabel('Dec')
        else:
            tpf_im[0].set_data(frame_data)
            tpf_im[0].set_clim(np.nanpercentile(frame_data, 5),
                               np.nanpercentile(frame_data, 99))

        for p in tpf_patches:
            p.remove()
        tpf_patches.clear()

        # Máscara de cielo (sombreado rojo)
        if sky_mask[0] is not None:
            ny, nx = sky_mask[0].shape
            for y in range(ny):
                for x in range(nx):
                    if sky_mask[0][y, x]:
                        rect = patches.Rectangle(
                            (x - 0.5, y - 0.5), 1, 1,
                            linewidth=1.5, edgecolor='C3', facecolor='none',
                            hatch='//', alpha=0.4
                        )
                        ax_tpf.add_patch(rect)
                        tpf_patches.append(rect)

        # Apertura (azul)
        if aperture_mask[0] is not None:
            ny, nx = aperture_mask[0].shape
            for y in range(ny):
                for x in range(nx):
                    if aperture_mask[0][y, x]:
                        rect = patches.Rectangle(
                            (x - 0.5, y - 0.5), 1, 1,
                            linewidth=2, edgecolor='blue', facecolor='none', alpha=0.7
                        )
                        ax_tpf.add_patch(rect)
                        tpf_patches.append(rect)

        n_sky = int(np.sum(sky_mask[0])) if sky_mask[0] is not None else 0
        n_ap  = int(np.sum(aperture_mask[0])) if aperture_mask[0] is not None else 0
        sel_tag = '  [SELECCION]' if selection_active[0] else ''
        ax_tpf.set_title(
            f'TPF  |  Cielo: {n_sky} px  |  Apertura: {n_ap} px{sel_tag}\n'
            f'threshold={threshold[0]:.2f}  PCA={pca_number[0]}',
            fontsize=10
        )

    def draw_pca():
        ax_pca.clear()
        if corr_mode[0] == 'median':
            bkg = bkg_state[0]
            if bkg is None:
                ax_pca.text(0.5, 0.5, 'Sin fondo calculado',
                            ha='center', va='center', transform=ax_pca.transAxes)
                ax_pca.set_xticks([])
                ax_pca.set_yticks([])
            else:
                ax_pca.plot(tpf.time.value, bkg, '.', markersize=2, color='C1')
                ax_pca.set_xlabel('Tiempo (TBJD)', fontsize=9)
                ax_pca.set_ylabel('Fondo (e-/s)', fontsize=9)
                n_sky = int(np.sum(sky_mask[0])) if sky_mask[0] is not None else 0
                ax_pca.set_title(f'Fondo por mediana  —  {n_sky} px de cielo', fontsize=10)
                ax_pca.grid(True, alpha=0.3, linestyle='--')
        else:
            dm = dm_state[0]
            if dm is None:
                ax_pca.text(0.5, 0.5, 'Sin máscara de cielo / apertura',
                            ha='center', va='center', transform=ax_pca.transAxes)
                ax_pca.set_xticks([])
                ax_pca.set_yticks([])
            else:
                offsets = np.arange(dm.values.shape[1]) * 0.2
                ax_pca.plot(tpf.time.value, dm.values + offsets, '.', markersize=2)
                ax_pca.set_xlabel('Tiempo (TBJD)', fontsize=9)
                ax_pca.set_ylabel('Flujo (norm.)', fontsize=9)
                n_sky = int(np.sum(sky_mask[0])) if sky_mask[0] is not None else 0
                ax_pca.set_title(f'Componentes PCA ({pca_number[0]})  —  {n_sky} px de cielo',
                                 fontsize=10)
                ax_pca.grid(True, alpha=0.3, linestyle='--')

    def draw_lc():
        ax_lc.clear()
        lc_raw = lc_raw_state[0]
        lc_cor = lc_cor_state[0]

        if lc_raw is None:
            ax_lc.text(0.5, 0.5, 'Sin curva de luz\n(cargue una apertura)',
                       ha='center', va='center', transform=ax_lc.transAxes)
            ax_lc.set_xticks([])
            ax_lc.set_yticks([])
            return

        t = lc_raw.time.value
        f = lc_raw.flux.value
        valid = np.isfinite(t) & np.isfinite(f)
        ax_lc.plot(t[valid], f[valid], '-', color='grey', alpha=0.6, lw=1.2,
                   label='Original')

        if lc_cor is not None:
            t2 = lc_cor.time.value
            f2 = lc_cor.flux.value
            valid2 = np.isfinite(t2) & np.isfinite(f2)
            ax_lc.plot(t2[valid2], f2[valid2], '-', color='C0', alpha=1, lw=1.2,
                       label='Corregida')

        ax_lc.set_xlabel('Tiempo (TBJD)', fontsize=9)
        ax_lc.set_ylabel('Flujo (e-/s)', fontsize=9)
        mode_tag = 'Mediana' if corr_mode[0] == 'median' else f'PCA={pca_number[0]}'
        ax_lc.set_title(f'Curva de luz  [{mode_tag}]', fontsize=10)
        ax_lc.grid(True, alpha=0.3, linestyle='--')
        ax_lc.legend(loc='best', fontsize=8)

    def refresh():
        draw_tpf()
        draw_pca()
        draw_lc()
        fig.canvas.draw_idle()

    refresh()

    def recompute():
        if sky_mask[0] is None or aperture_mask[0] is None:
            return
        if corr_mode[0] == 'pca':
            dm_state[0], lc_raw_state[0], lc_cor_state[0] = compute_correction(
                tpf, sky_mask[0], aperture_mask[0], pca_number[0])
            bkg_state[0] = None
        else:
            bkg_state[0], lc_raw_state[0], lc_cor_state[0] = compute_correction_median(
                tpf, sky_mask[0], aperture_mask[0])
            dm_state[0] = None

    # ── Helpers de guardado ───────────────────────────────────────────────────

    def do_save_lc():
        if lc_cor_state[0] is None:
            print("  No hay curva corregida que guardar.")
            return
        try:
            os.makedirs('lcs', exist_ok=True)
            out = f'lcs/tess-corrected_{tgt}.csv'
            lc = lc_cor_state[0].copy()
            lc['sector'] = tpf.sector
            lc.to_csv(path_or_buf=out, overwrite=True)
            print(f"  Curva corregida guardada: {os.path.abspath(out)}")
        except Exception as e:
            print(f"  Error guardando curva: {e}")

    # ── Clic en TPF ───────────────────────────────────────────────────────────

    def on_tpf_click(event):
        if not selection_active[0] or event.inaxes != ax_tpf:
            return
        if event.button != 1:
            return
        if sky_mask[0] is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        ny, nx = sky_mask[0].shape
        if not (0 <= x < nx and 0 <= y < ny):
            return

        sky_mask[0][y, x] = not sky_mask[0][y, x]
        action = 'agregado' if sky_mask[0][y, x] else 'eliminado'
        print(f"  Pixel ({x}, {y}) {action} de máscara de cielo"
              f"  ({int(np.sum(sky_mask[0]))} px)")

        recompute()
        refresh()

    # ── Teclado ───────────────────────────────────────────────────────────────

    def on_key(event):
        if event.key == 'q':
            pending[0] = 'quit'
            plt.close(fig)

        elif event.key == 'a':
            selection_active[0] = not selection_active[0]
            state = 'ACTIVADA' if selection_active[0] else 'DESACTIVADA'
            print(f"  Selección de máscara de cielo {state}")
            draw_tpf()
            fig.canvas.draw_idle()

        elif event.key == 'z':
            do_save_lc()

        elif event.key == 'm':
            if corr_mode[0] == 'pca':
                corr_mode[0] = 'median'
                print("  Modo: corrección por MEDIANA")
            else:
                corr_mode[0] = 'pca'
                print(f"  Modo: corrección por PCA ({pca_number[0]} componentes)")
            recompute()
            refresh()

        elif event.key in ('+', '='):
            threshold[0] = round(threshold[0] + 0.05, 3)
            print(f"  Threshold: {threshold[0]:.2f}")
            sky_mask[0] = compute_sky_mask(tpf, threshold[0])
            recompute()
            refresh()

        elif event.key == '-':
            threshold[0] = round(threshold[0] - 0.05, 3)
            print(f"  Threshold: {threshold[0]:.2f}")
            sky_mask[0] = compute_sky_mask(tpf, threshold[0])
            recompute()
            refresh()

        elif event.key == 'up':
            pca_number[0] = min(pca_number[0] + 1, 6)
            print(f"  PCA componentes: {pca_number[0]}")
            recompute()
            refresh()

        elif event.key == 'down':
            pca_number[0] = max(pca_number[0] - 1, 1)
            print(f"  PCA componentes: {pca_number[0]}")
            recompute()
            refresh()

    fig.canvas.mpl_connect('button_press_event', on_tpf_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    fig.suptitle(os.path.basename(filename), fontsize=11, fontweight='bold')

    print("\n" + "="*50)
    print("TESS SKY CORRECTOR")
    print("="*50)
    print("  a         activar/desactivar modo selección de máscara de cielo")
    print("            (en modo selección: click para agregar/quitar pixel del cielo)")
    print("  m         alternar modo corrección: PCA  <->  Mediana")
    print("  +/-       aumentar/disminuir threshold de máscara de cielo")
    print("  ↑/↓       más/menos componentes PCA")
    print("  z         guardar curva corregida  ->  lcs/tess-corrected_*.csv")
    print("  q         cerrar (pregunta si guardar)")
    print("="*50)

    plt.show()

    if pending[0] == 'quit':
        if lc_cor_state[0] is not None:
            resp = input("\n  Guardar curva corregida? [S/n]: ").strip().lower()
            if resp not in ('n', 'no'):
                do_save_lc()


def main():
    parser = argparse.ArgumentParser(description='TESS sky background corrector')
    parser.add_argument('filename', help='FITS TPF file')
    args = parser.parse_args()

    tpf = load_tpf(args.filename)
    if tpf is None:
        sys.exit(1)

    plot_sky_viewer(tpf, args.filename)


if __name__ == "__main__":
    main()
