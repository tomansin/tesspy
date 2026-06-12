#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
norm_lc.py - Normalizador interactivo de curvas de luz TESS.

Uso:
    norm_lc.py <curva.csv>
"""

import bisect
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse
import termios
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import SpanSelector


def load_lightcurve(filename):
    try:
        with open(filename) as f:
            header = f.readline().strip().split(',')
        data     = np.loadtxt(filename, delimiter=',', skiprows=1)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        time     = data[:, 0]
        flux     = data[:, 1]
        flux_err = data[:, 2]
        n_nan = int(np.sum(~np.isfinite(flux)))
        print(f"LOADED LC FROM {filename}")
        print(f"  Puntos:  {len(time)}")
        print(f"  Tiempo:  {np.nanmin(time):.4f} - {np.nanmax(time):.4f} TBJD")
        print(f"  Columnas: {header}")
        if n_nan:
            print(f"  Puntos con flujo NaN: {n_nan} (excluidos de los ajustes)")
        return time, flux, flux_err, data, header
    except Exception as e:
        print(f"Error cargando curva de luz: {e}")
        return None, None, None, None, None


def sigma_clip_legendre(t_norm, flux, flux_err, order, xi_lo=2.0, xi_hi=3.0):
    """Ajuste iterativo de polinomio Legendre con sigma clipping. Retorna (coeffs, t_c, f_c) o (None, None, None)."""
    valid = np.isfinite(t_norm) & np.isfinite(flux)
    t_norm, flux, flux_err = t_norm[valid], flux[valid], flux_err[valid]
    if len(t_norm) <= order + 1:
        return None, None, None
    try:
        w = 1.0 / np.where(flux_err > 0, flux_err, 1.0)
        sigma0 = np.std(flux)
        coeffs = np.polynomial.legendre.legfit(t_norm, flux, order, w=w)
        baseline = np.polynomial.legendre.legval(t_norm, coeffs)
        res = flux - baseline
        keep = (res < xi_hi * sigma0) & (res > -xi_lo * sigma0)
        t_c, f_c, w_c = t_norm[keep], flux[keep], w[keep]

        for _ in range(50):
            if len(t_c) <= order + 1:
                break
            sigma = np.std(f_c)
            c = np.polynomial.legendre.legfit(t_c, f_c, order, w=w_c)
            b = np.polynomial.legendre.legval(t_c, c)
            r = f_c - b
            new_keep = (r < xi_hi * sigma) & (r > -xi_lo * sigma)
            if not np.any(~new_keep):
                coeffs = c
                break
            t_c, f_c, w_c = t_c[new_keep], f_c[new_keep], w_c[new_keep]
            coeffs = c

        return coeffs, t_c, f_c
    except Exception as e:
        print(f"  Error en ajuste: {e}")
        return None, None, None


def plot_norm_viewer(time, flux, flux_err, data_all, header, filename):
    """Normalizador interactivo: LC + polinomio arriba, normalizada abajo."""

    plt.rcParams['keymap.yscale'] = [k for k in plt.rcParams['keymap.yscale'] if k != 'l']
    plt.rcParams['keymap.back']   = [k for k in plt.rcParams['keymap.back']   if k != 'left']
    plt.rcParams['keymap.forward'] = [k for k in plt.rcParams['keymap.forward'] if k != 'right']

    fig = plt.figure(figsize=(14, 8))
    gs  = GridSpec(2, 1, hspace=0.4)
    ax_lc   = fig.add_subplot(gs[0])
    ax_norm = fig.add_subplot(gs[1], sharex=ax_lc)

    tmin, tmax = np.nanmin(time), np.nanmax(time)
    t_norm = (time - tmin) / (tmax - tmin)

    # Discontinuidades: lista ordenada de tiempos reales
    discontinuities = []   # sorted list of t_real
    add_history     = []   # orden de adición, para deshacer con 'd'
    disc_vlines     = []   # vlines en ax_lc, en el mismo orden que discontinuities

    # Parámetros por segmento: un dict {'order', 'xi_lo', 'xi_hi'} por segmento
    # Siempre hay len(discontinuities)+1 segmentos
    def _default_params():
        return {'order': 1, 'xi_lo': 2.0, 'xi_hi': 3.0}

    seg_params = [_default_params()]   # inicia con un único segmento

    active_seg = [0]    # índice del segmento activo
    disc_mode  = [False]
    pending    = [None]
    view_state = {'init': False}   # conserva zoom/pan entre refits

    # Artists del ajuste (reconstruidos en refit)
    poly_artists   = []
    culled_artists = []
    seg_coeffs     = []   # coeffs o None, por segmento

    # ── Setup inicial ─────────────────────────────────────────────────────────

    ax_lc.plot(time, flux, 'k.', markersize=3, alpha=0.8, label='Curva original')
    ax_lc.set_ylabel('Flujo (e-/s)', fontsize=10)
    ax_lc.grid(True, alpha=0.3, linestyle='--')

    ax_norm.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax_norm.set_xlabel('Tiempo (TBJD)', fontsize=10)
    ax_norm.set_ylabel('Flujo normalizado', fontsize=10)
    ax_norm.set_title('Curva normalizada', fontsize=10)
    ax_norm.grid(True, alpha=0.3, linestyle='--')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_segments():
        bounds = [tmin] + discontinuities + [tmax]
        return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    def _update_title():
        p = seg_params[active_seg[0]]
        n_seg = len(seg_params)
        disc_tag = '  [DISCONTINUIDAD]' if disc_mode[0] else ''
        ax_lc.set_title(
            f'{os.path.basename(filename)}  |  '
            f'Seg {active_seg[0] + 1}/{n_seg}  |  '
            f'orden={p["order"]}  |  '
            f'σ_inf={p["xi_lo"]:.2f}  σ_sup={p["xi_hi"]:.2f}'
            f'{disc_tag}',
            fontsize=10
        )

    def _clear_fit_artists():
        for a in poly_artists:
            if a is not None:
                a.remove()
        poly_artists.clear()
        for a in culled_artists:
            if a is not None:
                a.remove()
        culled_artists.clear()
        seg_coeffs.clear()

    COLORS = plt.cm.tab10.colors

    def refit():
        keep_view = view_state['init']
        if keep_view:
            xlim      = ax_lc.get_xlim()
            ylim_lc   = ax_lc.get_ylim()
            ylim_norm = ax_norm.get_ylim()

        _clear_fit_artists()
        segments = get_segments()

        for i, (t0, t1) in enumerate(segments):
            mask = (time >= t0) & (time <= t1)
            p = seg_params[i]
            if np.sum(mask) <= p['order'] + 1:
                seg_coeffs.append(None)
                poly_artists.append(None)
                culled_artists.append(None)
                print(f"  Segmento {i+1}: muy pocos puntos, ignorado.")
                continue

            c, t_c, f_c = sigma_clip_legendre(
                t_norm[mask], flux[mask], flux_err[mask],
                p['order'], p['xi_lo'], p['xi_hi']
            )
            seg_coeffs.append(c)

            if c is None:
                poly_artists.append(None)
                culled_artists.append(None)
                continue

            color = COLORS[i % len(COLORS)]
            lw    = 1.5 if i == active_seg[0] else 0.8
            alpha = 1.0 if i == active_seg[0] else 0.5

            tn0    = (t0 - tmin) / (tmax - tmin)
            tn1    = (t1 - tmin) / (tmax - tmin)
            x_fine = np.linspace(tn0, tn1, 500)
            y_fine = np.polynomial.legendre.legval(x_fine, c)
            t_fine = x_fine * (tmax - tmin) + tmin
            line,  = ax_lc.plot(t_fine, y_fine, '-', color=color,
                                linewidth=lw, alpha=alpha,
                                label=f'Seg {i+1} ord={p["order"]}')
            poly_artists.append(line)

            sc = None
            if t_c is not None and len(t_c) > 0:
                t_c_real = t_c * (tmax - tmin) + tmin
                sc = ax_lc.scatter(t_c_real, f_c, color=color,
                                   s=10, alpha=0.6, zorder=3,
                                   marker='+')
            culled_artists.append(sc)

        ax_lc.legend(loc='best', fontsize=8)
        if keep_view:
            update_norm(ylim_norm)
            # ax_norm.clear() puede resetear el eje x compartido (sharex)
            ax_lc.set_xlim(xlim)
            ax_lc.set_ylim(ylim_lc)
        else:
            ax_lc.relim()
            ax_lc.autoscale_view()
            update_norm(None)
            view_state['init'] = True

    def update_highlight():
        """Actualiza solo el grosor/alpha de las líneas sin refitear."""
        for i, art in enumerate(poly_artists):
            if art is None:
                continue
            art.set_linewidth(2.5 if i == active_seg[0] else 1.5)
            art.set_alpha(1.0 if i == active_seg[0] else 0.5)
        _update_title()
        fig.canvas.draw_idle()

    def build_continuum():
        segments = get_segments()
        if not seg_coeffs or all(c is None for c in seg_coeffs):
            return None
        cont = np.ones(len(time))
        for i, (t0, t1) in enumerate(segments):
            if i >= len(seg_coeffs) or seg_coeffs[i] is None:
                continue
            mask = (time >= t0) & (time <= t1)
            cont[mask] = np.polynomial.legendre.legval(t_norm[mask], seg_coeffs[i])
        return cont

    def update_norm(keep_ylim=None):
        ax_norm.clear()
        ax_norm.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax_norm.set_xlabel('Tiempo (TBJD)', fontsize=10)
        ax_norm.set_ylabel('Flujo normalizado', fontsize=10)
        ax_norm.set_title('Curva normalizada', fontsize=10)
        ax_norm.grid(True, alpha=0.3, linestyle='--')

        cont = build_continuum()
        if cont is not None:
            norm_flux = flux / cont
            ax_norm.plot(time, norm_flux, 'k.', markersize=3, alpha=0.8,
                         label='Normalizada')
            for t_d in discontinuities:
                ax_norm.axvline(t_d, color='blue', linestyle='--',
                                linewidth=1.2, alpha=0.7)
            ax_norm.legend(loc='best', fontsize=8)
            if keep_ylim is not None:
                ax_norm.set_ylim(keep_ylim)
            else:
                valid = norm_flux[np.isfinite(norm_flux)]
                if len(valid) > 0:
                    lo = np.percentile(valid, 1)
                    hi = np.percentile(valid, 99)
                    margin = (hi - lo) * 0.2
                    ax_norm.set_ylim(lo - margin, hi + margin)

        _update_title()
        fig.canvas.draw_idle()

    # ── Discontinuidades ──────────────────────────────────────────────────────

    def add_discontinuity(t_val):
        idx = bisect.bisect_left(discontinuities, t_val)
        discontinuities.insert(idx, t_val)
        add_history.append(t_val)
        vl = ax_lc.axvline(t_val, color='blue', linestyle='--',
                            linewidth=1.2, alpha=0.7)
        disc_vlines.insert(idx, vl)
        # El segmento idx se divide: insertar copia de sus parámetros en idx+1
        seg_params.insert(idx + 1, dict(seg_params[idx]))
        print(f"  Discontinuidad en t={t_val:.4f}  ({len(discontinuities)} total)")
        refit()

    def remove_last_discontinuity():
        if not add_history:
            return
        t_rem = add_history.pop()
        idx   = discontinuities.index(t_rem)
        discontinuities.pop(idx)
        disc_vlines.pop(idx).remove()
        # Fusionar: conservar seg_params[idx], eliminar seg_params[idx+1]
        seg_params.pop(idx + 1)
        active_seg[0] = min(active_seg[0], len(seg_params) - 1)
        print(f"  Discontinuidad t={t_rem:.4f} eliminada. Quedan {len(discontinuities)}.")
        refit()

    def on_click(event):
        if not disc_mode[0] or event.inaxes not in (ax_lc, ax_norm):
            return
        if event.button != 1:
            return
        t_val = event.xdata
        if t_val is None or not (tmin < t_val < tmax):
            return
        disc_mode[0] = False
        add_discontinuity(t_val)

    # ── Guardado ──────────────────────────────────────────────────────────────

    def _target_part():
        base = os.path.splitext(os.path.basename(filename))[0]
        for prefix in ('tess-corrected_', 'tess-uncorrected_', 'tess-normalized_'):
            if prefix in base:
                return base.split(prefix)[-1]
        return base

    def do_save():
        cont = build_continuum()
        if cont is None:
            print("  No hay normalización que guardar.")
            return
        try:
            norm_flux = flux / cont
            norm_err  = flux_err / cont
            os.makedirs('lcs', exist_ok=True)
            out       = f'lcs/tess-normalized_{_target_part()}.csv'
            out_data  = np.column_stack([data_all, norm_flux, norm_err])
            out_header = ','.join(header) + ',norm_flux,norm_flux_err'
            np.savetxt(out, out_data, delimiter=',',
                       header=out_header, comments='', fmt='%.17g')
            print(f"  Curva normalizada guardada: {os.path.abspath(out)}")
        except Exception as e:
            print(f"  Error guardando: {e}")

    # ── Teclado ───────────────────────────────────────────────────────────────

    def on_key(event):
        if event.key == 'q':
            pending[0] = 'quit'
            plt.close(fig)

        elif event.key == ' ':
            n = len(seg_params)
            active_seg[0] = (active_seg[0] + 1) % n
            print(f"  Segmento activo: {active_seg[0] + 1}/{n}")
            update_highlight()

        elif event.key == 'a':
            disc_mode[0] = not disc_mode[0]
            state = 'ACTIVADO' if disc_mode[0] else 'DESACTIVADO'
            print(f"  Modo discontinuidad {state} — click en el gráfico para colocarla")
            _update_title()
            fig.canvas.draw_idle()

        elif event.key == 'd':
            remove_last_discontinuity()

        elif event.key in ('+', '='):
            seg_params[active_seg[0]]['order'] = min(
                seg_params[active_seg[0]]['order'] + 1, 18)
            print(f"  Seg {active_seg[0]+1} orden: {seg_params[active_seg[0]]['order']}")
            refit()

        elif event.key == '-':
            seg_params[active_seg[0]]['order'] = max(
                seg_params[active_seg[0]]['order'] - 1, 1)
            print(f"  Seg {active_seg[0]+1} orden: {seg_params[active_seg[0]]['order']}")
            refit()

        elif event.key == 'down':
            p = seg_params[active_seg[0]]
            p['xi_lo'] = max(round(p['xi_lo'] - 0.25, 2), 0.25)
            print(f"  Seg {active_seg[0]+1} σ_inf: {p['xi_lo']:.2f}")
            refit()

        elif event.key == 'up':
            p = seg_params[active_seg[0]]
            p['xi_lo'] = round(p['xi_lo'] + 0.25, 2)
            print(f"  Seg {active_seg[0]+1} σ_inf: {p['xi_lo']:.2f}")
            refit()

        elif event.key == 'left':
            p = seg_params[active_seg[0]]
            p['xi_hi'] = max(round(p['xi_hi'] - 0.25, 2), 0.25)
            print(f"  Seg {active_seg[0]+1} σ_sup: {p['xi_hi']:.2f}")
            refit()

        elif event.key == 'right':
            p = seg_params[active_seg[0]]
            p['xi_hi'] = round(p['xi_hi'] + 0.25, 2)
            print(f"  Seg {active_seg[0]+1} σ_sup: {p['xi_hi']:.2f}")
            refit()

        elif event.key == 'z':
            do_save()

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', on_click)

    fig.suptitle(os.path.basename(filename), fontsize=11, fontweight='bold')

    refit()

    print("\n" + "="*50)
    print("TESS LC NORMALIZER")
    print("="*50)
    print("  espacio   ciclar al siguiente segmento")
    print("  a         activar modo discontinuidad (click para colocar)")
    print("  d         eliminar última discontinuidad")
    print("  +/-       subir/bajar orden del segmento activo")
    print("  ↓/↑       bajar/subir σ inferior del segmento activo")
    print("  ←/→       bajar/subir σ superior del segmento activo")
    print("  z         guardar normalizada  ->  lcs/tess-normalized_*.csv")
    print("  q         cerrar (pregunta si guardar)")
    print("="*50)

    plt.show()

    # ── Vista de la curva normalizada ─────────────────────────────────────────
    cont = build_continuum()
    if pending[0] != 'quit' or cont is None:
        return

    norm_flux = flux / cont
    norm_err  = flux_err / cont

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    fig2.suptitle(os.path.basename(filename) + '  —  curva normalizada',
                  fontsize=11, fontweight='bold')

    removed_ranges  = []   # lista de (t0, t1) excluidos
    span2           = [None]
    sel2_active     = [False]
    pending2        = [None]
    view2_state     = {'init': False}   # conserva zoom/pan entre redraws

    def _build_save_mask():
        mask = np.ones(len(time), dtype=bool)
        for t0, t1 in removed_ranges:
            mask &= ~((time >= t0) & (time <= t1))
        return mask

    def _redraw2():
        keep_view = view2_state['init']
        if keep_view:
            xlim2 = ax2.get_xlim()
            ylim2 = ax2.get_ylim()

        ax2.clear()
        mask = _build_save_mask()
        ax2.plot(time[mask], norm_flux[mask], 'k.', markersize=3, alpha=0.8,
                 label='Normalizada')
        if np.any(~mask):
            ax2.plot(time[~mask], norm_flux[~mask], '.', color='lightgray',
                     markersize=3, alpha=0.6, label='Eliminada')
        for t_d in discontinuities:
            ax2.axvline(t_d, color='blue', linestyle='--', linewidth=1.2, alpha=0.7)
        for t0, t1 in removed_ranges:
            ax2.axvspan(t0, t1, alpha=0.15, color='red', zorder=0)
        ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax2.set_xlabel('Tiempo (TBJD)', fontsize=10)
        ax2.set_ylabel('Flujo normalizado', fontsize=10)
        sel_tag = '  [SELECCION]' if sel2_active[0] else ''
        n_rem = len(removed_ranges)
        rem_tag = f'  |  {n_rem} rango(s) eliminado(s)' if n_rem else ''
        ax2.set_title(f'Curva normalizada{rem_tag}{sel_tag}', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='best', fontsize=8)
        if keep_view:
            ax2.set_xlim(xlim2)
            ax2.set_ylim(ylim2)
        else:
            view2_state['init'] = True
        fig2.canvas.draw_idle()

    _redraw2()
    fig2.tight_layout()

    def do_save2():
        mask = _build_save_mask()
        try:
            out_data   = np.column_stack([data_all[mask], norm_flux[mask], norm_err[mask]])
            out_header = ','.join(header) + ',norm_flux,norm_flux_err'
            os.makedirs('lcs', exist_ok=True)
            out = f'lcs/tess-normalized_{_target_part()}.csv'
            np.savetxt(out, out_data, delimiter=',',
                       header=out_header, comments='', fmt='%.17g')
            n_kept = int(np.sum(mask))
            print(f"  Guardado: {os.path.abspath(out)}  ({n_kept} puntos)")
        except Exception as e:
            print(f"  Error guardando: {e}")

    def onselect2(xmin, xmax):
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        removed_ranges.append((xmin, xmax))
        print(f"  Rango eliminado: [{xmin:.4f}, {xmax:.4f}]  ({len(removed_ranges)} total)")
        _redraw2()

    def toggle_sel2():
        sel2_active[0] = not sel2_active[0]
        if sel2_active[0]:
            if span2[0] is not None:
                span2[0].disconnect_events()
            span2[0] = SpanSelector(
                ax2, onselect2, 'horizontal',
                useblit=True,
                props=dict(alpha=0.2, facecolor='red'),
                interactive=True,
                drag_from_anywhere=True
            )
            print("  Selección ACTIVADA")
        else:
            if span2[0] is not None:
                span2[0].disconnect_events()
                span2[0] = None
            print("  Selección DESACTIVADA")
        _redraw2()

    def on_key2(event):
        if event.key == 'q':
            pending2[0] = 'quit'
            if span2[0] is not None:
                span2[0].disconnect_events()
            plt.close(fig2)
        elif event.key == 'a':
            toggle_sel2()
        elif event.key == 'e':
            if removed_ranges:
                removed_ranges.pop()
                print(f"  Último rango restaurado. Quedan {len(removed_ranges)}.")
                _redraw2()
        elif event.key == 'z':
            do_save2()

    fig2.canvas.mpl_connect('key_press_event', on_key2)

    print("\n" + "="*50)
    print("VISTA NORMALIZADA")
    print("="*50)
    print("  a         activar/desactivar selección de rangos a eliminar")
    print("  e         restaurar último rango eliminado")
    print("  z         guardar  ->  lcs/tess-normalized_*.csv")
    print("  q         cerrar (pregunta si guardar)")
    print("="*50)
    plt.show()

    if pending2[0] == 'quit':
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        resp = input("\n  Guardar curva normalizada? [S/n]: ").strip().lower()
        if resp not in ('n', 'no'):
            do_save2()


def main():
    parser = argparse.ArgumentParser(description='TESS light curve normalizer')
    parser.add_argument('filename', help='CSV light curve file')
    args = parser.parse_args()

    time, flux, flux_err, data_all, header = load_lightcurve(args.filename)
    if time is None:
        sys.exit(1)

    plot_norm_viewer(time, flux, flux_err, data_all, header, args.filename)


if __name__ == "__main__":
    main()
