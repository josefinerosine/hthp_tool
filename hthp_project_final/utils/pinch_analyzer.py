"""
Pinch Analyzer
==============
Performs pinch analysis and creates T-Q diagrams.
Receives streams directly from PinchParamBuilder.get_pinch_params().

Algorithm:
  1. Hot composite curve:  built from HOT stream breakpoints only, Q=0 at T_min_hot.
     Cold composite curve: built from COLD stream breakpoints only, Q=0 at T_min_cold.
     Both curves have strictly positive slope in T-Q space.

  2. The cold curve is shifted horizontally by Q_C_min so that the minimum vertical
     distance between the two curves equals ΔT_min (graphical pinch method).
     Q_C_min is found via binary search.

  3. The pinch point is the (Q, T_hot, T_cold) triple where T_hot − T_cold is
     minimised in the overlap zone.  By construction this minimum equals ΔT_min.

  4. Energy balance (after shift):
       Cold curve spans  [Q_C_min,  Q_C_min + Q_cold_total]
       Hot  curve spans  [0,        Q_hot_total]
       Q_H_min   = max(0,  Q_C_min + Q_cold_total − Q_hot_total)
       Q_recovery = Q_hot_total − Q_C_min

  5. Grand Composite Curve is derived from the Problem Table Algorithm using the
     same Q_H_min obtained above.

Usage:
    from pinch_params import PinchParamBuilder
    from pinch_analyzer import PinchAnalyzer

    builder = PinchParamBuilder(params, delta_T_min=10.0)
    pinch_input = builder.get_pinch_params()

    analyzer = PinchAnalyzer(**pinch_input)
    analyzer.run()
    analyzer.plot()        # T-Q diagram
    analyzer.plot_gcc()    # Grand Composite Curve
    analyzer.plot_all()    # both side-by-side
    analyzer.print_results()

Version: 3.0  (fully graphical composite-curve method)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PinchStream:
    """Process stream – identical to pinch_params.PinchStream."""
    name:        str
    stream_type: str    # 'hot' | 'cold'
    T_supply:    float  # °C
    T_target:    float  # °C
    CP:          float  # kW/K
    Q:           float  # kW
    note:        str = ''


@dataclass
class PinchResults:
    T_pinch_hot:     float = None  # real pinch temperature, hot side  [°C]
    T_pinch_cold:    float = None  # real pinch temperature, cold side [°C]
    T_pinch_shifted: float = None  # shifted pinch temperature          [°C]
    Q_H_min:         float = None  # minimum hot utility  [kW]
    Q_C_min:         float = None  # minimum cold utility [kW]
    Q_recovery:      float = None  # maximum recoverable heat [kW]
    delta_T_min:     float = None


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class PinchAnalyzer:
    """
    Pinch analysis calculator.

    Parameters
    ----------
    streams     : list of PinchStream objects (or dicts with same keys)
    delta_T_min : minimum temperature approach at the pinch [K]
    warnings    : optional warning list from PinchParamBuilder
    """

    def __init__(
        self,
        streams:     list,
        delta_T_min: float = 10.0,
        warnings:    list  = None,
    ):
        self.streams: list[PinchStream] = []
        for s in streams:
            if isinstance(s, dict):
                self.streams.append(PinchStream(**s))
            else:
                self.streams.append(s)

        self.delta_T_min = delta_T_min
        self.ext_warnings = warnings or []
        self.results: PinchResults = PinchResults(delta_T_min=delta_T_min)

        # Composite curve arrays (populated after run())
        self._Q_hot:  np.ndarray = None   # ascending, Q=0 at T_min_hot
        self._T_hot:  np.ndarray = None   # ascending
        self._Q_cold: np.ndarray = None   # ascending, starts at Q_C_min
        self._T_cold: np.ndarray = None   # ascending
        self._Q_gcc:  np.ndarray = None
        self._T_gcc:  np.ndarray = None   # shifted temperatures, descending

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> 'PinchAnalyzer':
        """Run the complete pinch analysis."""
        self._run_algorithm()
        self._build_gcc()
        return self

    def print_results(self):
        r = self.results
        print("\n" + "═" * 60)
        print("  PINCH ANALYSIS RESULTS")
        print("═" * 60)
        print(f"  ΔT_min              : {self.delta_T_min:.1f} K")
        if r.T_pinch_hot is not None:
            print(f"  Pinch (hot side)    : {r.T_pinch_hot:.2f} °C")
            print(f"  Pinch (cold side)   : {r.T_pinch_cold:.2f} °C")
        print(f"  Q_H,min  (Heating)  : {r.Q_H_min:.2f} kW")
        print(f"  Q_C,min  (Cooling)  : {r.Q_C_min:.2f} kW")
        print(f"  Q_recovery (max)    : {r.Q_recovery:.2f} kW")
        print()
        print(f"  {'#':<3} {'Type':<5} {'Name':<34} "
              f"{'Ts [°C]':>8} {'Tt [°C]':>8} {'CP [kW/K]':>10} {'Q [kW]':>8}")
        print("  " + "─" * 78)
        for i, s in enumerate(self.streams, 1):
            arr = '→' if s.stream_type == 'hot' else '←'
            print(f"  {i:<3} {s.stream_type.upper():<5} {s.name:<34} "
                  f"{s.T_supply:>8.2f} {arr} {s.T_target:>8.2f} "
                  f"{s.CP:>10.4f} {s.Q:>8.1f}")
        if self.ext_warnings:
            print("\n  Warnings:")
            for w in self.ext_warnings:
                print(f"  ⚠  {w}")
        print("═" * 60)

    def plot(self, ax=None, save_path: str = None, figsize=(11, 7)):
        """T-Q diagram with composite curves."""
        if self._Q_hot is None:
            raise RuntimeError("Call run() before plot().")
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        self._draw_tq_diagram(ax)
        if standalone:
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {save_path}")
            else:
                plt.show()
        return fig if standalone else ax

    def plot_gcc(self, ax=None, save_path: str = None, figsize=(7, 7)):
        """Grand Composite Curve."""
        if self._Q_gcc is None:
            raise RuntimeError("Call run() before plot_gcc().")
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=figsize)
        self._draw_gcc(ax)
        if standalone:
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
            else:
                plt.show()
        return fig if standalone else ax

    def plot_all(self, save_path: str = None, figsize=(17, 7)):
        """T-Q diagram + Grand Composite Curve side by side."""
        if self._Q_hot is None:
            raise RuntimeError("Call run() before plot_all().")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        self._draw_tq_diagram(ax1)
        self._draw_gcc(ax2)
        fig.tight_layout(pad=2.5)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        return fig

    # ── Core algorithm ────────────────────────────────────────────────────────

    def _run_algorithm(self):
        """
        Graphical composite-curve method.

        Each composite curve is built independently from its own stream types,
        using only those streams' temperature breakpoints.  This ensures the
        hot curve starts at the actual minimum hot-stream temperature (not at
        a spurious low temperature driven by cold streams) and vice versa.

        The cold curve is then shifted horizontally until the minimum vertical
        gap between the two curves equals ΔT_min.  This shift is Q_C_min.
        """
        dT      = self.delta_T_min
        hot_s   = [s for s in self.streams if s.stream_type == 'hot']
        cold_s  = [s for s in self.streams if s.stream_type == 'cold']

        # 1. Independent composite curves (Q=0 at each curve's own T_min)
        Q_hot_raw,  T_hot_raw  = self._build_composite(hot_s)
        Q_cold_raw, T_cold_raw = self._build_composite(cold_s)

        # 2. Horizontal shift of cold curve: binary search so that
        #    min(T_hot(Q) − T_cold(Q)) = ΔT_min in the overlap zone
        Q_C_min = self._find_shift(Q_hot_raw, T_hot_raw,
                                   Q_cold_raw, T_cold_raw, dT)

        # 3. Apply shift
        Q_cold_shifted = Q_cold_raw + Q_C_min

        # 4. Energy balance
        Q_H_min    = max(0.0, Q_cold_shifted[-1] - Q_hot_raw[-1])
        Q_recovery = Q_hot_raw[-1] - Q_C_min

        # 5. Pinch point: Q where T_hot − T_cold is minimum in overlap zone
        T_ph, T_pc, Q_pinch = self._locate_pinch(
            Q_hot_raw, T_hot_raw, Q_cold_shifted, T_cold_raw
        )

        # 6. PTA for Grand Composite Curve
        self._run_pta(dT, Q_H_min)

        # Store composite curve data
        self._Q_hot  = Q_hot_raw
        self._T_hot  = T_hot_raw
        self._Q_cold = Q_cold_shifted
        self._T_cold = T_cold_raw

        T_ps = ((T_ph + T_pc) / 2.0) if (T_ph is not None and T_pc is not None) else None

        self.results.Q_H_min         = round(Q_H_min,            3)
        self.results.Q_C_min         = round(Q_C_min,            3)
        self.results.Q_recovery      = round(max(0.0, Q_recovery), 3)
        self.results.T_pinch_shifted = T_ps
        self.results.T_pinch_hot     = T_ph
        self.results.T_pinch_cold    = T_pc

    # ── Composite curve construction ──────────────────────────────────────────

    @staticmethod
    def _build_composite(streams: list) -> tuple[np.ndarray, np.ndarray]:
        """
        Build composite curve for a homogeneous list of streams (all hot or all cold)
        in real temperature space.

        Returns
        -------
        Q : np.ndarray  [kW]  – cumulative enthalpy, Q=0 at T_min, ascending
        T : np.ndarray  [°C]  – temperature, ascending
        """
        if not streams:
            return np.array([0.0]), np.array([0.0])

        # Temperature breakpoints from this stream type only
        temps = sorted({round(t, 9)
                        for s in streams
                        for t in (s.T_supply, s.T_target)})

        Q_list = [0.0]
        T_list = [temps[0]]
        q = 0.0

        for i in range(len(temps) - 1):
            T_lo, T_hi = temps[i], temps[i + 1]

            # Sum CP of streams that are active across the full interval [T_lo, T_hi]
            CP = sum(
                s.CP for s in streams
                if min(s.T_supply, s.T_target) <= T_lo + 1e-9
                and max(s.T_supply, s.T_target) >= T_hi - 1e-9
            )

            if CP > 1e-9:
                q += CP * (T_hi - T_lo)
                Q_list.append(q)
                T_list.append(T_hi)

        return np.array(Q_list), np.array(T_list)

    # ── Graphical shift (binary search) ──────────────────────────────────────

    @staticmethod
    def _find_shift(
        Q_hot: np.ndarray, T_hot: np.ndarray,
        Q_cold: np.ndarray, T_cold: np.ndarray,
        dT_min: float,
    ) -> float:
        """
        Find the horizontal shift s for the cold curve such that the minimum
        vertical distance between the two curves equals dT_min.

        The cold curve is placed at [s, s + Q_cold_total] on the Q-axis.
        We evaluate min(T_hot(Q) − T_cold(Q − s)) over the overlap zone and
        binary-search on s.

        Returns
        -------
        s : float  [kW]  – horizontal shift = Q_C_min
        """
        def min_gap(s: float) -> float:
            q_lo = float(s)
            q_hi = float(min(Q_hot[-1], s + Q_cold[-1]))
            if q_hi <= q_lo + 1e-9:
                return np.inf                           # no overlap
            g = np.linspace(q_lo, q_hi, 3000)
            t_h = np.interp(g, Q_hot, T_hot)
            t_c = np.interp(g - s, Q_cold, T_cold)     # cold raw at (Q − s)
            return float(np.min(t_h - t_c))

        # Threshold case: gap already satisfied at s = 0 (no shift needed)
        if min_gap(0.0) >= dT_min - 1e-6:
            return 0.0

        # Expand upper bound until min_gap exceeds dT_min
        s_hi = max(1.0, float(Q_hot[-1]) * 0.1)
        while min_gap(s_hi) < dT_min:
            s_hi *= 2.0
            if s_hi > 1e10:
                break   # pathological; return best estimate

        # Binary search (80 iterations → ~10-digit precision)
        s_lo = 0.0
        for _ in range(80):
            s_mid = (s_lo + s_hi) / 2.0
            if min_gap(s_mid) < dT_min:
                s_lo = s_mid
            else:
                s_hi = s_mid

        return (s_lo + s_hi) / 2.0

    # ── Pinch point localisation ──────────────────────────────────────────────

    @staticmethod
    def _locate_pinch(
        Q_hot: np.ndarray, T_hot: np.ndarray,
        Q_cold_shifted: np.ndarray, T_cold: np.ndarray,
    ) -> tuple:
        """
        Find pinch temperatures and Q-position as the point of minimum vertical
        separation between the two curves in the overlap zone.

        Returns (T_pinch_hot, T_pinch_cold, Q_pinch)  or  (None, None, None).
        """
        q_lo = float(Q_cold_shifted[0])   # cold curve left edge (= Q_C_min)
        q_hi = float(Q_hot[-1])           # hot curve right edge
        if q_hi <= q_lo + 1e-9:
            return None, None, None

        g    = np.linspace(q_lo, q_hi, 10_000)
        t_h  = np.interp(g, Q_hot, T_hot)
        t_c  = np.interp(g, Q_cold_shifted, T_cold)
        idx  = int(np.argmin(t_h - t_c))

        return float(t_h[idx]), float(t_c[idx]), float(g[idx])

    # ── PTA for Grand Composite Curve only ───────────────────────────────────

    def _run_pta(self, dT: float, Q_H_min: float):
        """
        Problem Table Algorithm, used only to build the Grand Composite Curve.
        Q_H_min is passed in from the graphical result so the GCC cascade is
        consistent with the composite-curve diagram.
        """
        # Shift all stream temperatures
        shifted = []
        for s in self.streams:
            if s.stream_type == 'hot':
                ss, st = s.T_supply - dT / 2, s.T_target - dT / 2
            else:
                ss, st = s.T_supply + dT / 2, s.T_target + dT / 2
            shifted.append({'stream': s, 'ss': ss, 'st': st})

        # Temperature intervals (descending shifted temps)
        temps_set = {round(sh['ss'], 9) for sh in shifted} | \
                    {round(sh['st'], 9) for sh in shifted}
        temperatures = sorted(temps_set, reverse=True)

        intervals = []
        for i in range(len(temperatures) - 1):
            t1, t2 = temperatures[i], temperatures[i + 1]
            delta_S  = t1 - t2
            delta_CP = 0.0

            for j, sh in enumerate(shifted):
                s = sh['stream']
                if s.stream_type == 'hot':
                    active = sh['ss'] >= t1 - 1e-9 and sh['st'] <= t2 + 1e-9
                    if active:
                        delta_CP += s.CP
                else:
                    active = sh['st'] >= t1 - 1e-9 and sh['ss'] <= t2 + 1e-9
                    if active:
                        delta_CP -= s.CP

            intervals.append({'t1': t1, 't2': t2,
                               'delta_H': delta_CP * delta_S})

        # Feasible cascade using the graphical Q_H_min
        exit_H = Q_H_min
        for iv in intervals:
            exit_H += iv['delta_H']
            iv['exit_H'] = exit_H

        self._temperatures_desc = temperatures
        self._intervals         = intervals

    # ── Grand Composite Curve ────────────────────────────────────────────────

    def _build_gcc(self):
        """
        GCC from the feasible cascade:
          Point 0 : H = Q_H_min  at  T* = T*_max  (top of diagram)
          Point i : H = exitH[i] at  T* = temperatures[i+1]
        Temperatures on the GCC y-axis are shifted (not real temperatures).
        """
        r = self.results
        H_gcc = [r.Q_H_min] + [iv['exit_H'] for iv in self._intervals]
        T_gcc = list(self._temperatures_desc)

        self._Q_gcc = np.array(H_gcc)
        self._T_gcc = np.array(T_gcc)

    # ── Plot: T-Q diagram ─────────────────────────────────────────────────────

    def _draw_tq_diagram(self, ax: plt.Axes):
        """
        T-Q diagram.  Both composite curves have a positive slope.

        Zone layout on Q-axis:
          [0,          Q_C_min]      Cooling Demand  (hot curve only)
          [Q_C_min,    Q_hot_total]  Heat Recovery   (overlap, shaded)
          [Q_hot_total, Q_cold_end]  Heating Demand  (cold curve only)

        The pinch point is where T_hot − T_cold is minimised (= ΔT_min).
        """
        dT     = self.delta_T_min
        Q_hot  = self._Q_hot
        T_hot  = self._T_hot
        Q_cold = self._Q_cold        # already shifted by Q_C_min
        T_cold = self._T_cold

        r         = self.results
        Q_H_min   = r.Q_H_min
        Q_C_min   = r.Q_C_min
        T_pinch_h = r.T_pinch_hot
        T_pinch_c = r.T_pinch_cold

        C_HOT   = '#9b1b30'
        C_COLD  = '#2c5f8a'
        C_SHADE = '#B0BEC5'
        C_GRID  = '#CFD8DC'

        ax.set_facecolor('#FAFBFC')

        x_ovl_start = Q_cold[0]    # = Q_C_min
        x_ovl_end   = Q_hot[-1]
        x_plot_end  = max(Q_hot[-1], Q_cold[-1])

        T_min_plot = min(T_hot[0],  T_cold[0])
        T_max_plot = max(T_hot[-1], T_cold[-1])
        T_span     = T_max_plot - T_min_plot

        # ── Heat Recovery shading ─────────────────────────────────────────────
        if x_ovl_end > x_ovl_start + 1e-6:
            q_grid  = np.linspace(x_ovl_start, x_ovl_end, 800)
            t_hot_i = np.interp(q_grid, Q_hot,  T_hot)
            t_cld_i = np.interp(q_grid, Q_cold, T_cold)
            ax.fill_between(q_grid, t_cld_i, t_hot_i,
                            color=C_SHADE, alpha=0.45, zorder=1)

        # ── Composite curves ──────────────────────────────────────────────────
        lw = 2.3
        ax.plot(Q_hot,  T_hot,  color=C_HOT,  linewidth=lw, zorder=3,
                solid_capstyle='round')
        ax.plot(Q_cold, T_cold, color=C_COLD, linewidth=lw, zorder=3,
                solid_capstyle='round')

        # Directional arrows at the ends where streams flow
        dq = max(x_plot_end * 0.025, 1.0)
        ax.annotate('', xy=(Q_hot[-1], T_hot[-1]),
                    xytext=(Q_hot[-1] - dq, T_hot[-1]),
                    arrowprops=dict(arrowstyle='->', color=C_HOT,
                                   lw=1.8, mutation_scale=14))
        ax.annotate('', xy=(Q_cold[0], T_cold[0]),
                    xytext=(Q_cold[0] + dq, T_cold[0]),
                    arrowprops=dict(arrowstyle='->', color=C_COLD,
                                   lw=1.8, mutation_scale=14))

        # ── Vertical zone dividers ────────────────────────────────────────────
        vkw = dict(color='#37474F', linewidth=1.1, linestyle='--',
                   zorder=2, alpha=0.85)
        if Q_C_min > 0.5:
            ax.axvline(x=Q_C_min, **vkw)
        ax.axvline(x=x_ovl_end, **vkw)

        # ── Zone labels ───────────────────────────────────────────────────────
        label_y  = (T_max_plot + T_span * 0.10) * 0.99
        arrow_kw = dict(arrowstyle='<->', color='#37474F', lw=1.2,
                        mutation_scale=10)
        txt_kw   = dict(ha='center', va='top', fontsize=9.5, color='#263238',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                  ec='none', alpha=0.7))

        if Q_C_min > 1.0:
            ax.annotate('', xy=(Q_C_min, label_y),
                        xytext=(0, label_y), arrowprops=arrow_kw)
            ax.text(Q_C_min / 2, label_y, 'Cooling\nDemand', **txt_kw)

        if x_ovl_end > x_ovl_start + 1.0:
            ax.annotate('', xy=(x_ovl_end, label_y),
                        xytext=(x_ovl_start if Q_C_min > 0.5 else 0, label_y),
                        arrowprops=arrow_kw)
            ax.text((x_ovl_start + x_ovl_end) / 2, label_y,
                    'Heat Recovery', **txt_kw)

        if Q_cold[-1] > x_ovl_end + 0.5:
            ax.annotate('', xy=(Q_cold[-1], label_y),
                        xytext=(x_ovl_end, label_y), arrowprops=arrow_kw)
            ax.text((x_ovl_end + Q_cold[-1]) / 2, label_y,
                    'Heating\nDemand', **txt_kw)

        # ── Pinch Point ───────────────────────────────────────────────────────
        if T_pinch_h is not None and T_pinch_c is not None:
            ax.axhline(T_pinch_h, color=C_HOT,  ls=':', lw=1.3,
                       alpha=0.7, zorder=2)
            ax.axhline(T_pinch_c, color=C_COLD, ls=':', lw=1.3,
                       alpha=0.7, zorder=2)

            # Pinch Q on the hot curve (T_hot is ascending → direct interp)
            Q_pinch = float(np.interp(T_pinch_h, T_hot, Q_hot))
            ax.plot(Q_pinch, T_pinch_h, 'o', color=C_HOT,
                    markersize=7, zorder=5,
                    markeredgecolor='white', markeredgewidth=1.5)

            ax.annotate(
                'Pinch Point',
                xy=(Q_pinch, T_pinch_h),
                xytext=(Q_pinch + x_plot_end * 0.07,
                        T_pinch_h + T_span * 0.08),
                fontsize=9, color='#263238',
                arrowprops=dict(arrowstyle='->', color='#263238', lw=1.2,
                               connectionstyle='arc3,rad=-0.2'),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#90A4AE',
                          alpha=0.92, lw=0.8),
            )

        # ── Heat-transfer arrows in overlap zone ──────────────────────────────
        if x_ovl_end > x_ovl_start + 1.0:
            n_arr = min(5, max(2, int((x_ovl_end - x_ovl_start) / 80)))
            q_arr = np.linspace(x_ovl_start + (x_ovl_end - x_ovl_start) * 0.15,
                                x_ovl_end   - (x_ovl_end - x_ovl_start) * 0.15,
                                n_arr)
            for qa in q_arr:
                t_h = float(np.interp(qa, Q_hot,  T_hot))
                t_c = float(np.interp(qa, Q_cold, T_cold))
                if t_h > t_c + dT * 0.8:
                    ax.annotate('', xy=(qa, t_c + (t_h - t_c) * 0.15),
                                xytext=(qa, t_h - (t_h - t_c) * 0.15),
                                arrowprops=dict(arrowstyle='->', color='#546E7A',
                                               lw=0.9, ls='dotted',
                                               mutation_scale=9))

        # ── Legend ────────────────────────────────────────────────────────────
        ax.legend(handles=[
            plt.Line2D([0], [0], color=C_HOT,  lw=2.2,
                       label='Hot Composite Curve'),
            plt.Line2D([0], [0], color=C_COLD, lw=2.2,
                       label='Cold Composite Curve'),
            mpatches.Patch(facecolor=C_SHADE, alpha=0.55,
                           label='Heat Recovery'),
        ], loc='lower right', fontsize=9, framealpha=0.92, edgecolor='#B0BEC5')

        # ── Info box ──────────────────────────────────────────────────────────
        info = [f"ΔT_min = {dT:.0f} K",
                f"Q_H,min = {r.Q_H_min:.1f} kW",
                f"Q_C,min = {r.Q_C_min:.1f} kW",
                f"Q_rec = {r.Q_recovery:.1f} kW"]
        if T_pinch_h is not None:
            info += [f"T_Pinch,hot = {T_pinch_h:.1f} °C",
                     f"T_Pinch,cold = {T_pinch_c:.1f} °C"]
        ax.text(0.015, 0.98, '\n'.join(info),
                transform=ax.transAxes, fontsize=8.5, va='top', color='#263238',
                bbox=dict(boxstyle='round,pad=0.4', fc='white',
                          ec='#B0BEC5', alpha=0.92, lw=0.8))

        # ── Axes ──────────────────────────────────────────────────────────────
        ax.set_xlabel('Heat Flow  Q [kW]', fontsize=11, labelpad=6)
        ax.set_ylabel('Temperature  T [°C]', fontsize=11, labelpad=6)
        ax.set_xlim(left=-x_plot_end * 0.02,
                    right=x_plot_end * 1.02)
        ax.set_ylim(T_min_plot - T_span * 0.05,
                    T_max_plot + T_span * 0.18)
        ax.grid(True, color=C_GRID, linewidth=0.7, alpha=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ── Plot: Grand Composite Curve ───────────────────────────────────────────

    def _draw_gcc(self, ax: plt.Axes):
        """
        GCC with shifted temperatures on Y-axis and net heat flow on X-axis.
        The curve touches H = 0 at the pinch temperature.
        """
        Q = self._Q_gcc
        T = self._T_gcc    # shifted, descending
        r = self.results

        C_GCC  = '#1565C0'
        C_GRID = '#CFD8DC'

        ax.set_facecolor('#FAFBFC')

        ax.plot(Q, T, color=C_GCC, linewidth=2.4, zorder=3)
        ax.fill_betweenx(T, 0, Q, color=C_GCC, alpha=0.12, zorder=1)
        ax.plot(Q, T, 'o', color=C_GCC, markersize=4.5, zorder=4,
                markeredgecolor='white', markeredgewidth=0.8)

        T_pinch_s = r.T_pinch_shifted
        if T_pinch_s is not None:
            ax.axhline(T_pinch_s, color='#C0392B', ls='--', lw=1.5, alpha=0.8,
                       label=f'Pinch  T* = {T_pinch_s:.1f} °C')

            Q_max   = float(np.max(Q)) if len(Q) > 0 else 1.0
            T_range = float(T[0] - T[-1]) if len(T) > 1 else 1.0

            if r.Q_H_min > 0.5:
                ax.annotate(f"Q_H,min\n{r.Q_H_min:.1f} kW",
                            xy=(float(Q[0]), float(T[0])),
                            xytext=(float(Q[0]) + Q_max * 0.08,
                                    float(T[0]) - T_range * 0.05),
                            fontsize=8.5, color='#C0392B',
                            arrowprops=dict(arrowstyle='->', color='#C0392B',
                                           lw=1.1),
                            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                      ec='#EF9A9A', alpha=0.9))

            if r.Q_C_min > 0.5:
                ax.annotate(f"Q_C,min\n{r.Q_C_min:.1f} kW",
                            xy=(float(Q[-1]), float(T[-1])),
                            xytext=(float(Q[-1]) + Q_max * 0.08,
                                    float(T[-1]) + T_range * 0.05),
                            fontsize=8.5, color='#1565C0',
                            arrowprops=dict(arrowstyle='->', color='#1565C0',
                                           lw=1.1),
                            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                      ec='#90CAF9', alpha=0.9))

        ax.set_xlabel('Net Heat Flow  H [kW]', fontsize=11, labelpad=6)
        ax.set_ylabel('Shifted Temperature  T* [°C]', fontsize=11, labelpad=6)
        ax.set_xlim(left=0)
        if T_pinch_s is not None:
            ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, color=C_GRID, linewidth=0.7, alpha=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


# ══════════════════════════════════════════════════════════════════════════════
# Standalone test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    path = sys.argv[1] if len(sys.argv) > 1 else 'HTHP_questionnaire.xlsx'

    try:
        from questionnaire_reader import QuestionnaireReader
        from pinch_params import PinchParamBuilder
        reader  = QuestionnaireReader(path)
        params  = reader.get_params()
        builder = PinchParamBuilder(params, delta_T_min=10.0)
        pinch_input = builder.get_pinch_params()
        print(f"Questionnaire: {path}")
    except Exception as e:
        print(f"Fallback demo parameters ({e})")
        from pinch_params import PinchStream
        pinch_input = {
            'delta_T_min': 20.0,
            'warnings': [],
            'streams': [
                PinchStream('Hot Process A',  'hot',  110,  40, 3.0, 330.0),
                PinchStream('Hot Process B',  'hot',  90,  40, 1.5, 180.0),
                PinchStream('Hot Process C',  'hot',  80,  50, 2.2, 176.0),
                PinchStream('Hot Process D',  'hot',   95,  60, 1.8, 108.0),
                PinchStream('Hot Process E',  'hot',  120,  20, 1.2, 132.0),
                PinchStream('Cold Process F', 'cold',  80, 180, 2.0, 230.0),
                PinchStream('Cold Process G', 'cold',  50,  90, 4.0, 240.0),
                PinchStream('Cold Process H', 'cold',  60, 100, 1.5, 112.5),
                PinchStream('Cold Process I', 'cold',  90,  200, 2.8, 252.0),
                PinchStream('Cold Process J', 'cold',  40,  90, 1.0,  60.0),
            ],
        }

    import matplotlib
    matplotlib.use('Agg')

    analyzer = PinchAnalyzer(**pinch_input)
    analyzer.run()
    analyzer.print_results()

    SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
    OUTPUT_DIR = SCRIPT_DIR / 'final_graphics' / 'pinch_analysis'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save individual plots as PDF
    save_path_tq = OUTPUT_DIR / 'pinch_tq_diagram.pdf'
    analyzer.plot(save_path=str(save_path_tq))
    print(f"Plot saved: {save_path_tq}")

    save_path_gcc = OUTPUT_DIR / 'pinch_gcc.pdf'
    analyzer.plot_gcc(save_path=str(save_path_gcc))
    print(f"Plot saved: {save_path_gcc}")

    # Save both plots side-by-side as combined PDF
    save_path_all = OUTPUT_DIR / 'pinch_diagram_combined.pdf'
    analyzer.plot_all(save_path=str(save_path_all))
    print(f"Plot saved: {save_path_all}")