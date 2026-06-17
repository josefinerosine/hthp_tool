"""
mvr_stage_selection.py

Bestimmung der Stufenanzahl für mehrstufige MVR-Systeme.

Standardmethode: Sättigungstemperatur-Differenz
    n = ceil( (T_sat(p_out) - T_sat(p_in)) / dT_per_stage )
    Standardwert: dT_per_stage = 10 K

Verwendung:
    from mvr_stage_selection import determine_n_stages

    n_stages, info = determine_n_stages(p_in=1.5, p_out=6.0)
    n_stages, info = determine_n_stages(p_in=1.5, p_out=6.0, dT_per_stage=12)
"""

import math
from CoolProp.CoolProp import PropsSI as PSI


# ── Auslegungsrichtwerte ─────────────────────────────────────────────────────

MVR_DESIGN_GUIDELINES = {
    # Sättigungstemperaturhub pro Stufe [K]  ← Standardmethode
    'dT_sat_per_stage': {
        'conservative': 8,
        'standard':    10,   # EMPFOHLEN
        'aggressive':  13,
    },
    # Max. Druckverhältnis pro Stufe [-]  ← Validierung / Fallback
    'PR_per_stage': {
        'conservative': 1.4,
        'standard':     1.6,
        'aggressive':   1.8,
    },
    'n_stages_max': 6,
    'n_stages_min': 1,
}


# ── Hilfsfunktion: Sättigungstemperatur ─────────────────────────────────────

def _T_sat(p_bar: float, wf: str = 'Water') -> float:
    """Sättigungstemperatur in °C bei gegebenem Druck in bar."""
    return PSI('T', 'P', p_bar * 1e5, 'Q', 1, wf) - 273.15


# ── Kernmethode: Sättigungstemperatur-Delta ──────────────────────────────────

def determine_n_stages_from_sat_temperature(
    p_in:  float,
    p_out: float,
    dT_per_stage: float | None = None,
    design_philosophy: str = 'standard',
    wf: str = 'Water',
) -> tuple[int, dict]:
    """
    Stufenanzahl aus Sättigungstemperaturdifferenz.

    n = ceil( (T_sat(p_out) - T_sat(p_in)) / dT_per_stage )

    Parameters
    ----------
    p_in, p_out      : Drücke [bar]
    dT_per_stage     : max. Sättigungs-ΔT je Stufe [K]; None → aus Philosophie
    design_philosophy: 'conservative' | 'standard' | 'aggressive'
    wf               : Arbeitsmittel (Default: 'Water')

    Returns
    -------
    n_stages : int
    info     : dict mit Zwischenwerten
    """
    if dT_per_stage is None:
        dT_per_stage = MVR_DESIGN_GUIDELINES['dT_sat_per_stage'][design_philosophy]

    T_sat_in  = _T_sat(p_in,  wf)
    T_sat_out = _T_sat(p_out, wf)
    dT_total  = T_sat_out - T_sat_in

    n = max(
        MVR_DESIGN_GUIDELINES['n_stages_min'],
        min(
            MVR_DESIGN_GUIDELINES['n_stages_max'],
            math.ceil(dT_total / dT_per_stage),
        ),
    )

    info = {
        'method':       'saturation_temperature',
        'T_sat_in':      T_sat_in,
        'T_sat_out':     T_sat_out,
        'dT_total':      dT_total,
        'dT_per_stage':  dT_per_stage,
        'PR_total':      p_out / p_in,
        'PR_per_stage':  (p_out / p_in) ** (1 / n),
    }
    return n, info


# ── Druckbasierte Methode (Validierung / Fallback) ───────────────────────────

def determine_n_stages_from_pressure(
    p_in:  float,
    p_out: float,
    PR_max_per_stage: float | None = None,
    design_philosophy: str = 'standard',
) -> tuple[int, dict]:
    """
    Stufenanzahl aus maximalem Druckverhältnis pro Stufe.

    n = ceil( ln(PR_total) / ln(PR_max) )
    """
    if PR_max_per_stage is None:
        PR_max_per_stage = MVR_DESIGN_GUIDELINES['PR_per_stage'][design_philosophy]

    PR_total = p_out / p_in
    n = max(
        MVR_DESIGN_GUIDELINES['n_stages_min'],
        min(
            MVR_DESIGN_GUIDELINES['n_stages_max'],
            math.ceil(math.log(PR_total) / math.log(PR_max_per_stage)),
        ),
    )

    info = {
        'method':        'pressure',
        'PR_total':       PR_total,
        'PR_max_per_stage': PR_max_per_stage,
        'PR_per_stage':  PR_total ** (1 / n),
    }
    return n, info


# ── Hauptfunktion ────────────────────────────────────────────────────────────

def determine_n_stages(
    p_in:  float,
    p_out: float,
    dT_per_stage:      float | None = None,
    PR_max_per_stage:  float | None = None,
    design_philosophy: str  = 'standard',
    wf:                str  = 'Water',
    verbose:           bool = True,
) -> tuple[int, dict]:
    """
    Empfohlene MVR-Stufenanzahl (Standardmethode: Sättigungstemperatur-Delta).

    Primär: Sättigungstemperaturhub-Methode  (dT_per_stage, Default 10 K)
    Sekundär: Druckverhältnis-Check als Validierung

    Parameters
    ----------
    p_in, p_out        : Drücke [bar]
    dT_per_stage       : max. ΔT_sat je Stufe [K]  (None → design_philosophy)
    PR_max_per_stage   : max. PR je Stufe [-]       (None → design_philosophy)
    design_philosophy  : 'conservative' | 'standard' | 'aggressive'
    wf                 : Arbeitsmittel
    verbose            : Konsolenausgabe

    Returns
    -------
    n_stages : int
    info     : dict  (Zwischenwerte und Begründung)
    """
    PR_total = p_out / p_in

    # ── Primär: Sättigungstemperatur-Methode ──────────────────────────────
    n_sat, info_sat = determine_n_stages_from_sat_temperature(
        p_in, p_out,
        dT_per_stage=dT_per_stage,
        design_philosophy=design_philosophy,
        wf=wf,
    )

    # ── Sekundär: Druckverhältnis-Validierung ─────────────────────────────
    n_pr, info_pr = determine_n_stages_from_pressure(
        p_in, p_out,
        PR_max_per_stage=PR_max_per_stage,
        design_philosophy=design_philosophy,
    )

    # Stufenanzahl ausschließlich über ΔT_sat-Methode bestimmen.
    # Die PR-Methode wird nur noch als Referenzwert mitgeführt, nicht verwendet.
    n_recommended = n_sat

    reason = (
        f"ΔT_sat-Methode: {info_sat['dT_total']:.1f} K / "
        f"{info_sat['dT_per_stage']} K·Stufe⁻¹ → {n_sat} Stufen"
        f"  (PR-Referenz: {n_pr} Stufen bei PR_max={info_pr['PR_max_per_stage']:.2f}, nicht verwendet)"
    )

    info = {
        'n_stages':           n_recommended,
        'n_sat_temperature':  n_sat,
        'n_pressure':         n_pr,          # Referenzwert, nicht aktiv
        'reason':             reason,
        'design_philosophy':  design_philosophy,
        'T_sat_in':           info_sat['T_sat_in'],
        'T_sat_out':          info_sat['T_sat_out'],
        'dT_total':           info_sat['dT_total'],
        'dT_per_stage':       info_sat['dT_per_stage'],
        'PR_total':           PR_total,
        'PR_per_stage':       PR_total ** (1 / n_recommended),
    }

    if verbose:
        print(f"  MVR Stufenbestimmung: p_in={p_in} bar → p_out={p_out} bar")
        print(f"    T_sat: {info['T_sat_in']:.1f}°C → {info['T_sat_out']:.1f}°C  "
              f"(ΔT={info['dT_total']:.1f} K)")
        print(f"    ΔT/Stufe={info['dT_per_stage']} K  → n_sat={n_sat} Stufen  "
              f"(PR-Referenz: n_pr={n_pr}, nicht aktiv)")
        print(f"    → {n_recommended} Stufen  (ausschließlich ΔT_sat-Methode)")

    return n_recommended, info


# ── Beispiele ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cases = [
        (1.5, 6.0,  "Questionnaire-Beispiel"),
        (0.7, 3.1,  "Low-Pressure MVR"),
        (1.0, 10.0, "Hohes Druckverhältnis"),
        (2.0, 4.0,  "Kleines Druckverhältnis"),
    ]
    for p_in, p_out, label in cases:
        print(f"\n{'─'*60}")
        print(f"  {label}  ({p_in} → {p_out} bar)")
        n, info = determine_n_stages(p_in, p_out)
        print(f"  Ergebnis: {n} Stufen  |  PR/Stufe={info['PR_per_stage']:.3f}")