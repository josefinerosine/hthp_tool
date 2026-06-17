"""
Case Calculator
===============
Zentrale Berechnungseinheit für alle Technologie-Cases.

Einstiegspunkt:
    results = calculate_cases(cases, params)

Einheitliches Case-Format  (Liste von Dicts):
──────────────────────────────────────────────────────────────────────────────
  Pflichtfelder (alle Cases):
    'id'    : str        eindeutiger Bezeichner, z.B. 'hthp_simple_R600a'
    'type'  : CaseType   CaseType.HTHP | CaseType.MVR | CaseType.HTHP_MVR

  Für CaseType.HTHP:
    'model'       : str   Klassenname aus heatpumps.models, z.B. 'HeatPumpSimple'
    'refrigerant' : str   Kältemittel  (single-stage)
    'refrigerant1': str   Kältemittel Niedertemperaturkreis  (Cascade)
    'refrigerant2': str   Kältemittel Hochtemperaturkreis    (Cascade)
    'econ_type'   : str   'open' | 'closed'  (nur für Econ-Modelle)
    'eta_s'       : float isentroper Wirkungsgrad Verdichter [-]

  Für CaseType.MVR:
    'n_stages'  : int | None   Stufenzahl (None = automatisch)
    'eta_s'     : float        isentroper Wirkungsgrad Verdichter [-]

  Für CaseType.HTHP_MVR:
    'model'           : str    HTHP-Modell für die Vorwärmstufe
    'refrigerant'     : str    Kältemittel der HTHP-Stufe
    'eta_s_hthp'      : float  η_s HTHP-Verdichter
    'eta_s_mvr'       : float  η_s MVR-Verdichter
    'n_stages'        : int | None
    'econ_type'       : str    (falls HTHP-Modell Econ hat)

Rückgabe:
    dict[case_id → CaseResult]

Modellquellen:
    HTHP      → heatpumps.models  (jfreissmann/heatpumps)
    MVR       → myModels.mvrMultiStage.MVRMultiStage
    HTHP+MVR  → myModels.hthp_mvr.HTHPMVRHybrid
                 (nutzt intern ebenfalls heatpumps für die HTHP-Stufe)
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class CaseType(Enum):
    """Typ des zu berechnenden Cases."""
    HTHP     = 'hthp'      # Wärmepumpe via heatpumps-Bibliothek
    MVR      = 'mvr'       # Mechanische Brüdenverdichtung
    HTHP_MVR = 'hthp_mvr'  # Hybrid: HTHP-Vorwärmung + MVR-Endstufe


class CaseStatus(Enum):
    """Berechnungsstatus."""
    PENDING = 'pending'
    SUCCESS = 'success'
    FAILED  = 'failed'
    SKIPPED = 'skipped'


# ── Modell-Taxonomie ──────────────────────────────────────────────────────────
# Cascade-Modelle: zwei Kältemittelkreise (wf1 / wf2, refrig1 / refrig2)
_CASCADE_MODELS: frozenset[str] = frozenset({
    'HeatPumpCascade',
    'HeatPumpCascadeTrans',
    'HeatPumpCascade2IHX',
    'HeatPumpCascade2IHXTrans',
    'HeatPumpCascadeIC',
    'HeatPumpCascadeICTrans',
    'HeatPumpCascadeFlash',
    'HeatPumpCascadeFlashTrans',
    'HeatPumpCascadeEcon',
    'HeatPumpCascadeEconTrans',
    'HeatPumpCascadeEconIHX',
    'HeatPumpCascadeEconIHXTrans',
    'HeatPumpCascadeIHXEcon',
    'HeatPumpCascadeIHXEconTrans',
    'HeatPumpCascadePC',
    'HeatPumpCascadePCTrans',
    'HeatPumpCascadePCIHX',
    'HeatPumpCascadePCIHXTrans',
    'HeatPumpCascadeIHXPC',
    'HeatPumpCascadeIHXPCTrans',
    'HeatPumpCascadeIHXPCIHX',
    'HeatPumpCascadeIHXPCIHXTrans',
})

# Econ-Modelle: benötigen econ_type='open'|'closed' für get_params()
_ECON_MODELS: frozenset[str] = frozenset({
    m for m in (
        'HeatPumpEcon',         'HeatPumpEconTrans',
        'HeatPumpEconIHX',      'HeatPumpEconIHXTrans',
        'HeatPumpIHXEcon',      'HeatPumpIHXEconTrans',
        'HeatPumpPC',           'HeatPumpPCTrans',
        'HeatPumpPCIHX',        'HeatPumpPCIHXTrans',
        'HeatPumpIHXPC',        'HeatPumpIHXPCTrans',
        'HeatPumpIHXPCIHX',     'HeatPumpIHXPCIHXTrans',
        'HeatPumpCascadeEcon',  'HeatPumpCascadeEconTrans',
        'HeatPumpCascadeEconIHX','HeatPumpCascadeEconIHXTrans',
        'HeatPumpCascadeIHXEcon','HeatPumpCascadeIHXEconTrans',
        'HeatPumpCascadePC',    'HeatPumpCascadePCTrans',
        'HeatPumpCascadePCIHX', 'HeatPumpCascadePCIHXTrans',
        'HeatPumpCascadeIHXPC', 'HeatPumpCascadeIHXPCTrans',
        'HeatPumpCascadeIHXPCIHX','HeatPumpCascadeIHXPCIHXTrans',
    )
})

# Modelle mit internem Wärmeübertrager (IHX) – unterstützen Überhitzung via dT_sh
# Alle anderen Modelle erzwingen x=1 (Sattdampf) am Verdichtereintritt.
# Namen direkt aus heatpumps.parameters (key-Format: ModelName_econ_type).
_IHX_MODELS: frozenset[str] = frozenset({
    # Simple / single-stage
    'HeatPumpIHX',              'HeatPumpIHXTrans',
    # Series Econ + IHX
    'HeatPumpIHXEcon',          'HeatPumpIHXEconTrans',
    'HeatPumpEconIHX',          'HeatPumpEconIHXTrans',
    # Parallel (PC) + IHX
    'HeatPumpIHXPC',            'HeatPumpIHXPCTrans',
    'HeatPumpPCIHX',            'HeatPumpPCIHXTrans',
    'HeatPumpIHXPCIHX',         'HeatPumpIHXPCIHXTrans',
    # Cascade base + 2× IHX
    'HeatPumpCascade2IHX',      'HeatPumpCascade2IHXTrans',
    # Cascade + Series Econ + IHX
    'HeatPumpCascadeIHXEcon',   'HeatPumpCascadeIHXEconTrans',
    'HeatPumpCascadeEconIHX',   'HeatPumpCascadeEconIHXTrans',
    # Cascade + Parallel (PC) + IHX
    'HeatPumpCascadeIHXPC',     'HeatPumpCascadeIHXPCTrans',
    'HeatPumpCascadePCIHX',     'HeatPumpCascadePCIHXTrans',
    'HeatPumpCascadeIHXPCIHX',  'HeatPumpCascadeIHXPCIHXTrans',
})



# ══════════════════════════════════════════════════════════════════════════════
# Ergebnis-Datenklasse
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CaseResult:
    """
    Einheitliches Ergebnisobjekt für jeden berechneten Case.

    Attribute
    ---------
    case_id        : Bezeichner (identisch mit cases[i]['id'])
    case_type      : CaseType
    status         : CaseStatus
    model_instance : Referenz auf das laufende Modellobjekt
                     (HeatPumpXxx / MVRMultiStage / HTHPMVRHybrid)
                     Kann für Weiterauswertung genutzt werden:
                       hp.generate_state_diagram(...)
                       hp.perform_exergy_analysis(...)
    params_used    : Tatsächlich an das Modell übergebene Parameter
    COP            : Coefficient of Performance  [-]
    Q_con          : Nutzwärme / Kondensationsleistung  [kW]
    W_comp         : Elektrische Verdichterleistung gesamt  [kW]
    epsilon        : Exergetischer Wirkungsgrad  [-]
    T_hot_out      : Austrittstemperatur Senke  [°C]   (= C3)
    T_cold_in      : Eintrittstemperatur Quelle  [°C]  (= B1)
    extra          : Erweiterungsdict für modellspezifische Größen
    error          : Fehlermeldung (nur bei status == FAILED)
    """
    case_id:        str
    case_type:      CaseType
    status:         CaseStatus        = CaseStatus.PENDING
    model_instance: Any               = field(default=None, repr=False)
    params_used:    dict              = field(default_factory=dict, repr=False)

    # Thermodynamische Kerngrößen
    COP:     float | None = None
    Q_con:   float | None = None   # kW
    W_comp:  float | None = None   # kW
    SEI:     float | None = None   # kWh/kg  (MVR / Hybrid)
    epsilon: float | None = None   # exergetischer Wirkungsgrad

    # Betriebspunkttemperaturen
    T_hot_out: float | None = None   # °C  (Senken-Vorlauf)
    T_cold_in: float | None = None   # °C  (Quellen-Vorlauf)

    # Erweiterbar
    extra: dict      = field(default_factory=dict)
    error: str | None = None

    def summary_line(self) -> str:
        cop_s = f"{self.COP:.3f}" if self.COP is not None else '—'
        q_s   = f"{self.Q_con:.1f} kW" if self.Q_con is not None else '—'
        w_s   = f"{self.W_comp:.1f} kW" if self.W_comp is not None else '—'
        e_s   = f"{self.epsilon:.3f}" if self.epsilon is not None else '—'
        base  = (f"[{self.status.value.upper():<7}]  "
                 f"{self.case_id:<40}  "
                 f"COP={cop_s}  Q={q_s}  W={w_s}  ε={e_s}")
        if self.error:
            base += f"\n           ERROR: {self.error}"
        return base

    def __repr__(self):
        return self.summary_line()


# ══════════════════════════════════════════════════════════════════════════════
# Haupt-Funktion
# ══════════════════════════════════════════════════════════════════════════════

def calculate_cases(
    cases:             list[dict],
    params:            dict,
    verbose:           bool = True,
    progress_callback: "callable | None" = None,
    on_progress:       "callable | None" = None,
) -> dict[str, CaseResult]:
    """
    Berechnet alle uebergebenen Cases sequenziell.

    Parameters
    ----------
    cases             : Liste von Case-Konfigurationen (siehe Modul-Docstring)
    params            : Questionnaire-Parameter aus QuestionnaireReader.get_params()
    verbose           : Fortschrittsausgabe auf Konsole
    progress_callback : Optional callable f(current, total, case_id, status_str)
                        current ist 1-indiziert; status_str in {'success','failed','skipped'}.
    on_progress       : Alias fuer progress_callback (rueckwaertskompatibel).

    Returns
    -------
    dict { case_id -> CaseResult }
    """
    if progress_callback is None and on_progress is not None:
        progress_callback = on_progress
    results: dict[str, CaseResult] = {}

    if verbose:
        print(f"\n{'═'*70}")
        print(f"  CASE CALCULATOR  –  {len(cases)} Case(s)")
        print(f"{'═'*70}")

    params = _normalize_params(params)  # Excel-Label → kanonische Keys
    for idx, case_cfg in enumerate(cases):
        case_id   = case_cfg.get('id', f'case_{idx}')
        case_type = case_cfg.get('type')

        if verbose:
            print(f"\n  [{idx+1}/{len(cases)}]  {case_id}  ({case_type})")
            print(f"  {'─'*67}")

        # ── Typ-Validierung ───────────────────────────────────────────────
        if case_type is None:
            results[case_id] = CaseResult(
                case_id=case_id,
                case_type=CaseType.HTHP,
                status=CaseStatus.SKIPPED,
                error="Kein 'type' angegeben.",
            )
            if verbose:
                print(f"  SKIPPED – kein Typ angegeben.")
            continue

        # ── Dispatch ──────────────────────────────────────────────────────
        try:
            if case_type == CaseType.HTHP:
                result = _run_hthp(case_cfg, params, verbose)

            elif case_type == CaseType.MVR:
                result = _run_mvr(case_cfg, params, verbose)

            elif case_type == CaseType.HTHP_MVR:
                result = _run_hthp_mvr(case_cfg, params, verbose)

            else:
                result = CaseResult(
                    case_id=case_id,
                    case_type=case_type,
                    status=CaseStatus.SKIPPED,
                    error=f"Unbekannter CaseType: {case_type}",
                )

        except Exception as exc:
            result = CaseResult(
                case_id=case_id,
                case_type=case_type,
                status=CaseStatus.FAILED,
                error=str(exc),
            )
            if verbose:
                traceback.print_exc()

        results[case_id] = result

        if progress_callback is not None:
            try:
                progress_callback(idx + 1, len(cases), case_id, result.status.value)
            except Exception:
                pass

        if verbose:
            print(f"  → {result.summary_line()}")

    if verbose:
        _print_summary(results)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# HTHP  –  heatpumps-Bibliothek (jfreissmann)
# ══════════════════════════════════════════════════════════════════════════════

def _run_hthp(case_cfg: dict, params: dict, verbose: bool) -> CaseResult:
    """
    Startet ein Wärmepumpen-Modell aus der heatpumps-Bibliothek.

    Unterstützte Modellklassen (case_cfg['model']):
    ┌────────────────────────────────┬──────────────────────────────────────┐
    │ Single-stage                   │ Cascade                              │
    ├────────────────────────────────┼──────────────────────────────────────┤
    │ HeatPumpSimple / SimpleTrans   │ HeatPumpCascade / CascadeTrans       │
    │ HeatPumpIHX / IHXTrans         │ HeatPumpCascade2IHX                  │
    │ HeatPumpIC / ICTrans           │ HeatPumpCascadeIC / ICTrans          │
    │ HeatPumpFlash / FlashTrans     │ HeatPumpCascadeFlash / FlashTrans    │
    │ HeatPumpEcon (+ econ_type)     │ HeatPumpCascadeEcon (+ econ_type)    │
    │ HeatPumpEconIHX (+ econ_type)  │ HeatPumpCascadeEconIHX (+ econ_type) │
    │ HeatPumpPC / PCIHX (+ econ)    │ HeatPumpCascadePC / PCIHX (+ econ)   │
    └────────────────────────────────┴──────────────────────────────────────┘

    Ablauf
    ------
    1.  Modell-Klasse & get_params() laden
    2.  Basis-Parameterset aus JSON holen
    3.  Questionnaire-Werte eintragen
    4.  Modell instanziieren → run_model()
    5.  Ergebnisse extrahieren
    """
    case_id    = case_cfg['id']
    model_name = case_cfg.get('model', 'HeatPumpSimple')
    is_cascade = model_name in _CASCADE_MODELS
    needs_econ = model_name in _ECON_MODELS

    if verbose:
        tag = 'CASCADE' if is_cascade else 'SINGLE'
        print(f"  Modell: {model_name}  [{tag}]")

    # ── 1. Imports ────────────────────────────────────────────────────────
    try:
        import heatpumps.src.heatpumps.models as _hp_models
        from heatpumps.src.heatpumps.parameters import get_params
    except ImportError as e:
        raise ImportError(
            f"heatpumps-Bibliothek nicht gefunden. "
            f"Bitte in hthp_project_final/heatpumps installieren.\n{e}"
        )

    ModelClass = getattr(_hp_models, model_name, None)
    if ModelClass is None:
        raise ValueError(
            f"Modell '{model_name}' nicht in heatpumps.models gefunden."
        )

    # ── 2. Basis-Parameterset ─────────────────────────────────────────────
    if needs_econ:
        econ_type = case_cfg.get('econ_type', 'closed')
        hp_params = get_params(model_name, econ_type=econ_type)
        if verbose:
            print(f"  econ_type: {econ_type}")
    else:
        hp_params = get_params(model_name)

    # ── 3. Parameter aus Questionnaire setzen ─────────────────────────────
    _parametrize_hthp(hp_params, case_cfg, params, is_cascade)

    # ── 4. Modell ausführen ───────────────────────────────────────────────
    _model_kwargs = {'econ_type': econ_type} if needs_econ else {}
    hp = ModelClass(params=hp_params, **_model_kwargs)
    # Kaskadentemperatur: T_mid nach init_simulation() auf t_cascade_hx setzen,
    # damit design_simulation() → A4.T = T_mid − ttd_u/2 den richtigen Wert nutzt.
    if is_cascade and case_cfg.get('t_cascade_hx') is not None:
        _patch_cascade_T_mid(hp, case_cfg['t_cascade_hx'])
    hp.run_model()

    # ── 5. Ergebnisse extrahieren ─────────────────────────────────────────
    return _extract_hthp_results(hp, case_id, case_cfg, hp_params, is_cascade)



def _apply_hthp_superheat(hp_params: dict, model_name: str,
                           sh_lp: float = 0.0, sh_hp: float = 0.0):
    """Setzt die IHX-Überhitzung am Verdichtereintritt.

    sh_lp : Überhitzung LP-Verdichter [K]  (IHX Variante A / einfacher IHX)
    sh_hp : Überhitzung HP-Verdichter [K]  (IHX Variante B / zweiter IHX)

    Mapping je Modell (ermittelt aus tatsächlichen ihx-Schlüsseln im Parameterset):
      'ihx'        → einzelner IHX → sh_lp anwenden
      'ihx1','ihx2'→ zwei IHX (Cascade / IHXPCIHXTrans) → ihx1=sh_lp, ihx2=sh_hp
      'ihx1'..'ihx4'→ vier IHX (CascadeIHXPCIHX) → verteilt auf LP/HP je Kreis
    Modelle ohne IHX-Schlüssel werden unverändert gelassen.
    """
    if model_name not in _IHX_MODELS:
        return

    ihx_keys = sorted(
        [k for k in hp_params if k == 'ihx' or (k.startswith('ihx') and k[3:].isdigit())]
    )
    n = len(ihx_keys)
    if n == 0:
        return

    if n == 1:
        # Single IHX (simple, Variant A, or Variant B) – use sh_lp if set, else sh_hp
        sh = sh_lp if sh_lp > 0 else sh_hp
        if sh > 0 and isinstance(hp_params[ihx_keys[0]], dict):
            hp_params[ihx_keys[0]]['dT_sh'] = sh

    elif n == 2:
        # ihx1 = LP side, ihx2 = HP side
        for key, sh in zip(ihx_keys, [sh_lp, sh_hp]):
            if sh > 0 and isinstance(hp_params[key], dict):
                hp_params[key]['dT_sh'] = sh

    elif n >= 3:
        # 4 IHX (CascadeIHXPCIHX): ihx1/ihx2 for LP circuit, ihx3/ihx4 for HP circuit
        # Apply sh_lp to all LP-side, sh_hp to all HP-side
        for i, key in enumerate(ihx_keys):
            sh = sh_lp if i < n // 2 else sh_hp
            if sh > 0 and isinstance(hp_params[key], dict):
                hp_params[key]['dT_sh'] = sh


def _patch_cascade_T_mid(hp, t_cascade_hx: float) -> None:
    """
    Setzt die Kaskaden-Mitteltemperatur (T_mid) auf den vom Nutzer gewünschten Wert.

    Mechanismus: Das heatpumps-Modell berechnet T_mid in init_simulation() als
        self.T_mid = (B2.T + C3.T) / 2
    und nutzt es dann in design_simulation() für:
        A4.T = self.T_mid − inter.ttd_u / 2

    Da run_model() init_simulation() und design_simulation() sequentiell aufruft
    und wir T_mid nicht über hp_params steuern können, wird hier init_simulation()
    als Instanzmethode überschrieben: die Original-Funktion läuft durch und setzt
    danach hp.T_mid = t_cascade_hx, bevor design_simulation() gestartet wird.

    Ergebnis: A4.T = t_cascade_hx − ttd_u/2 (bei ttd_u=4K → A4.T = T_casc − 2K)
    """
    _orig_init = hp.init_simulation

    def _patched_init(**kwargs):
        _orig_init(**kwargs)        # originale init_simulation (setzt T_mid aus B2/C3)
        hp.T_mid = float(t_cascade_hx)  # überschreiben mit Nutzerwert

    hp.init_simulation = _patched_init


def _parametrize_hthp(
    hp_params:  dict,
    case_cfg:   dict,
    params:     dict,
    is_cascade: bool,
):
    """
    Überträgt Questionnaire-Werte in das heatpumps-Parameterset.

    Parameter-Schema (aus params_hp_simple.json / params_hp_cascade.json):
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Quelle  (Heat Source / Evaporator)                                 │
    │    hp_params['B1']['T']    Quelleintritt  [°C]                      │
    │    hp_params['B1']['p']    Quell-Druck    [bar]                     │
    │    hp_params['B2']['T']    Quellaustritt  [°C]                      │
    │                                                                     │
    │  Senke  (Heat Sink / Condenser)                                     │
    │    hp_params['C1']['T']    Senkeneintritt  [°C]  (Rücklauf)         │
    │    hp_params['C3']['T']    Senkenaustritt  [°C]  (Vorlauf)          │
    │    hp_params['C3']['p']    Senkendruck     [bar]                    │
    │                                                                     │
    │  Leistung                                                           │
    │    hp_params['cons']['Q']  Wärmebedarf  [W]  →  NEGATIV übergeben   │
    │                                                                     │
    │  Single-stage Kältemittel                                           │
    │    hp_params['setup']['refrig']   z.B. 'R600a'                      │
    │    hp_params['fluids']['wf']      identisch mit refrig              │
    │                                                                     │
    │  Cascade Kältemittel                                                │
    │    hp_params['setup']['refrig1']  Niedertemperaturkreis             │
    │    hp_params['setup']['refrig2']  Hochtemperaturkreis               │
    │    hp_params['fluids']['wf1']     = refrig1                         │
    │    hp_params['fluids']['wf2']     = refrig2                         │
    │                                                                     │
    │  Verdichter                                                         │
    │    hp_params['comp']['eta_s']     isentroper Wirkungsgrad [-]       │
    │    hp_params['LT_comp']['eta_s']  (Cascade, Niedertemperatur)       │
    │    hp_params['HT_comp']['eta_s']  (Cascade, Hochtemperatur)         │
    └─────────────────────────────────────────────────────────────────────┘
    """
    app = _val(params, 'application_type')

    T_source_in  = _val(params, 'source_temp_in')  or 85.0
    T_source_out = _val(params, 'source_temp_out') or 70.0
    p_source     = _val(params, 'source_pressure') or 1.013

    hp_params['B1']['T'] = T_source_in
    hp_params['B1']['p'] = p_source
    hp_params['B2']['T'] = T_source_out
    if 'p' in hp_params.get('B2', {}):
        hp_params['B2']['p'] = p_source

    if app == 'Hot water generation':
        T_sink_in  = _val(params, 'hw_temp_inlet')           or 50.0
        T_sink_out = _val(params, 'hw_temp_outlet_required')  or 90.0
        Q_kW       = _val(params, 'hw_heat_power')            or 500.0
        p_sink     = 1.013
    else: # Steam generation
        T_sink_in = _val(params, 'steam_temp_inlet') or T_source_out
        p_sink    = _val(params, 'steam_pressure_outlet') or 3.0
        try:
            from CoolProp.CoolProp import PropsSI
            T_sat = PropsSI('T', 'P', p_sink * 1e5, 'Q', 1, 'Water') - 273.15
            superheat = _val(params, 'steam_superheat') or 0.0
            steam_q   = str(_val(params, 'steam_quality') or '').lower()
            # MIN_SUPERHEAT verhindert exakt-Siedelinien-Problem (T=T_sat & p=p_sat gleichzeitig)
            MIN_SUPERHEAT = 1.0
            T_sink_out = T_sat + max(superheat, MIN_SUPERHEAT)
        except Exception:
            T_sink_out = T_sink_in + 20.0
        Q_kW = _val(params, 'steam_heat_power') or 0.0
        if not Q_kW:
            m_kg_h = _val(params, 'steam_mass_flow_inlet') or 0.0
            if m_kg_h:
                try:
                    from CoolProp.CoolProp import PropsSI
                    # Gesamtenthalpiehub: Speisewasser (T_sink_in, p_sink) → Sattdampf (p_sink)
                    h_steam = PropsSI('H', 'P', p_sink * 1e5, 'Q', 1, 'Water') / 1000.0  # kJ/kg
                    h_fw    = PropsSI('H', 'T', T_sink_in + 273.15, 'P', p_sink * 1e5, 'Water') / 1000.0
                    Q_kW = (m_kg_h / 3600.0) * (h_steam - h_fw)
                except Exception:
                    Q_kW = 500.0
            else:
                Q_kW = 500.0

    hp_params['C1']['T'] = T_sink_in
    hp_params['C3']['T'] = T_sink_out
    hp_params['C3']['p'] = p_sink
    if 'p' in hp_params.get('C1', {}):
        hp_params['C1']['p'] = p_sink

    hp_params['cons']['Q'] = -(Q_kW * 1000.0)

    # -- Terminal Temperature Differences (TTD) --------------------------------
    # cond/trans_hx:  ttd_u = 5 K
    # evap:           ttd_l = 0 K
    # inter (Kaskaden-HX): nur ttd_u = 4 K (Pinch); kein ttd_l-Parameter
    _TTD_U, _TTD_L = 5.0, 0.0
    _CASCADE_PINCH  = 4.0
    for _ck in ('cond', 'trans_hx', 'gas_cooler'):
        if isinstance(hp_params.get(_ck), dict):
            hp_params[_ck]['ttd_u'] = _TTD_U
    for _ck in ('evap',):
        if isinstance(hp_params.get(_ck), dict):
            hp_params[_ck]['ttd_l'] = _TTD_L
    # 'inter' ist der korrekte Kaskadenparameter-Key (Condenser in TESPy-Modell)
    for _ck in ('inter', 'casc_HX', 'casc_hx', 'inter_hx'):
        if isinstance(hp_params.get(_ck), dict):
            hp_params[_ck]['ttd_u'] = _CASCADE_PINCH

    # -- Wirkungsgrade ----------------------------------------------------------
    # Unterstuetzt: eta_s (einheitlich), eta_s_lt/eta_s_ht (Kaskade getrennt),
    # eta_s_1/eta_s_2 (Alias). UI uebergibt effs=[lp, hp] -> eta_s_lt / eta_s_ht.
    eta_s    = case_cfg.get('eta_s', 0.75)
    eta_s_lt = case_cfg.get('eta_s_lt', case_cfg.get('eta_s_1', eta_s))
    eta_s_ht = case_cfg.get('eta_s_ht', case_cfg.get('eta_s_2', eta_s))

    if is_cascade:
        r1 = _coolprop_name(case_cfg.get('refrigerant1', 'R717'))
        r2 = _coolprop_name(case_cfg.get('refrigerant2', 'R245fa'))
        hp_params['setup']['refrig1'] = r1
        hp_params['setup']['refrig2'] = r2
        hp_params['fluids']['wf1']    = r1
        hp_params['fluids']['wf2']    = r2
        if 'LT_comp' in hp_params:
            hp_params['LT_comp']['eta_s'] = eta_s_lt   # Niedertemperaturkreis
        if 'HT_comp' in hp_params:
            hp_params['HT_comp']['eta_s'] = eta_s_ht   # Hochtemperaturkreis
    else:
        refrig = _coolprop_name(case_cfg.get('refrigerant', 'R717'))
        hp_params['setup']['refrig'] = refrig
        hp_params['fluids']['wf']    = refrig
        if 'comp' in hp_params:
            hp_params['comp']['eta_s'] = eta_s
        if 'comp1' in hp_params:
            hp_params['comp1']['eta_s'] = eta_s
        if 'comp2' in hp_params:
            hp_params['comp2']['eta_s'] = eta_s

    # -- Kaeltemittelueberhitzung am Verdichtereintritt (nur IHX-Modelle) ----
    # overheats: [sh_lp] or [sh_lp, sh_hp] depending on which IHX sides exist.
    # Fallback 'superheat' scalar → treated as LP-side.
    overheats = case_cfg.get('overheats') or []
    if case_cfg.get('superheat') is not None:
        sh_lp = float(case_cfg['superheat'])
        sh_hp = 0.0
    else:
        sh_lp = float(overheats[0]) if len(overheats) >= 1 else 0.0
        sh_hp = float(overheats[1]) if len(overheats) >= 2 else 0.0
    _apply_hthp_superheat(hp_params, case_cfg.get('model', ''), sh_lp=sh_lp, sh_hp=sh_hp)


def _extract_hthp_results(
    hp,
    case_id:    str,
    case_cfg:   dict,
    hp_params:  dict,
    is_cascade: bool,
) -> CaseResult:
    """
    Liest Ergebnisse aus dem ausgeführten heatpumps-Modell.

    Verfügbare Attribute nach hp.run_model()  (aus HeatPumpBase):
        hp.cop              COP (simuliert)  [-]
        hp.cop_lorenz       Lorenz-COP  [-]
        hp.eta_lorenz       Lorenz-Wirkungsgrad  [-]
        hp.cop_carnot       Carnot-COP  [-]
        hp.epsilon          Exergetischer Wirkungsgrad  [-]
        hp.buses['heat output'].P.val    Wärmestrom  [W]  (negativ!)
        hp.buses['power input'].P.val    Verdichterleistung  [W]
        hp.nw                            TESPy-Netzwerk (alle Zustände)
    """
    Q_con  = abs(hp.buses['heat output'].P.val) / 1000.0   # W → kW
    W_comp = hp.buses['power input'].P.val / 1000.0        # W → kW

    extra = {
        'model':       case_cfg.get('model'),
        'cop_lorenz':  getattr(hp, 'cop_lorenz',  None),
        'eta_lorenz':  getattr(hp, 'eta_lorenz',  None),
        'cop_carnot':  getattr(hp, 'cop_carnot',  None),
    }
    if is_cascade:
        extra['refrigerant1'] = case_cfg.get('refrigerant1')
        extra['refrigerant2'] = case_cfg.get('refrigerant2')
    else:
        extra['refrigerant'] = case_cfg.get('refrigerant')

    return CaseResult(
        case_id=case_id,
        case_type=CaseType.HTHP,
        status=CaseStatus.SUCCESS,
        model_instance=hp,
        params_used=hp_params,
        COP=hp.cop,
        Q_con=Q_con,
        W_comp=W_comp,
        epsilon=getattr(hp, 'epsilon', None),
        T_hot_out=hp_params.get('C3', {}).get('T'),
        T_cold_in=hp_params.get('B1', {}).get('T'),
        extra=extra,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MVR  –  myModels/mvrMultiStage.py
# ══════════════════════════════════════════════════════════════════════════════

def _run_mvr(case_cfg: dict, params: dict, verbose: bool) -> CaseResult:
    """
    Startet das MVR-Modell (Mechanische Brüdenverdichtung).

    Das MVR-Modell arbeitet mit Wasserdampf als Arbeitsmedium – kein Kältemittel.
    Es komprimiert Brüden aus einem Verdampfer auf den Zielbetriebsdruck.

    Ablauf
    ------
    1.  MVRMultiStage aus myModels importieren
    2.  Parameter aus Questionnaire zusammenstellen 
    3.  Modell ausführen  →  mvr.run()
    4.  Ergebnisse extrahieren
    """
    case_id = case_cfg['id']

    if verbose:
        n_stg = case_cfg.get('n_stages')
        print(f"  Stufen: {n_stg if n_stg else 'automatisch'}")

    # ── 1. Import ─────────────────────────────────────────────────────────
    try:
        from my_models.MVRMultiStage import MVRMultiStage
    except ImportError as e:
        raise ImportError(
            f"myModels.mvrMultiStage nicht gefunden. "
            f"Pfad korrekt (myModels/ im Projektverzeichnis)?\n{e}"
        )

    # ── 2. Parameter zusammenstellen ──────────────────────────────────────
    mvr_params: dict = {}
    _parametrize_mvr(mvr_params, case_cfg, params)

    # ── 3. Modell ausführen ───────────────────────────────────────────────
    mvr = MVRMultiStage(mvr_params)
    mvr.run_model(print_results=False)

    # ── 4. Ergebnisse extrahieren ─────────────────────────────────────────
    return _extract_mvr_results(mvr, case_id, case_cfg, mvr_params)


def _parametrize_mvr(mvr_params: dict, case_cfg: dict, params: dict):
    """
    Befüllt das MVR-Parameterset aus dem Questionnaire.

    Orientiert sich an create_multistage_params() aus main_alt.py und den
    in MVRMultiStage.init_simulation() erwarteten Schlüsseln:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  setup                                                              │
    │    params['setup']['name']          Bezeichner                      │
    │                                                                     │
    │  fluids                                                             │
    │    params['fluids']['wf']           Arbeitsmedium  (='Water')       │
    │                                                                     │
    │  inlet  –  Brüdeneintritt am MVR                                    │
    │    params['inlet']['T']   Sättigungstemperatur bei p_inlet  [°C]    │
    │    params['inlet']['p']   Eintrittsdruck (Brüdendruck)      [bar]   │
    │    params['inlet']['m']   Massenstrom                       [kg/s]  │
    │                                                                     │
    │  outlet                                                             │
    │    params['outlet']['p_out']  Zieldruck (HD-Dampf)          [bar]   │
    │                                                                     │
    │  compressors  –  isentrope Wirkungsgrade                            │
    │    params['compressors']['eta_s']        einheitlich alle Stufen    │
    │    params['compressors']['eta_s_1'] ...  stufenweise (optional)     │
    │                                                                     │
    │  intercoolers                                                       │
    │    params['intercoolers']['superheat']   Überhitzung nach Injektion │
    │                                          [K]  (0 = Sattdampf)      │
    │                                                                     │
    │  cooling_water                                                      │
    │    params['cooling_water']['T_in']   Einspritzwassertemperatur [°C] │
    └─────────────────────────────────────────────────────────────────────┘

    Stufenzahl und Verdichter-Wirkungsgrade kommen aus case_cfg –
    sie wurden bereits bei der Case-Erzeugung festgelegt.
    """
    from CoolProp.CoolProp import PropsSI

    # ── Drücke ────────────────────────────────────────────────────────────
    p_inlet  = _val(params, 'steam_pressure_inlet')
    p_outlet = _val(params, 'steam_pressure_outlet')

    if p_inlet is None or p_inlet <= 0:
        raise ValueError(
            "_parametrize_mvr: 'steam_pressure_inlet' nicht im Questionnaire!"
        )
    if p_outlet is None or p_outlet <= 0:
        raise ValueError(
            "_parametrize_mvr: 'steam_pressure_outlet' nicht im Questionnaire!"
        )

    # ── Massenstrom (Ziel-Ausgangsmassenstrom) ────────────────────────────
    # Vorgabe ist der gewünschte Dampf-Ausgangsmassenstrom am Zieldruck.
    # Der Eintrittsmassenstrom wird in MVRMultiStage.init_simulation zurückgerechnet.
    m_kg_h = _val(params, 'steam_mass_flow_inlet')
    if m_kg_h and m_kg_h > 0:
        m_target = m_kg_h / 3600.0
    else:
        # Aus Wärmeleistung und Verdampfungsenthalpie bei p_outlet
        Q_kW = _val(params, 'steam_heat_power') or 0.0
        if Q_kW > 0:
            h_fg = (PropsSI('H', 'P', p_outlet * 1e5, 'Q', 1, 'Water')
                    - PropsSI('H', 'P', p_outlet * 1e5, 'Q', 0, 'Water'))  # J/kg
            m_target = (Q_kW * 1000.0) / h_fg if h_fg > 0 else 1.0
        else:
            raise ValueError(
                "_parametrize_mvr: Weder 'steam_mass_flow_inlet' noch 'steam_heat_power' "
                "im Questionnaire gefunden – Ziel-Massenstrom nicht bestimmbar!"
            )

    # ── Eingangstemperatur: Sättigungstemperatur bei p_inlet ──────────────
    # MVR arbeitet mit gesättigtem Dampf aus dem Verdampfer
    T_inlet = _val(params, 'steam_temp_inlet')
    if T_inlet is None:
        T_inlet = PropsSI('T', 'P', p_inlet * 1e5, 'Q', 1, 'Water') - 273.15

    MIN_SUPERHEAT = 1.0   # K
    T_sat_in = PropsSI('T', 'P', p_inlet * 1e5, 'Q', 1, 'Water') - 273.15
    T_inlet_mvr = max(T_inlet, T_sat_in + MIN_SUPERHEAT)

    # Hinweis: Vorwärmleistung (falls Speisewasser unterkühlt ist) wird in
    # MVRMultiStage.init_simulation berechnet und als mvr.Q_preheat_kW bereitgestellt.

    # ── Stufenzahl ────────────────────────────────────────────────────────
    # Wenn n_stages nicht explizit gesetzt: automatisch per Sättigungstemperatur-Methode
    n_stages = case_cfg.get('n_stages')
    if n_stages is None:
        try:
            from utils.mvr_stage_selection import determine_n_stages
            n_stages, _stg_info = determine_n_stages(
                p_in=p_inlet,
                p_out=p_outlet,
                dT_per_stage=case_cfg.get('dT_per_stage'),       # K; None → 10 K
                PR_max_per_stage=case_cfg.get('PR_max_per_stage'),
                design_philosophy=case_cfg.get('design_philosophy', 'standard'),
                verbose=True,
            )
        except ImportError:
            # Fallback: einfache Druckverhältnis-Schätzung
            import math
            n_stages = max(1, math.ceil(math.log(p_outlet / p_inlet) / math.log(1.6)))

    # Prüfe ob stufenweise eta_s vorhanden (eta_s_1, eta_s_2, …)
    per_stage = {
        i: case_cfg[f'eta_s_{i}']
        for i in range(1, n_stages + 1)
        if f'eta_s_{i}' in case_cfg
    }

    if per_stage:
        # Stufenweise: alle Stufen müssen vorhanden sein
        eta_s_default = case_cfg.get('eta_s', 0.85)
        compressors = {
            f'eta_s_{i}': per_stage.get(i, eta_s_default)
            for i in range(1, n_stages + 1)
        }
    else:
        # Einheitlicher Wirkungsgrad für alle Stufen
        compressors = {'eta_s': case_cfg['eta_s']}

    # ── Parameterset zusammenstellen ──────────────────────────────────────
    mvr_params.update({
        'setup': {
            'name': case_cfg['id'],
        },
        'fluids': {
            'wf': 'Water',
        },
        'inlet': {
            'T': T_inlet_mvr,   # T_sat(p_in) + MIN_SUPERHEAT – Eintrittstemperatur Stufe 1
            'p': p_inlet,
            # Kein 'm' hier – wird in init_simulation aus m_target zurückgerechnet
        },
        'outlet': {
            'p_out': p_outlet,
            'm':     m_target,  # Ziel-Ausgangsmassenstrom [kg/s]
        },
        'n_stages': n_stages,
        'compressors': compressors,
        'intercoolers': {
            # mvr_sh kommt aus dem UI (case_cfg['mvr_sh']); superheat_intercooler als Legacy-Fallback
            'superheat': max(1.0, float(case_cfg.get('mvr_sh', case_cfg.get('superheat_intercooler', 0)))),
        },
        'cooling_water': {
            # Einspritzwasser = Speisewasser der Senke (Sink-Eintritt), nicht Quelltemperatur
            'T_in': _val(params, 'steam_temp_inlet') or _val(params, 'source_temp_in') or 15.0,
        },
    })


def _extract_mvr_results(
    mvr,
    case_id:    str,
    case_cfg:   dict,
    mvr_params: dict,
) -> CaseResult:
    """
    Liest Ergebnisse aus dem ausgeführten MVRMultiStage-Modell.

    Relevante Attribute nach mvr.run_model()  (aus MVRMultiStage / MVRBase):
        mvr.total_power          Gesamte elektrische Verdichterleistung  [kW]
        mvr.SEI                  Specific Energy Input  [kWh/kg Dampf_out]
        mvr.SEI_MJ_per_kg        SEI in  MJ/kg
        mvr.compression_ratio    Gesamt-Druckverhältnis  [-]
        mvr.specific_work        Spez. Verdichtungsarbeit  [kJ/kg]
        mvr.m_design             Eintrittsmassenstrom (zurückgerechnet)  [kg/s]
        mvr.m_target             Ziel-Ausgangsmassenstrom  [kg/s]
        mvr.Q_preheat_kW         Vorwärmleistung (Speisewasser → Sattdampf) [kW]
        mvr.total_water_injected Gesamtmenge Einspritzwasser  [kg/s]
        mvr.water_injection_rates Liste Einspritzwasser je Zwischenstufe  [kg/s]
        mvr.stage_results        Dict mit Ergebnissen je Stufe
    """
    n_stages = mvr.n_stages
    W_comp   = float(mvr.total_power)   # kW  (compressor only)
    SEI      = float(mvr.SEI)           # kWh/kg (bezogen auf m_out)

    # Electric pre-heating (liquid feed water → steam) if applicable
    # Q_preheat_kW is set by MVRMultiStage.init_simulation using the back-calculated m_inlet
    W_preheat   = float(getattr(mvr, 'Q_preheat_kW', 0.0))
    W_total     = W_comp + W_preheat
    has_preheat = W_preheat > 0.0

    # Inlet (back-calculated) and outlet (target) mass flows
    m_steam_in  = float(mvr.m_design)                                  # back-calculated inlet
    m_steam_out = float(getattr(mvr, 'm_target', mvr.m_design))        # target outlet

    # Recalculate SEI including pre-heater, referenced to outlet steam production
    SEI_total = (W_total / (m_steam_out * 3600)) if m_steam_out > 0 else SEI

    # Austrittstemperatur letzte Verdichterstufe
    T_out = float(mvr.comps[f'comp_{n_stages}'].outl[0].T.val)

    # COP und Q_heat aus Modell lesen (float-NaN-Prüfung via != self)
    _cop  = getattr(mvr, 'COP',    None)
    _qh   = getattr(mvr, 'Q_heat', None)
    cop_val  = float(_cop) if (_cop  is not None and float(_cop)  == float(_cop))  else None
    q_con_val= float(_qh)  if (_qh   is not None and float(_qh)   == float(_qh))   else None

    # Adjust COP denominator to include pre-heating
    if cop_val is not None and W_total > 0 and q_con_val is not None:
        cop_val = q_con_val / W_total

    extra = {
        'SEI_compressor_kWh_per_kg':  SEI,
        'SEI_total_kWh_per_kg':       SEI_total,
        'SEI_MJ_per_kg':              float(mvr.SEI_MJ_per_kg),
        'compression_ratio':          float(mvr.compression_ratio),
        'specific_work_kJ_per_kg':    float(mvr.specific_work),
        'm_steam_in_kg_s':            m_steam_in,
        'm_steam_out_kg_s':           m_steam_out,
        'total_water_injected_kg_s':  float(mvr.total_water_injected),
        'water_injection_rates_kg_s': [float(x) for x in mvr.water_injection_rates],
        'T_steam_out':                T_out,
        'n_stages':                   n_stages,
        'W_compressor_kW':            W_comp,
    }
    T_fw = float(mvr_params['cooling_water']['T_in'])
    if has_preheat:
        extra['W_electric_preheater_kW'] = W_preheat
        extra['T_feedwater_in_C']        = T_fw
        extra['T_steam_mvr_inlet_C']     = float(mvr_params['inlet']['T'])
        extra['note'] = (
            f"Feed water at {T_fw:.1f} °C is subcooled – "
            f"electric pre-heater ({W_preheat:.1f} kW) included in W_total."
        )

    return CaseResult(
        case_id=case_id,
        case_type=CaseType.MVR,
        status=CaseStatus.SUCCESS,
        model_instance=mvr,
        params_used=mvr_params,
        COP=cop_val,
        Q_con=q_con_val,
        W_comp=W_total,
        SEI=SEI_total,
        T_hot_out=T_out,
        T_cold_in=float(case_cfg.get('_T_water_in', mvr_params['cooling_water']['T_in'])),
        extra=extra,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HTHP + MVR  –  myModels/hthp_mvr.py
# ══════════════════════════════════════════════════════════════════════════════

def _run_hthp_mvr(case_cfg: dict, params: dict, verbose: bool) -> CaseResult:
    """
    Hybrid HTHP+MVR  –  sequentielle Ausführung beider Teilmodelle.

    Ablauf
    ------
    1.  HTHP-Parametrierung  →  heatpumps-Modell erzeugt Dampf bei p_intermediate
    2.  HTHP-Simulation via heatpumps-Bibliothek
    3.  Dampfmassenstrom am Zwischendruck aus HTHP-Kondensationsleistung ableiten
    4.  MVR-Parametrierung  →  MVRMultiStage komprimiert von p_int auf p_final
    5.  MVR-Simulation
    6.  Kombiniertes Ergebnis zusammenstellen
    """
    from CoolProp.CoolProp import PropsSI

    case_id = case_cfg['id']

    # ── Imports ───────────────────────────────────────────────────────────
    try:
        import heatpumps.src.heatpumps.models as _hp_models
        from heatpumps.src.heatpumps.parameters import get_params
    except ImportError as e:
        raise ImportError(f"heatpumps-Bibliothek nicht gefunden.\n{e}")

    try:
        from my_models.MVRMultiStage import MVRMultiStage
    except ImportError as e:
        raise ImportError(f"myModels.MVRMultiStage nicht gefunden.\n{e}")

    # ── 1+2. HTHP ─────────────────────────────────────────────────────────
    model_name = case_cfg.get('model', 'HeatPumpSimple')
    is_cascade = model_name in _CASCADE_MODELS
    needs_econ = model_name in _ECON_MODELS

    if verbose:
        refrig_str = (
            f"{case_cfg.get('refrigerant1','?')} / {case_cfg.get('refrigerant2','?')}  [cascade]"
            if is_cascade else case_cfg.get('refrigerant', '—')
        )
        print(f"  HTHP-Modell: {model_name}  Kältemittel: {refrig_str}  "
              f"n_stages MVR: {case_cfg.get('n_stages', '?')}")

    if needs_econ:
        hp_params = get_params(model_name, econ_type=case_cfg.get('econ_type', 'closed'))
    else:
        hp_params = get_params(model_name)

    _parametrize_hthp_for_hybrid(hp_params, case_cfg, params, is_cascade)

    ModelClass = getattr(_hp_models, model_name, None)
    if ModelClass is None:
        raise ValueError(f"Modell '{model_name}' nicht in heatpumps.models gefunden.")

    # ── Werte VOR run_model() sichern – TESPy überschreibt hp_params nach dem
    #    Lösen mit SI-Einheiten (K statt °C, Pa statt bar).
    p_intermediate = float(hp_params['C3']['p'])   # bar
    T_fw           = float(hp_params['C1']['T'])   # °C

    _model_kwargs = {'econ_type': case_cfg.get('econ_type', 'closed')} if needs_econ else {}
    hp = ModelClass(params=hp_params, **_model_kwargs)
    # Kaskadentemperatur: T_mid nach init_simulation() überschreiben
    if is_cascade and case_cfg.get('t_cascade_hx') is not None:
        _patch_cascade_T_mid(hp, case_cfg['t_cascade_hx'])
    hp.run_model()

    # ── 3. Dampfmassenstrom am Zwischendruck ableiten ────────────────────────
    Q_cond_kW = abs(hp.buses['heat output'].P.val) / 1000.0   # W → kW

    h_steam_int  = PropsSI('H', 'P', p_intermediate * 1e5, 'Q', 1, 'Water') / 1000.0
    h_water_feed = PropsSI('H', 'T', T_fw + 273.15, 'P', p_intermediate * 1e5, 'Water') / 1000.0
    dh_evap      = max(h_steam_int - h_water_feed, 0.1)   # kJ/kg, Schutz vor /0
    m_steam_int  = Q_cond_kW / dh_evap                    # kg/s

    if verbose:
        print(f"  HTHP: COP={hp.cop:.3f}  Q_cond={Q_cond_kW:.1f} kW  "
              f"m_steam@{p_intermediate:.2f}bar = {m_steam_int*3600:.1f} kg/h")

    # ── 4+5. MVR ──────────────────────────────────────────────────────────
    mvr_params: dict = {}
    _parametrize_mvr_for_hybrid(mvr_params, case_cfg, params, p_intermediate, m_steam_int)

    mvr = MVRMultiStage(mvr_params)
    mvr.run_model(print_results=False)

    # ── 6. Ergebnisse zusammenstellen ────────────────────────────────────
    return _extract_hthp_mvr_results(hp, mvr, case_id, case_cfg, hp_params, mvr_params)


def _parametrize_hthp_for_hybrid(
    hp_params:  dict,
    case_cfg:   dict,
    params:     dict,
    is_cascade: bool,
):
    """
    Setzt HTHP-Parameter für den Hybridfall:
    Senke = Dampferzeugung bei p_intermediate  (nicht Enddampfdruck!).

    Der HTHP kondensiert bei p_intermediate – der MVR übernimmt danach
    die weitere Verdichtung auf den Zieldruck.
    """
    from CoolProp.CoolProp import PropsSI
    MIN_SUPERHEAT = 1.0   # K  – verhindert exakt-Siedelinien-Problem

    # Quelle
    T_source_in  = _val(params, 'source_temp_in')  or 85.0
    T_source_out = _val(params, 'source_temp_out') or 70.0
    p_source     = _val(params, 'source_pressure') or 1.013

    hp_params['B1']['T'] = T_source_in
    hp_params['B1']['p'] = p_source
    hp_params['B2']['T'] = T_source_out
    if 'p' in hp_params.get('B2', {}):
        hp_params['B2']['p'] = p_source

    # Zwischendruck  (HTHP-Kondensation)
    p_final        = _val(params, 'steam_pressure_outlet') or 3.0
    p_intermediate = case_cfg.get('p_intermediate', p_final * 0.5)
    p_intermediate = max(p_intermediate, 1.013)   # mindestens atm

    T_sat_int    = PropsSI('T', 'P', p_intermediate * 1e5, 'Q', 1, 'Water') - 273.15
    T_sink_out   = T_sat_int + MIN_SUPERHEAT       # C3: knapp über Siedelinie

    # Speisewasser-Eintrittstemperatur (C1)
    T_fw = _val(params, 'hw_temp_inlet') or _val(params, 'steam_temp_inlet') or 20.0
    T_fw = min(T_fw, T_sat_int - 5.0)             # muss unter Sättigungstemperatur liegen

    hp_params['C1']['T'] = T_fw
    hp_params['C3']['T'] = T_sink_out
    hp_params['C3']['p'] = p_intermediate
    if 'p' in hp_params.get('C1', {}):
        hp_params['C1']['p'] = p_intermediate

    # Wärmeleistung der HTHP-Stufe:
    # Korrekt: Enthalpiedifferenz von Speisewasser (T_fw, p_int) bis Sattdampf (p_int)
    # — nicht h_fg(p_final), da HTHP bei p_intermediate kondensiert.
    m_kg_h = _val(params, 'steam_mass_flow_inlet') or 0.0
    Q_kW   = _val(params, 'steam_heat_power') or 0.0
    if m_kg_h > 0:
        h_steam_int  = PropsSI('H', 'P', p_intermediate * 1e5, 'Q', 1, 'Water') / 1000.0  # kJ/kg
        h_water_int  = PropsSI('H', 'T', T_fw + 273.15, 'P', p_intermediate * 1e5, 'Water') / 1000.0
        dh_hthp      = max(h_steam_int - h_water_int, 1.0)  # kJ/kg (Schutz)
        Q_kW = (m_kg_h / 3600.0) * dh_hthp
    if Q_kW <= 0:
        Q_kW = 500.0   # Fallback

    hp_params['cons']['Q'] = -(Q_kW * 1000.0)   # W, negativ

    # -- Terminal Temperature Differences (TTD) --------------------------------
    _TTD_U, _TTD_L = 5.0, 0.0
    _CASCADE_PINCH  = 4.0
    for _ck in ('cond', 'trans_hx', 'gas_cooler'):
        if isinstance(hp_params.get(_ck), dict):
            hp_params[_ck]['ttd_u'] = _TTD_U
    for _ck in ('evap',):
        if isinstance(hp_params.get(_ck), dict):
            hp_params[_ck]['ttd_l'] = _TTD_L
    for _ck in ('inter', 'casc_HX', 'casc_hx', 'inter_hx'):
        if isinstance(hp_params.get(_ck), dict):
            hp_params[_ck]['ttd_u'] = _CASCADE_PINCH

    # -- Kaeltemittel & Verdichter -----------------------------------------
    eta_s    = case_cfg.get('eta_s_hthp', case_cfg.get('eta_s', 0.85))
    eta_s_lt = case_cfg.get('eta_s_lt', case_cfg.get('eta_s_1', eta_s))
    eta_s_ht = case_cfg.get('eta_s_ht', case_cfg.get('eta_s_2', eta_s))
    if is_cascade:
        r1 = _coolprop_name(case_cfg.get('refrigerant1', 'R717'))
        r2 = _coolprop_name(case_cfg.get('refrigerant2', 'R245fa'))
        hp_params['setup']['refrig1'] = r1
        hp_params['setup']['refrig2'] = r2
        hp_params['fluids']['wf1']    = r1
        hp_params['fluids']['wf2']    = r2
        if 'LT_comp' in hp_params:
            hp_params['LT_comp']['eta_s'] = eta_s_lt
        if 'HT_comp' in hp_params:
            hp_params['HT_comp']['eta_s'] = eta_s_ht
    else:
        refrig = _coolprop_name(case_cfg.get('refrigerant', 'R717'))
        hp_params['setup']['refrig'] = refrig
        hp_params['fluids']['wf']    = refrig
        if 'comp' in hp_params:
            hp_params['comp']['eta_s'] = eta_s

    # -- Kaeltemittelueberhitzung ------------------------------------------
    overheats = case_cfg.get('overheats') or []
    if case_cfg.get('superheat') is not None:
        sh_lp = float(case_cfg['superheat'])
        sh_hp = 0.0
    else:
        sh_lp = float(overheats[0]) if len(overheats) >= 1 else 0.0
        sh_hp = float(overheats[1]) if len(overheats) >= 2 else 0.0
    _apply_hthp_superheat(hp_params, case_cfg.get('model', ''), sh_lp=sh_lp, sh_hp=sh_hp)


def _parametrize_mvr_for_hybrid(
    mvr_params:    dict,
    case_cfg:      dict,
    params:        dict,
    p_intermediate: float,
    m_steam_int:   float,
):
    """
    Setzt MVR-Parameter für den Hybridfall.

    Eingang: Dampf bei p_intermediate  (Ausgang des HTHP-Kondensators)
    Ausgang: Dampf bei p_steam_final

    m_steam_int kommt direkt aus dem HTHP-Ergebnis.
    """
    from CoolProp.CoolProp import PropsSI
    MIN_SUPERHEAT = 1.0   # K

    p_final  = _val(params, 'steam_pressure_outlet') or 3.0
    T_inlet  = PropsSI('T', 'P', p_intermediate * 1e5, 'Q', 1, 'Water') - 273.15 + MIN_SUPERHEAT

    n_stages = case_cfg.get('n_stages')
    if n_stages is None:
        try:
            from mvr_stage_selection import determine_n_stages
            n_stages, _ = determine_n_stages(
                p_in=p_intermediate,
                p_out=p_final,
                dT_per_stage=case_cfg.get('dT_per_stage'),
                PR_max_per_stage=case_cfg.get('PR_max_per_stage'),
                design_philosophy=case_cfg.get('design_philosophy', 'standard'),
                verbose=True,
            )
        except ImportError:
            import math
            n_stages = max(1, math.ceil(math.log(p_final / p_intermediate) / math.log(1.6)))

    per_stage = {
        i: case_cfg[f'eta_s_mvr_{i}']
        for i in range(1, n_stages + 1)
        if f'eta_s_mvr_{i}' in case_cfg
    }
    if per_stage:
        compressors = {f'eta_s_{i}': per_stage.get(i, case_cfg.get('eta_s_mvr', 0.85))
                       for i in range(1, n_stages + 1)}
    else:
        compressors = {'eta_s': case_cfg.get('eta_s_mvr', case_cfg.get('eta_s', 0.85))}

    mvr_params.update({
        'setup':  {'name': f"{case_cfg['id']}_mvr"},
        'fluids': {'wf': 'Water'},
        'inlet':  {'T': T_inlet, 'p': p_intermediate},
        'outlet': {'p_out': p_final, 'm': m_steam_int},
        'n_stages': n_stages,
        'compressors': compressors,
        'intercoolers': {'superheat': max(1.0, float(case_cfg.get('mvr_sh', case_cfg.get('superheat_intercooler', 0))))},
        'cooling_water': {'T_in': _val(params, 'steam_temp_inlet') or _val(params, 'source_temp_in') or 15.0},
    })


def _extract_hthp_mvr_results(
    hp,
    mvr,
    case_id:    str,
    case_cfg:   dict,
    hp_params:  dict,
    mvr_params: dict,
) -> CaseResult:
    """
    Kombiniert Ergebnisse aus HTHP- und MVR-Teilmodell.

    Definitionen:
      Q_cond_hthp  = m_design × (h_Dampf_Eintritt_Stufe1 − h_Speisewasser(T_fw, p_int))
                     (HTHP-Kondensator: Speisewasser → Sattdampf bei p_int)
      W_mvr        = Σ_k  m_k × (h_aus,k − h_ein,k)
                     (Summe Verdichterleistungen stufenbezogen, m_k ist der jeweilige
                      Stufenmassenstrom – durch Wassereinspritzung unterschiedlich)
      Q_heat_mvr   = m_aus_gesamt × (h_aus_letzte_Stufe − h_ein_erste_Stufe)
                     (Enthalpiegewinn des Dampfstroms durch die MVR)
      Q_system     = Q_cond_hthp + W_mvr   (1. HS: = m_gesamt × (h_aus − h_Speisewasser))
      SEI_system   = W_total / (m_steam_out × 3600)   [kWh/kg Enddampf]
    """
    from CoolProp.CoolProp import PropsSI

    n_stages = mvr.n_stages

    # ── Zustandsgrößen an Systemgrenzen direkt aus TESPy ─────────────────
    comp_first = mvr.comps['comp_1']
    comp_last  = mvr.comps[f'comp_{n_stages}']

    h_in_first  = float(comp_first.inl[0].h.val)    # kJ/kg  Eintritt Stufe 1
    h_out_last  = float(comp_last.outl[0].h.val)    # kJ/kg  Austritt letzte Stufe
    m_steam_in  = float(comp_first.inl[0].m.val)    # kg/s   Stufenmassenstrom 1 = m_design
    m_steam_out = float(comp_last.outl[0].m.val)    # kg/s   Gesamtmassenstrom Ausgang
    T_final     = float(comp_last.outl[0].T.val)    # °C

    # ── W_MVR: Summe stufenbezogener Verdichterleistungen ─────────────────
    # TESPy berechnet comp.P.val = m_k × (h_aus,k − h_ein,k) für jede Stufe.
    # Der Bus summiert diese korrekt; zur Transparenz auch explizit:
    W_mvr = float(mvr.total_power)   # kW — von TESPy-Bus (= Σ m_k × Δh_k)
    # Explizite Stufensumme zur Verifikation (sollte mit total_power übereinstimmen):
    W_mvr_stages = sum(
        float(mvr.stage_results[f'stage_{k}']['m']) *
        (float(mvr.stage_results[f'stage_{k}']['h_out']) -
         float(mvr.stage_results[f'stage_{k}']['h_in']))
        for k in range(1, n_stages + 1)
        if f'stage_{k}' in mvr.stage_results
    )
    # Falls stage_results befüllt, nimm explizite Summe (konsistenter mit Stufenwerten)
    if abs(W_mvr_stages) > 0:
        W_mvr = W_mvr_stages

    # ── Q_heat_MVR: Enthalpiegewinn des Gesamtmassenstrom durch MVR ────────
    # Q_heat_mvr = m_gesamt_aus × (h_aus_letzte_Stufe − h_ein_erste_Stufe)
    Q_heat_mvr = m_steam_out * (h_out_last - h_in_first)   # kW

    SEI_mvr = float(mvr.SEI)

    # ── Q_cond_hthp: HTHP-Kondensatorleistung auf Basis m_design ──────────
    # Eintrittsseite des Kondensators = Speisewasser (T_fw, p_int)
    # Austrittsseite = Dampf an Eintritt Stufe 1 der MVR (direkt aus TESPy)
    p_int  = float(mvr_params['inlet']['p'])            # bar
    T_fw_C = float(mvr_params['cooling_water']['T_in']) # °C
    h_water_feed = PropsSI('H', 'T', T_fw_C + 273.15, 'P', p_int * 1e5, 'Water') / 1000.0
    # h_eintritt_Stufe1 = h_in_first (bereits aus TESPy)
    dh_cond     = max(h_in_first - h_water_feed, 0.1)   # kJ/kg
    Q_cond_hthp = m_steam_in * dh_cond                  # kW

    # ── HTHP-Verdichterleistung skalieren ────────────────────────────────
    COP_hthp = hp.cop
    W_hthp   = Q_cond_hthp / COP_hthp if COP_hthp > 0 else (
        hp.buses['power input'].P.val / 1000.0
    )

    # ── Systemgrößen ──────────────────────────────────────────────────────
    # Q_system = Q_cond_hthp + W_mvr  (1. HS: = m_out × (h_out − h_fw))
    W_total    = W_hthp + W_mvr
    Q_system   = Q_cond_hthp + W_mvr
    SEI_system = W_total / (m_steam_out * 3600) if m_steam_out > 0 else None
    COP_system = Q_system / W_total             if W_total    > 0 else None

    return CaseResult(
        case_id=case_id,
        case_type=CaseType.HTHP_MVR,
        status=CaseStatus.SUCCESS,
        model_instance=(hp, mvr),
        params_used={'hthp': hp_params, 'mvr': mvr_params},
        COP=COP_system,
        Q_con=Q_system,
        W_comp=W_total,
        SEI=SEI_system,
        T_hot_out=T_final,
        T_cold_in=hp_params.get('B1', {}).get('T'),
        extra={
            'model_hthp':                case_cfg.get('model'),
            'refrigerant':               case_cfg.get('refrigerant'),
            'refrigerant1':              case_cfg.get('refrigerant1'),
            'refrigerant2':              case_cfg.get('refrigerant2'),
            'COP_hthp':                  COP_hthp,
            'COP_system':                COP_system,
            'cop_lorenz_hthp':           getattr(hp, 'cop_lorenz', None),
            'W_hthp_kW':                 W_hthp,
            'W_mvr_kW':                  W_mvr,
            'Q_cond_hthp_kW':            Q_cond_hthp,
            'Q_heat_mvr_kW':             Q_heat_mvr,   # m_out × (h_out_last − h_in_first)
            'SEI_mvr_kWh_per_kg':        SEI_mvr,
            'SEI_system_kWh_per_kg':     SEI_system,
            'n_stages_mvr':              n_stages,
            'm_steam_int_kg_s':          m_steam_in,   # Eintritt MVR-Stufe 1 (= HTHP-Ausgang)
            'm_steam_out_kg_s':          m_steam_out,  # Enddampf am MVR-Ausgang
            'p_intermediate_bar':        mvr_params['inlet']['p'],
            'total_water_injected_kg_s': float(mvr.total_water_injected),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# Hilfsfunktionen
# ══════════════════════════════════════════════════════════════════════════════

# Tabelle: tatsächliche Questionnaire-Schlüssel → kanonischer interner Key.
# Wird benötigt wenn der questionnaire_reader eine ältere oder abweichende
# Feldbezeichnung aus dem Excel-Formular liefert.
_KEY_ALIASES: dict[str, str] = {
    # Quelle
    'Available Heat Capacity':      'source_heat_power',
    # Dampf-Senke (abweichende Label-Varianten im Excel)
    'Feed water/steam temperature': 'steam_temp_inlet',
    'Feed water/steam pressure':    'steam_pressure_inlet',
    'Feed water/supply mass flow':  'steam_mass_flow_inlet',
    'Steam target pressure':        'steam_pressure_outlet',
    'Output steam quality':         'steam_quality',
    'Output steam quality ':        'steam_quality',
}


def _normalize_params(params: dict) -> dict:
    """Übersetzt abweichende Questionnaire-Schlüssel in kanonische interne Keys.

    Der questionnaire_reader bildet Excel-Label direkt auf Python-Keys ab.
    Ändert sich das Formular oder existieren alternative Label-Varianten,
    entstehen fehlende Keys.  Diese Funktion harmonisiert den Dict, ohne den
    questionnaire_reader selbst anzupassen.

    Es wird immer eine Kopie zurückgegeben; das Original bleibt unverändert.
    """
    out = dict(params)
    for alias, canonical in _KEY_ALIASES.items():
        if alias in out and canonical not in out:
            out[canonical] = out[alias]
    return out


def _val(params: dict, key: str) -> Any:
    """Holt Wert sicher aus QuestionnaireReader-params-Dict."""
    entry = params.get(key)
    if entry is None:
        return None
    return entry.get('value') if isinstance(entry, dict) else entry


# Kaeltemittel-Kurzbezeichnungen -> CoolProp-Namen
# Kältemittel-Kurzbezeichnungen -> CoolProp-Namen
# Alle Einträge aus dem erweiterten Katalog (Annex 58 / Annex 68)
_REFRIGERANT_MAP = {
    # Hydrocarbons
    'R290':         'Propane',
    'R600':         'n-Butane',
    'R600a':        'IsoButane',
    'R601':         'n-Pentane',
    'R601a':        'Isopentane',
    'RC270':        'CycloPropane',
    # Natural
    'R717':         'Ammonia',
    'R744':         'CO2',
    'R718':         'Water',
    # HFOs
    'R1234yf':      'R1234yf',
    'R1234ze(E)':   'R1234ze(E)',
    'R1234ze(Z)':   'R1234ze(Z)',
    'R1336mzz(Z)':  'R1336mzz(Z)',
    # HCFOs
    'R1233zd(E)':   'R1233zd(E)',
    'R1224yd(Z)':   'R1224yd(Z)',
    # HFCs
    'R245fa':       'R245fa',
    'R152a':        'R152A',
    'R32':          'R32',
}


def _coolprop_name(refrigerant):
    """Kaeltemittel-Kurzbezeichnung -> CoolProp-Name.
    CoolProp benoetigt z.B. 'IsoButane' statt 'R600a', 'Propane' statt 'R290'.
    Alle anderen Bezeichnungen werden unveraendert weitergegeben.
    """
    return _REFRIGERANT_MAP.get(refrigerant, refrigerant)



def _print_summary(results: dict[str, CaseResult]):
    n_ok   = sum(1 for r in results.values() if r.status == CaseStatus.SUCCESS)
    n_fail = sum(1 for r in results.values() if r.status == CaseStatus.FAILED)
    n_skip = sum(1 for r in results.values() if r.status == CaseStatus.SKIPPED)

    print(f"\n{'═'*70}")
    print(f"  ERGEBNISSE  |  OK: {n_ok}  FAILED: {n_fail}  SKIPPED: {n_skip}")
    print(f"  {'─'*67}")
    print(f"  {'ID':<40} {'Typ':<10} {'COP':>6} {'Q [kW]':>8} {'W [kW]':>8}")
    print(f"  {'─'*67}")

    for r in results.values():
        cop_s = f"{r.COP:.3f}" if r.COP is not None else '—'
        q_s   = f"{r.Q_con:.1f}" if r.Q_con is not None else '—'
        w_s   = f"{r.W_comp:.1f}" if r.W_comp is not None else '—'
        typ   = r.case_type.value if r.case_type else '?'
        flag  = '' if r.status == CaseStatus.SUCCESS else f'  [{r.status.value}]'
        print(f"  {r.case_id:<40} {typ:<10} {cop_s:>6} {q_s:>8} {w_s:>8}{flag}")

    print(f"{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Standalone-Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    """
    Demo: Grundgerüst ohne Modell-Importe testen.
    Alle Cases landen bei FAILED (Import nicht verfügbar),
    aber die Struktur und der Dispatch sind verifizierbar.
    """
    demo_cases = [
        # # ── Single-Stage HTHP ─────────────────────────────────────────────
        # {
        #     'id':          'hthp_simple_R1233ZDE',
        #     'type':        CaseType.HTHP,
        #     'model':       'HeatPumpSimple',
        #     'refrigerant': 'R1233ZDE',
        #     'eta_s':       0.75,
        #     'superheat':   5,            # K – Kältemittelüberhitzung (nur IHX-Modelle)
        # },
        # {
        #     'id':          'hthp_ihx_R1233ZDE',
        #     'type':        CaseType.HTHP,
        #     'model':       'HeatPumpIHX',
        #     'refrigerant': 'R1233ZDE',
        #     'eta_s':       0.75,
        #     'superheat':   5,            # K – Kältemittelüberhitzung (nur IHX-Modelle)
        # },
        # # ── Cascade HTHP ──────────────────────────────────────────────────
        # {
        #     'id':           'hthp_cascade_R1233ZDE_R1233ZDE',
        #     'type':         CaseType.HTHP,
        #     'model':        'HeatPumpCascade',
        #     'refrigerant1': 'R1233ZDE',
        #     'refrigerant2': 'R1233ZDE',
        #     'eta_s':        0.75,
        #     'superheat':   5,            # K – Kältemittelüberhitzung (nur IHX-Modelle)
        # },
        # # ── Econ-Modell ───────────────────────────────────────────────────
        # {
        #     'id':          'hthp_econ_closed_R1233ZDE',
        #     'type':        CaseType.HTHP,
        #     'model':       'HeatPumpEconIHX',
        #     'econ_type':   'closed',
        #     'refrigerant': 'R1233ZDE',
        #     'eta_s':       0.75,
        #     'superheat':   5,            # K – Kältemittelüberhitzung (nur IHX-Modelle)
        # },
        # ── MVR ───────────────────────────────────────────────────────────
        {
            'id':       'mvr_auto',
            'type':     CaseType.MVR,
            'n_stages': None,
            'eta_s':    0.80,
        },
        {
            'id':       'mvr_2stage',
            'type':     CaseType.MVR,
            'n_stages': 2,
            'eta_s':    0.80,
        },
        # ── HTHP + MVR ────────────────────────────────────────────────────
        {
            'id':          'hybrid_R1233ZDE_mvr2',
            'type':        CaseType.HTHP_MVR,
            'model':       'HeatPumpSimple',
            'refrigerant': 'R1233ZDE',
            'eta_s_hthp':  0.75,
            'superheat':   5,            # K – Kältemittelüberhitzung (nur IHX-Modelle)
            'eta_s_mvr':   0.80,
            'n_stages':    None,
        },
        # ── Fehlerhafter Case (kein Typ) ──────────────────────────────────
        {
            'id': 'case_ohne_typ',
        },
    ]
    from utils.questionnaire_reader import QuestionnaireReader
    reader = QuestionnaireReader(r'hthp_project_final\HTHP_questionnaire.xlsx')
    demo_params = reader.get_params()

    calculate_cases(demo_cases, demo_params, verbose=True)