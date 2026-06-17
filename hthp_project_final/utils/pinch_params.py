"""
Pinch Parameter Builder
=======================
Wertet die Parameter aus dem QuestionnaireReader aus und liefert
alle Streams, die für eine Pinch-Analyse benötigt werden.

Jeder Stream hat:
    name        : Bezeichnung
    stream_type : 'hot' oder 'cold'
    T_supply    : Eintrittstemperatur [°C]
    T_target    : Austrittstemperatur [°C]
    CP          : Wärmekapazitätsstrom m_dot * cp [kW/K]
    Q           : Wärmestrom [kW]

Verwendung:
    from questionnaire_reader import QuestionnaireReader
    from pinch_params import PinchParamBuilder

    reader = QuestionnaireReader('HTHP_questionnaire.xlsx')
    params = reader.get_params()

    builder = PinchParamBuilder(params, delta_T_min=10.0)
    pinch_input = builder.get_pinch_params()

    # pinch_input ist dann direkt an den PinchAnalysis-Rechner übergebar:
    # pinch = PinchAnalysis(**pinch_input)
    # pinch.run()

Version: 1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import CoolProp.CoolProp as CP
    _COOLPROP = True
except ImportError:
    _COOLPROP = False


# ══════════════════════════════════════════════════════════════════════════════
# Stream-Datenklasse
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PinchStream:
    """
    Repräsentiert einen Prozessstrom für die Pinch-Analyse.

    Felder:
        name        : Bezeichnung des Stroms
        stream_type : 'hot'  → wird abgekühlt (gibt Wärme ab)
                      'cold' → wird erwärmt   (nimmt Wärme auf)
        T_supply    : Eintrittstemperatur [°C]
        T_target    : Austrittstemperatur [°C]
        CP          : Wärmekapazitätsstrom m_dot·cp [kW/K]
        Q           : Wärmestrom |CP · ΔT| [kW]
        note        : Optionaler Hinweis (z.B. 'Phasenwechsel – CP numerisch')
    """
    name:        str
    stream_type: str     # 'hot' | 'cold'
    T_supply:    float   # °C
    T_target:    float   # °C
    CP:          float   # kW/K
    Q:           float   # kW
    note:        str = ''

    def __post_init__(self):
        # Konsistenzcheck
        if self.stream_type == 'hot' and self.T_supply < self.T_target:
            raise ValueError(f"Hot stream '{self.name}': T_supply muss > T_target sein.")
        if self.stream_type == 'cold' and self.T_supply > self.T_target:
            raise ValueError(f"Cold stream '{self.name}': T_supply muss < T_target sein.")

    def __repr__(self):
        direction = '→' if self.stream_type == 'hot' else '←'
        return (f"PinchStream({self.stream_type.upper():<4}  "
                f"{self.name:<35}  "
                f"{self.T_supply:.1f}°C {direction} {self.T_target:.1f}°C  "
                f"CP={self.CP:.4f} kW/K  Q={self.Q:.1f} kW)")


# ══════════════════════════════════════════════════════════════════════════════
# Hilfsfunktion
# ══════════════════════════════════════════════════════════════════════════════

def _val(params: dict, key: str) -> Any:
    """Holt Wert sicher aus QuestionnaireReader-params-Dict."""
    entry = params.get(key)
    if entry is None:
        return None
    return entry.get('value') if isinstance(entry, dict) else entry


# ══════════════════════════════════════════════════════════════════════════════
# Haupt-Klasse
# ══════════════════════════════════════════════════════════════════════════════

class PinchParamBuilder:
    """
    Baut aus QuestionnaireReader-Parametern alle Streams für die Pinch-Analyse.

    Parameters
    ----------
    params : dict
        Ausgabe von QuestionnaireReader.get_params()
    delta_T_min : float
        Minimale Temperaturdifferenz ΔTmin [K]  (Standard: 10 K)
        Beeinflusst keine Stream-Berechnung, wird aber als Metadatum
        an den Pinch-Rechner weitergegeben.

    Ausgabe von get_pinch_params():
        {
            'delta_T_min' : float,
            'streams'     : list[PinchStream],
            'warnings'    : list[str],
        }

    Auf Streams zugreifen:
        result = builder.get_pinch_params()
        for s in result['streams']:
            print(s.name, s.CP, s.T_supply, s.T_target)
    """

    # Numerischer Trick für isotherme Segmente (Verdampfung):
    # PyPinch und eigene Solver brauchen immer ein ΔT > 0.
    # CP = Q / DT_ISOTHERMAL → Q bleibt exakt erhalten.
    DT_ISOTHERMAL = 0.01   # K

    def __init__(self, params: dict, delta_T_min: float = 10.0):
        self.params      = params
        self.delta_T_min = delta_T_min
        self._streams:  list[PinchStream] = []
        self._warnings: list[str]         = []

    # ── Öffentliche Schnittstelle ─────────────────────────────────────────────

    def get_pinch_params(self) -> dict:
        """
        Berechnet und gibt alle Pinch-Parameter zurück.

        Returns
        -------
        dict mit:
            'delta_T_min' : float          – ΔTmin für den Pinch-Rechner
            'streams'     : list[PinchStream]
            'warnings'    : list[str]      – Hinweise auf fehlende/geschätzte Werte
        """
        self._streams  = []
        self._warnings = []

        self._build_source_stream()
        self._build_sink_streams()
        self._build_waste_heat_streams()

        self._validate()

        return {
            'delta_T_min': self.delta_T_min,
            'streams':     self._streams,
            'warnings':    self._warnings,
        }

    def print_summary(self):
        """Gibt alle Streams tabellarisch aus."""
        result = self.get_pinch_params()
        streams = result['streams']

        print("\n" + "=" * 85)
        print("PINCH PARAMETER SUMMARY")
        print(f"ΔTmin = {self.delta_T_min} K")
        print("=" * 85)
        print(f"  {'#':<3} {'Typ':<5} {'Name':<35} {'T_s [°C]':>9} "
              f"{'T_t [°C]':>9} {'CP [kW/K]':>11} {'Q [kW]':>9}  Note")
        print("  " + "-" * 83)

        hot_streams  = [s for s in streams if s.stream_type == 'hot']
        cold_streams = [s for s in streams if s.stream_type == 'cold']

        for group_label, group in [('HOT STREAMS', hot_streams),
                                    ('COLD STREAMS', cold_streams)]:
            print(f"\n  ── {group_label}")
            for i, s in enumerate(group, 1):
                note = f'  [{s.note}]' if s.note else ''
                print(f"  {i:<3} {s.stream_type.upper():<5} {s.name:<35} "
                      f"{s.T_supply:>9.2f} {s.T_target:>9.2f} "
                      f"{s.CP:>11.4f} {s.Q:>9.1f}{note}")

        if result['warnings']:
            print("\n  ── WARNUNGEN")
            for w in result['warnings']:
                print(f"  ⚠  {w}")

        print("=" * 85 + "\n")

    # ── Stream-Builder ────────────────────────────────────────────────────────

    def _build_source_stream(self):
        """
        Hot Stream: Wärmequelle (Abwärme, Kühlwasser, etc.)

        CP wird aus Leistung und Temperaturspreizung berechnet:
            CP = Q_source / (T_in - T_out)
        Falls Q_source nicht angegeben: Fallback CP über Wassermassenstrom-Schätzung.
        """
        T_in  = _val(self.params, 'source_temp_in')
        T_out = _val(self.params, 'source_temp_out')
        Q     = _val(self.params, 'source_heat_capacity')

        if T_in is None or T_out is None:
            self._warnings.append("Wärmequelle: T_in oder T_out fehlt – kein Hot Stream erzeugt.")
            return

        T_in, T_out = float(T_in), float(T_out)
        dT = T_in - T_out

        if dT <= 0:
            self._warnings.append(
                f"Wärmequelle: T_in ({T_in}°C) ≤ T_out ({T_out}°C) – übersprungen."
            )
            return

        if Q:
            CP = float(Q) / dT
            note = ''
        else:
            # Fallback: 5 kg/s Wasser (typisch für industrielle Quellen)
            CP = 5.0 * 4.18
            note = 'CP geschätzt (kein Q angegeben)'
            self._warnings.append("Wärmequelle: Keine Leistungsangabe → CP = 5 kg/s × 4,18 kJ/kgK")

        self._streams.append(PinchStream(
            name='Heat Source',
            stream_type='hot',
            T_supply=T_in,
            T_target=T_out,
            CP=CP,
            Q=CP * dT,
            note=note,
        ))

    def _build_sink_streams(self):
        """Senke je nach Anwendungsfall (Heißwasser oder Dampf)."""
        app_type = str(_val(self.params, 'application_type') or '').lower()

        if 'steam' in app_type:
            self._build_steam_streams()
        else:
            self._build_hot_water_streams()

    def _build_hot_water_streams(self):
        """
        Cold Streams: Heißwassererzeuger (Fall A)

        Segment: T_feed → T_outlet  (sensibel, Wasser)
            CP = Q / ΔT  oder  m_dot * cp_water
        """
        T_in  = _val(self.params, 'hw_temp_inlet')
        T_out = (_val(self.params, 'hw_temp_outlet_required')
                 or _val(self.params, 'hw_temp_outlet_min'))
        Q     = _val(self.params, 'hw_heat_capacity')
        m     = _val(self.params, 'hw_mass_flow')

        if T_in is None or T_out is None:
            self._warnings.append("Heißwasser: T_in oder T_out fehlt – kein Cold Stream.")
            return

        T_in, T_out = float(T_in), float(T_out)
        dT = T_out - T_in
        if dT <= 0:
            self._warnings.append(f"Heißwasser: T_out ({T_out}°C) ≤ T_in ({T_in}°C) – übersprungen.")
            return

        if Q:
            CP = float(Q) / dT
            note = ''
        elif m:
            CP = float(m) * 4.18
            note = 'CP aus Massenstrom × cp_Wasser'
        else:
            CP = 2.0 * 4.18
            note = 'CP geschätzt (2 kg/s × 4,18 kJ/kgK)'
            self._warnings.append("Heißwasser: Keine Leistungs-/Massenstromangabe → CP geschätzt")

        self._streams.append(PinchStream(
            name='Hot Water Sink',
            stream_type='cold',
            T_supply=T_in,
            T_target=T_out,
            CP=CP,
            Q=CP * dT,
            note=note,
        ))

        # Optionaler zweiter Verbraucher
        T_a_in  = _val(self.params, 'add_hw_temp_inlet')
        T_a_out = _val(self.params, 'add_hw_temp_outlet_required')
        Q_a     = _val(self.params, 'add_hw_heat_capacity')

        if T_a_in and T_a_out:
            T_a_in, T_a_out = float(T_a_in), float(T_a_out)
            dT2 = T_a_out - T_a_in
            if dT2 > 0:
                CP2 = float(Q_a) / dT2 if Q_a else 2.0 * 4.18
                self._streams.append(PinchStream(
                    name='Additional Hot Water Sink',
                    stream_type='cold',
                    T_supply=T_a_in,
                    T_target=T_a_out,
                    CP=CP2,
                    Q=CP2 * dT2,
                ))

    def _build_steam_streams(self):
        """
        Cold Streams: Dampferzeuger (Fall B)

        Drei Segmente (sofern thermodynamisch relevant):

        1. Vorwärmung (sensibel):
               T_feed → T_sat
               CP = m_dot · cp_water(T_mean)

        2. Verdampfung (isotherm, Phasenwechsel):
               T_sat → T_sat + DT_ISOTHERMAL
               CP = Q_evap / DT_ISOTHERMAL
               (numerischer Trick: ΔT ≠ 0, Q bleibt exakt)

        3. Überhitzung (sensibel, falls angegeben):
               T_sat + DT_ISOTHERMAL → T_sat + superheat_K
               CP = m_dot · cp_steam(T_mean)
        """
        if not _COOLPROP:
            self._warnings.append(
                "CoolProp nicht verfügbar – Dampfsegmente können nicht berechnet werden."
            )
            return

        T_feed    = _val(self.params, 'steam_temp_inlet')
        p_feed    = _val(self.params, 'steam_pressure_inlet')
        p_steam   = _val(self.params, 'steam_pressure_outlet')
        m_flow    = _val(self.params, 'steam_mass_flow_inlet')
        Q_src     = _val(self.params, 'source_heat_capacity')  # Fallback für Leistung
        superheat = float(_val(self.params, 'steam_superheat') or 0.0)

        if T_feed is None or p_steam is None:
            self._warnings.append("Dampf: T_feed oder p_steam fehlt – keine Dampfsegmente.")
            return

        T_feed  = float(T_feed)
        p_steam = float(p_steam)
        p_feed  = float(p_feed) if p_feed else p_steam

        # Sättigungstemperatur bei Ausgangsdruck
        p_Pa  = p_steam * 1e5
        T_sat = CP.PropsSI('T', 'P', p_Pa, 'Q', 0, 'Water') - 273.15  # °C

        # Enthalpien
        h_feed  = CP.PropsSI('H', 'T', T_feed + 273.15, 'P', p_feed * 1e5, 'Water')
        h_sat_l = CP.PropsSI('H', 'P', p_Pa, 'Q', 0, 'Water')
        h_sat_v = CP.PropsSI('H', 'P', p_Pa, 'Q', 1, 'Water')
        T_out   = T_sat + superheat

        # Sättigungsdampf (superheat = 0): Zustand liegt exakt auf der Siedelinie.
        # PropsSI('H', 'T', T_sat, 'P', p_sat) ist dort nicht eindeutig (x=0..1).
        # → stattdessen Q=1 (gesättigter Dampf) verwenden.
        if superheat < 0.1:
            h_out = h_sat_v
        else:
            h_out = CP.PropsSI('H', 'T', T_out + 273.15, 'P', p_Pa, 'Water')

        h_total = h_out - h_feed   # J/kg

        # Massenstrom
        if m_flow and float(m_flow) > 0:
            m_kg_s = float(m_flow) / 3600.0
            note_m = ''
        elif Q_src and float(Q_src) > 0:
            m_kg_s = (float(Q_src) * 1000.0) / h_total
            note_m = 'Massenstrom aus Quellleistung geschätzt'
            self._warnings.append(
                f"Dampf: Kein Massenstrom angegeben → m_dot aus Q_source = {Q_src} kW berechnet"
            )
        else:
            self._warnings.append("Dampf: Weder Massenstrom noch Leistung angegeben – übersprungen.")
            return

        # ── Segment 1: Vorwärmung ──────────────────────────────────────────
        Q_preheat = m_kg_s * (h_sat_l - h_feed) / 1000.0   # kW
        if Q_preheat > 0.1 and T_sat > T_feed:
            T_mean_pre = (T_feed + T_sat) / 2
            cp_water   = CP.PropsSI('C', 'T', T_mean_pre + 273.15,
                                    'P', p_feed * 1e5, 'Water') / 1000.0   # kJ/kgK
            CP_pre = m_kg_s * cp_water
            self._streams.append(PinchStream(
                name='Steam: Feed Water Preheating',
                stream_type='cold',
                T_supply=T_feed,
                T_target=T_sat,
                CP=CP_pre,
                Q=Q_preheat,
                note=note_m,
            ))

        # ── Segment 2: Verdampfung ─────────────────────────────────────────
        Q_evap = m_kg_s * (h_sat_v - h_sat_l) / 1000.0   # kW
        if Q_evap > 0.1:
            CP_evap = Q_evap / self.DT_ISOTHERMAL
            self._streams.append(PinchStream(
                name='Steam: Evaporation',
                stream_type='cold',
                T_supply=T_sat,
                T_target=T_sat + self.DT_ISOTHERMAL,
                CP=CP_evap,
                Q=Q_evap,
                note=f'Phasenwechsel – ΔT={self.DT_ISOTHERMAL} K (numerisch)',
            ))

        # ── Segment 3: Überhitzung ─────────────────────────────────────────
        Q_super = m_kg_s * (h_out - h_sat_v) / 1000.0   # kW
        if superheat > 0.1 and Q_super > 0.1:
            T_mean_sup = (T_sat + T_out) / 2
            cp_steam   = CP.PropsSI('C', 'T', T_mean_sup + 273.15,
                                    'P', p_Pa, 'Water') / 1000.0
            CP_sup = m_kg_s * cp_steam
            self._streams.append(PinchStream(
                name='Steam: Superheating',
                stream_type='cold',
                T_supply=T_sat + self.DT_ISOTHERMAL,
                T_target=T_out,
                CP=CP_sup,
                Q=Q_super,
            ))

    def _build_waste_heat_streams(self):
        """
        Hot Streams: Zusätzliche Abwärmequellen (Waste Heat 1–3)

        CP = m_dot [kg/h] / 3600 × 4,18  (Wasser-Annahme)
        Falls kein Massenstrom: CP = 10 kW/K (Hinweis)
        """
        n = int(_val(self.params, 'waste_heat_count') or 0)
        for i in range(1, n + 1):
            T_in  = _val(self.params, f'waste_heat_{i}_temp_supply')
            T_out = _val(self.params, f'waste_heat_{i}_temp_outlet')
            m     = _val(self.params, f'waste_heat_{i}_mass_flow')

            if T_in is None or T_out is None:
                continue

            T_in, T_out = float(T_in), float(T_out)
            dT = T_in - T_out   # Hot stream: T_in > T_out

            if abs(dT) < 0.1:
                self._warnings.append(f"Waste Heat {i}: ΔT < 0,1 K – übersprungen.")
                continue

            # Sicherstellen, dass Hot-Stream korrekt orientiert ist
            if dT < 0:
                T_in, T_out = T_out, T_in
                dT = -dT
                self._warnings.append(
                    f"Waste Heat {i}: T_supply < T_target → Richtung umgekehrt."
                )

            if m:
                CP   = float(m) / 3600.0 * 4.18
                note = ''
            else:
                CP   = 10.0
                note = 'CP = 10 kW/K (Schätzwert – kein Massenstrom angegeben)'
                self._warnings.append(f"Waste Heat {i}: Kein Massenstrom → CP geschätzt")

            self._streams.append(PinchStream(
                name=f'Waste Heat {i}',
                stream_type='hot',
                T_supply=T_in,
                T_target=T_out,
                CP=CP,
                Q=CP * dT,
                note=note,
            ))

    # ── Validierung ───────────────────────────────────────────────────────────

    def _validate(self):
        """Prüft Mindestanforderungen für die Pinch-Analyse."""
        hot  = [s for s in self._streams if s.stream_type == 'hot']
        cold = [s for s in self._streams if s.stream_type == 'cold']

        if not hot:
            self._warnings.append("FEHLER: Keine Hot Streams – Pinch-Analyse nicht möglich.")
        if not cold:
            self._warnings.append("FEHLER: Keine Cold Streams – Pinch-Analyse nicht möglich.")
        if len(self._streams) < 2:
            self._warnings.append("FEHLER: Mindestens 2 Streams erforderlich.")


# ══════════════════════════════════════════════════════════════════════════════
# Standalone Test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    path = sys.argv[1] if len(sys.argv) > 1 else 'HTHP_questionnaire.xlsx'

    try:
        from questionnaire_reader import QuestionnaireReader
        reader = QuestionnaireReader(path)
        params = reader.get_params()
        print(f"Questionnaire geladen: {path}")
    except Exception as e:
        print(f"Questionnaire nicht verfügbar ({e}) – Demo-Parameter")
        def _p(v, u=None): return {'value': v, 'unit': u}
        params = {
            'source_temp_in':        _p(90.0,  '°C'),
            'source_temp_out':       _p(70.0,  '°C'),
            'source_heat_capacity':  _p(500.0, 'kW'),
            'application_type':      _p('Steam generation'),
            'steam_temp_inlet':      _p(20.0,  '°C'),
            'steam_pressure_inlet':  _p(1.013, 'bar'),
            'steam_pressure_outlet': _p(2.1,   'bar'),
            'steam_mass_flow_inlet': _p(810.0, 'kg/h'),
            'steam_superheat':       _p(19.0,  'K'),
            'waste_heat_count':      _p(0),
        }

    builder = PinchParamBuilder(params, delta_T_min=10.0)
    builder.print_summary()

    result = builder.get_pinch_params()
    print("Direkter Zugriff auf Streams:")
    for s in result['streams']:
        print(f"  {s}")