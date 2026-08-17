"""
MVRMultiStage - Mehrstufiger MVR mit Wassereinspritzung
Offenes System für Dampfverdichtung ohne Wärmerückgewinnung

Aufbau:
  Eingang → Stufe 1 → Wassereinspritzung 1 → Stufe 2 → ... → Stufe n → Ausgang

Features:
- Flexible Anzahl von Verdichterstufen (2-5 empfohlen)
- Wassereinspritzung zwischen Stufen zur Zwischenkühlung
- Automatische optimale Druckverteilung
- Compressor für jede Stufe
- SEI-Berechnung (Specific Energy Input) statt COP

Änderungen gegenüber alter Version:
- Intercooler entfernt
- Direkte Wassereinspritzung implementiert
- Massenstrom erhöht sich durch Wassereinspritzung
- SEI (kWh/kg Dampf) als Hauptkennzahl
"""

import numpy as np
from tespy.components import Compressor, Merge, Sink, Source
from tespy.connections import Bus, Connection, Ref
from tespy.tools.characteristics import CharLine

import sys
import os

# Füge aktuelles Verzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from MVRBase import MVRBase
except ImportError:
    from .MVRBase import MVRBase


class MVRMultiStage(MVRBase):
    """Mehrstufiger MVR-Verdichter mit Wassereinspritzung (offenes System)."""

    def __init__(self, params):
        """Initialisiert mehrstufigen MVR."""
        # Anzahl der Stufen aus Parametern
        self.n_stages = params.get('n_stages', 2)
        
        if self.n_stages < 1:
            raise ValueError("MVR benötigt mindestens 1 Stufe!")
        if self.n_stages > 5:
            print(f"⚠ Warning: {self.n_stages} stages is unusual. Recommended: 1-4 Stufen.")
        
        # Basis-Initialisierung
        super().__init__(params)
        
        # Zusätzliche Attribute für mehrstufigen Betrieb
        self.stage_results = {}
        self.total_power = np.nan
        self.stage_pressures = []
        self.stage_temperatures = []
        self.water_injection_rates = []  # kg/s pro Stufe
        self.total_water_injected = 0.0  # Gesamt eingespritzte Wassermenge
        
        # Massenströme
        self.m_target = np.nan       # Ziel-Ausgangsmassenstrom [kg/s] (Vorgabe)
        self.Q_preheat_kW = 0.0      # Vorwärmleistung Speisewasser → Sattdampf [kW]

        # SEI statt COP (offenes System!)
        self.SEI = np.nan           # kWh/kg Dampf
        self.SEI_MJ_per_kg = np.nan # MJ/kg Dampf
        self.COP = np.nan           # Q_nutz / W_el  [-]
        self.Q_heat = np.nan        # Nutzwärme an Prozess [kW]

    def generate_components(self):
        """Initialisiert Komponenten des mehrstufigen MVR-Systems."""
        # Eingang (Dampf)
        self.comps['inlet'] = Source('MVR Inlet')
        
        # Verdichterstufen und Wassereinspritzung
        for stage in range(1, self.n_stages + 1):
            # Verdichter für diese Stufe
            self.comps[f'comp_{stage}'] = Compressor(f'Compressor Stage {stage}')
            
            # Wassereinspritzung nach dieser Stufe (außer nach letzter Stufe)
            if stage < self.n_stages:
                # Merge-Komponente: Dampf + Wasser zusammenführen
                self.comps[f'merge_{stage}'] = Merge(f'Water Injection {stage}')
                
                # Wasserquelle für Einspritzung
                self.comps[f'water_source_{stage}'] = Source(f'Cooling Water Source {stage}')
        
        # Ausgang
        self.comps['outlet'] = Sink('MVR Outlet')

    def generate_connections(self):
        """Initialisiert und fügt Verbindungen und Busse zum Netzwerk hinzu."""
        # Zähler für Verbindungen
        conn_idx = 0
        
        # === HAUPTSTRANG (Prozessgas/Dampf) ===
        
        # Eingang zu erstem Verdichter
        self.conns[f'{conn_idx}'] = Connection(
            self.comps['inlet'], 'out1', 
            self.comps['comp_1'], 'in1', 
            f'{conn_idx}'
        )
        conn_idx += 1
        
        # Durch alle Stufen
        for stage in range(1, self.n_stages + 1):
            if stage < self.n_stages:
                # Verdichter zu Merge (Wassereinspritzung)
                self.conns[f'{conn_idx}'] = Connection(
                    self.comps[f'comp_{stage}'], 'out1',
                    self.comps[f'merge_{stage}'], 'in1',
                    f'{conn_idx}'
                )
                conn_idx += 1
                
                # Wasser-Einspritzung zu Merge
                self.conns[f'water_{stage}_in'] = Connection(
                    self.comps[f'water_source_{stage}'], 'out1',
                    self.comps[f'merge_{stage}'], 'in2',
                    f'water_{stage}_in'
                )
                
                # Merge zu nächstem Verdichter
                self.conns[f'{conn_idx}'] = Connection(
                    self.comps[f'merge_{stage}'], 'out1',
                    self.comps[f'comp_{stage + 1}'], 'in1',
                    f'{conn_idx}'
                )
                conn_idx += 1
            else:
                # Letzter Verdichter zum Ausgang
                self.conns[f'{conn_idx}'] = Connection(
                    self.comps[f'comp_{stage}'], 'out1',
                    self.comps['outlet'], 'in1',
                    f'{conn_idx}'
                )
                conn_idx += 1
        
        # Alle Verbindungen zum Netzwerk hinzufügen
        self.nw.add_conns(*[conn for conn in self.conns.values()])
        
        # === POWER BUS ===
        
        # Motor-Kennlinie
        mot_x = np.array([
            0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
            0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15,
            1.2, 10
        ])
        mot_y = (np.array([
            0.01, 0.3148, 0.5346, 0.6843, 0.7835, 0.8477, 0.8885, 0.9145,
            0.9318, 0.9443, 0.9546, 0.9638, 0.9724, 0.9806, 0.9878, 0.9938,
            0.9982, 1.0009, 1.002, 1.0015, 1, 0.9977, 0.9947, 0.9909, 0.9853,
            0.9644
        ]) * 0.98)
        mot = CharLine(x=mot_x, y=mot_y)
        
        self.buses['power input'] = Bus('power input')
        
        # Alle Verdichter zum Power Bus hinzufügen
        for stage in range(1, self.n_stages + 1):
            self.buses['power input'].add_comps(
                {'comp': self.comps[f'comp_{stage}'], 'char': mot, 'base': 'bus'}
            )
        
        self.nw.add_busses(*[bus for bus in self.buses.values()])

    def calculate_stage_pressures(self, p_in, p_out):
        """
        Berechnet optimale Druckverteilung über alle Stufen.
        Verwendet geometrische Verteilung: PR_stage = PR_total^(1/n)
        
        Args:
            p_in: Eintrittsdruck [bar]
            p_out: Austrittsdruck [bar]
        
        Returns:
            list: Drücke nach jeder Stufe [bar]
        """
        PR_total = p_out / p_in
        PR_stage = PR_total ** (1 / self.n_stages)
        
        pressures = [p_in]
        for stage in range(1, self.n_stages + 1):
            pressures.append(pressures[-1] * PR_stage)
        
        return pressures

    def calculate_water_injection_rate(self, stage, T_steam_in, p, T_target, m_steam):
        """
        Berechnet benötigte Wassereinspritzrate für gewünschte Zieltemperatur.
        
        Energiebilanz:
        m_steam * h_steam + m_water * h_water = (m_steam + m_water) * h_mix
        
        Args:
            stage: Stufennummer
            T_steam_in: Dampftemperatur vor Einspritzung [°C]
            p: Druck [bar]
            T_target: Zieltemperatur nach Einspritzung [°C]
            m_steam: Dampfmassenstrom [kg/s]
        
        Returns:
            m_water: Benötigte Wassereinspritzrate [kg/s]
        """
        from CoolProp.CoolProp import PropsSI as PSI
        
        # Enthalpien berechnen
        p_Pa = p * 1e5
        
        # Dampf vor Einspritzung (überhitzter Dampf)
        h_steam = PSI('H', 'T', T_steam_in + 273.15, 'P', p_Pa, 'Water') / 1000  # kJ/kg
        
        # Einspritzwasser (Flüssigkeit, kaltes Leitungswasser)
        T_water = self.params['cooling_water']['T_in']
        h_water = PSI('H', 'T', T_water + 273.15, 'P', p_Pa, 'Water') / 1000  # kJ/kg
        
        # Zielzustand: gesättigter Dampf bei Zieltemperatur + Überhitzung
        superheat = self.params['intercoolers'].get('superheat', 0)  # K
        T_mix = T_target + superheat
        h_mix = PSI('H', 'T', T_mix + 273.15, 'P', p_Pa, 'Water') / 1000  # kJ/kg
        
        # Energiebilanz auflösen nach m_water
        # m_steam * h_steam + m_water * h_water = (m_steam + m_water) * h_mix
        # m_water * (h_water - h_mix) = m_steam * (h_mix - h_steam)
        # m_water = m_steam * (h_mix - h_steam) / (h_water - h_mix)
        
        if abs(h_water - h_mix) < 1e-6:
            # Vermeidung Division durch Null
            m_water = 0.0
        else:
            m_water = m_steam * (h_mix - h_steam) / (h_water - h_mix)
        
        # Sicherheitscheck
        if m_water < 0:
            print(f"⚠ Warning Stufe {stage}: Berechnete negative Wasserrate! Setze auf 0.")
            m_water = 0.0
        
        return max(0.0, m_water)

    def init_simulation(self, **kwargs):
        """
        Führt initiale Parametrierung durch.

        Strategie: Rück-Iteration des Massenstroms vom Ziel-Ausgangsmassenstrom.

        Vorgabe: m_target [kg/s] am Ausgang nach letztem Verdichter.
        Rückrechnung: m_inlet (Eintritt Stufe 1) und m_inj_k je Einspritzpunkt.

        Energiebilanz am Einspritzpunkt zwischen Stufe k und k+1:
            m_k * h_out_k + m_inj_k * h_fw = m_{k+1} * h_in_{k+1}
        mit  m_{k+1} = m_k + m_inj_k
        →    m_k = m_{k+1} * (h_in_{k+1} - h_fw) / (h_out_k - h_fw)
        →  m_inj_k = m_{k+1} - m_k

        Alle spezifischen Enthalpien sind a priori bekannt (CoolProp), da der
        Eintrittszustand jeder Stufe durch T_sat(p_k) + ΔT_sh vollständig definiert ist.
        """
        from CoolProp.CoolProp import PropsSI as PSI

        MIN_SUPERHEAT = 1.0  # K – verhindert CoolProp-Fehler auf der Siedelinie

        # ── Grundparameter ──────────────────────────────────────────────────
        p_in = self.params['inlet']['p']
        if 'p_out' in self.params['outlet']:
            p_out = self.params['outlet']['p_out']
        elif 'PR_total' in self.params['outlet']:
            p_out = p_in * self.params['outlet']['PR_total']
        else:
            raise ValueError("Entweder 'p_out' oder 'PR_total' in params['outlet'] erforderlich!")

        m_target = self.params['outlet']['m']          # Ziel-Ausgangsmassenstrom [kg/s]
        T_fw     = self.params['cooling_water']['T_in'] # Speisewassertemperatur [°C]
        sh_inter = max(
            MIN_SUPERHEAT,
            self.params['intercoolers'].get('superheat', MIN_SUPERHEAT)
        )

        # ── Druckverteilung ─────────────────────────────────────────────────
        self.stage_pressures = self.calculate_stage_pressures(p_in, p_out)

        print(f"\nDruckverteilung über {self.n_stages} Stufen:")
        print(f"  Eintritt: {self.stage_pressures[0]:.3f} bar")
        for k in range(self.n_stages):
            PR_k = self.stage_pressures[k + 1] / self.stage_pressures[k]
            print(f"  Nach Stufe {k + 1}: {self.stage_pressures[k + 1]:.3f} bar  (PR = {PR_k:.3f})")

        # ── Isentrope Wirkungsgrade ──────────────────────────────────────────
        if 'eta_s' in self.params['compressors']:
            eta_s_list = [self.params['compressors']['eta_s']] * self.n_stages
        else:
            eta_s_list = [
                self.params['compressors'].get(f'eta_s_{i}', 0.85)
                for i in range(1, self.n_stages + 1)
            ]

        # ── Spezifische Zustandsgrößen je Stufe (CoolProp, 0-indiziert) ────
        # Eintritt jeder Stufe: T_sat(p_in_k) + sh_inter → vollständig definiert
        h_in_stages  = []   # kJ/kg
        h_out_stages = []   # kJ/kg (reale Verdichteraustrittsenthalpie)
        T_in_stages  = []   # °C

        for k in range(self.n_stages):
            p_k_in  = self.stage_pressures[k]
            p_k_out = self.stage_pressures[k + 1]

            T_sat_k = PSI('T', 'P', p_k_in * 1e5, 'Q', 1, 'Water') - 273.15
            T_in_k  = T_sat_k + sh_inter

            h_in_k  = PSI('H', 'T', T_in_k + 273.15, 'P', p_k_in  * 1e5, 'Water') / 1000
            s_in_k  = PSI('S', 'T', T_in_k + 273.15, 'P', p_k_in  * 1e5, 'Water') / 1000
            h_is_k  = PSI('H', 'P', p_k_out * 1e5, 'S', s_in_k * 1000, 'Water') / 1000
            h_out_k = h_in_k + (h_is_k - h_in_k) / eta_s_list[k]

            h_in_stages.append(h_in_k)
            h_out_stages.append(h_out_k)
            T_in_stages.append(T_in_k)

        # ── Rück-Iteration Massenstrom ───────────────────────────────────────
        # m_stages[k] = Massenstrom durch Verdichter k+1 (0-indiziert)
        m_stages = [None] * self.n_stages
        m_stages[-1] = m_target          # letzter Verdichter fördert Ziel-Massenstrom

        self.water_injection_rates = []  # wird rückwärts befüllt, dann umgekehrt

        inj_rates_rev = []
        for k in range(self.n_stages - 2, -1, -1):
            # Einspritzpunkt nach Verdichter k (0-indiziert) = vor Verdichter k+1
            p_inj_k  = self.stage_pressures[k + 1]

            # Speisewasser-Enthalpie bei Einspritzdruck (Flüssigkeit, nahezu druckunabhängig)
            h_fw_k = PSI('H', 'T', T_fw + 273.15, 'P', p_inj_k * 1e5, 'Water') / 1000

            h_out_k  = h_out_stages[k]       # Austrittsenth. Verdichter k
            h_in_kp1 = h_in_stages[k + 1]   # Eintrittsenth. Verdichter k+1 (nach Einspritzung)

            denom = h_out_k - h_fw_k
            if abs(denom) < 1.0:
                raise ValueError(
                    f"Stufe {k + 1}: (h_out - h_fw) = {denom:.2f} kJ/kg zu klein. "
                    f"Speisewasser zu heiß oder Verdichteraustritt zu kalt?"
                )

            m_k     = m_stages[k + 1] * (h_in_kp1 - h_fw_k) / denom
            m_inj_k = m_stages[k + 1] - m_k

            if m_inj_k < 0:
                raise ValueError(
                    f"Stufe {k + 1}: Negative Einspritzrate ({m_inj_k:.5f} kg/s). "
                    f"h_out={h_out_k:.1f}, h_in_next={h_in_kp1:.1f}, h_fw={h_fw_k:.1f} kJ/kg."
                )

            m_stages[k] = m_k
            inj_rates_rev.append(m_inj_k)

        # Einspritzraten in richtiger Reihenfolge (Stufe 1→2, 2→3, …)
        self.water_injection_rates = list(reversed(inj_rates_rev))
        self.total_water_injected  = sum(self.water_injection_rates)

        m_inlet = m_stages[0]

        print(f"\nRück-iterierter Eintrittsmassenstrom:  {m_inlet:.5f} kg/s")
        print(f"Ziel-Ausgangsmassenstrom:              {m_target:.5f} kg/s")
        for k in range(self.n_stages - 1):
            p_inj = self.stage_pressures[k + 1]
            T_sat_inj = PSI('T', 'P', p_inj * 1e5, 'Q', 1, 'Water') - 273.15
            print(
                f"  Einspritzung nach Stufe {k + 1}: "
                f"{self.water_injection_rates[k]:.5f} kg/s  "
                f"(p={p_inj:.3f} bar, T_sat={T_sat_inj:.1f} °C)"
            )

        # ── Vorwärmleistung (Speisewasser unterkühlt bei p_in) ───────────────
        T_sat_in = PSI('T', 'P', p_in * 1e5, 'Q', 1, 'Water') - 273.15
        if T_fw < T_sat_in - 0.5:
            h_fw_at_pin = PSI('H', 'T', T_fw + 273.15, 'P', p_in * 1e5, 'Water') / 1000
            h_sat_steam = PSI('H', 'P', p_in * 1e5, 'Q', 1, 'Water') / 1000
            self.Q_preheat_kW = m_inlet * (h_sat_steam - h_fw_at_pin)
            print(f"  Vorwärmleistung (Speisewasser → Sattdampf bei p_in): "
                  f"{self.Q_preheat_kW:.1f} kW")
        else:
            self.Q_preheat_kW = 0.0

        # ── TESPy Parametrierung ────────────────────────────────────────────

        # Eintrittsverbindung mit zurückgerechnetem Massenstrom
        self.conns['0'].set_attr(
            T=T_in_stages[0],
            p=p_in,
            m=m_inlet,
            fluid={self.wf: 1}
        )

        # Verdichter (Druckverhältnis + isentroper Wirkungsgrad)
        for stage in range(1, self.n_stages + 1):
            self.comps[f'comp_{stage}'].set_attr(
                eta_s=eta_s_list[stage - 1],
                pr=self.stage_pressures[stage] / self.stage_pressures[stage - 1]
            )

        # Einspritzverbindungen (Druck wird von TESPy aus Mischungsbilanz ermittelt)
        for stage in range(1, self.n_stages):
            self.conns[f'water_{stage}_in'].set_attr(
                T=T_fw,
                m=self.water_injection_rates[stage - 1],
                fluid={self.wf: 1}
            )

    def design_simulation(self, **kwargs):
        """Führt finale Parametrierung und Design-Simulation durch."""
        self._solve_model(**kwargs)
        
        # Inlet-Massenstrom (zurückgerechnet) und Ziel-Ausgangsmassenstrom
        self.m_design = self.conns['0'].m.val
        self.m_target = self.comps[f'comp_{self.n_stages}'].outl[0].m.val
        
        # Gesamtleistung berechnen
        self.total_power = abs(self.buses['power input'].P.val) / 1000.0  # W → kW

    def calc_performance(self):
        """Berechnet Performance-Kennzahlen des mehrstufigen MVR-Systems."""
        # Basis-Performance (von MVRBase)
        # Beachte: COP ist hier nicht sinnvoll (offenes System!)
        
        # Stufen-spezifische Ergebnisse
        self.stage_results = {}
        self.stage_temperatures = []
        
        for stage in range(1, self.n_stages + 1):
            comp = self.comps[f'comp_{stage}']
            
            stage_data = {
                'p_in': comp.inl[0].p.val,
                'p_out': comp.outl[0].p.val,
                'T_in': comp.inl[0].T.val,
                'T_out': comp.outl[0].T.val,
                'h_in': comp.inl[0].h.val,
                'h_out': comp.outl[0].h.val,
                'PR': comp.outl[0].p.val / comp.inl[0].p.val,
                'eta_s': comp.eta_s.val,
                'P': comp.P.val / 1000.0,  # W → kW
                'specific_work': comp.outl[0].h.val - comp.inl[0].h.val,
                'm': comp.inl[0].m.val
            }
            
            self.stage_results[f'stage_{stage}'] = stage_data
            self.stage_temperatures.append(comp.outl[0].T.val)
        
        # Gesamtwirkungsgrad (isentrop, über alle Stufen)
        p_in = self.comps['comp_1'].inl[0].p.val
        p_out = self.comps[f'comp_{self.n_stages}'].outl[0].p.val
        self.compression_ratio = p_out / p_in
        
        # Gesamtleistung
        self.total_power = abs(self.buses['power input'].P.val) / 1000.0  # W → kW
        
        # SEI berechnen (Specific Energy Input)
        # Bezugsgröße: Ausgangs-Dampfmassenstrom (= Ziel-Produktion)
        m_steam_out = self.comps[f'comp_{self.n_stages}'].outl[0].m.val
        
        if m_steam_out > 0:
            self.SEI = self.total_power / m_steam_out / 3600  # kWh/kg
            self.SEI_MJ_per_kg = self.SEI * 3.6  # kWh/kg → MJ/kg
        else:
            self.SEI = np.nan
            self.SEI_MJ_per_kg = np.nan
        
        # Spezifische Arbeit (gesamt)
        h_in = self.comps['comp_1'].inl[0].h.val
        h_out = self.comps[f'comp_{self.n_stages}'].outl[0].h.val
        self.specific_work = h_out - h_in
        
        # Volumenstrom am Eintritt
        self.volumetric_flow_in = self.comps['comp_1'].inl[0].vol.val_SI * 3600

        # COP berechnen (analog zu HeatPump: Q_nutz / W_el)
        # Q_nutz = m_out_final * (h_out_final - h_liq_sat(p_in)) [kW]
        # Referenz: gesättigte Flüssigkeit am MVR-Eintrittsdruck
        from CoolProp.CoolProp import PropsSI as _PSI_cop
        _p_in_val  = self.comps['comp_1'].inl[0].p.val               # bar
        _h_liq_ref = _PSI_cop('H', 'P', _p_in_val * 1e5, 'Q', 0, 'Water') / 1000.0  # kJ/kg
        _h_out_f   = self.comps[f'comp_{self.n_stages}'].outl[0].h.val  # kJ/kg
        _m_out_f   = self.comps[f'comp_{self.n_stages}'].outl[0].m.val  # kg/s  (= m_target)
        self.Q_heat = max(0.0, _m_out_f * (_h_out_f - _h_liq_ref))   # kW
        if self.total_power > 0:
            self.COP = self.Q_heat / self.total_power
        else:
            self.COP = np.nan

    def calc_exergy_losses(self, T_amb_C: float = 25.0, p_amb_bar: float = 1.01325) -> dict:
        """
        Manual exergy-destruction analysis for all MVR components.

        Reference state: ambient conditions (T_amb_C, p_amb_bar).
        Physical exergy: e_ph = (h - h0) - T0*(s - s0)   [kJ/kg]

        Returns
        -------
        dict with keys:
            'T_amb_C', 'p_amb_bar',
            'h0_kJ_kg', 's0_kJ_kgK',           reference state
            'E_F_total_kW',                      total electricity input
            'E_P_total_kW',                      net physical exergy gained by steam
            'E_D_total_kW',                      total exergy destruction
            'epsilon_total',                     overall exergetic efficiency
            'stages': list of dicts per compressor (+ mixer between stages)
        """
        from CoolProp.CoolProp import PropsSI as _PSI

        T0 = T_amb_C + 273.15          # K
        p0 = p_amb_bar * 1e5           # Pa

        # Reference state properties (liquid water at ambient)
        h0 = _PSI('H', 'T', T0, 'P', p0, 'Water') / 1000.0   # kJ/kg
        s0 = _PSI('S', 'T', T0, 'P', p0, 'Water') / 1000.0   # kJ/(kg·K)

        def _eph(h_kJ, s_kJ_K):
            """Specific physical exergy [kJ/kg]."""
            return (h_kJ - h0) - T0 * (s_kJ_K - s0)

        stage_data = []

        for k in range(1, self.n_stages + 1):
            comp = self.comps[f'comp_{k}']
            inl  = comp.inl[0]
            outl = comp.outl[0]

            m_in  = inl.m.val              # kg/s
            h_in  = inl.h.val              # kJ/kg
            s_in  = inl.s.val              # kJ/(kg·K)
            h_out = outl.h.val
            s_out = outl.s.val

            e_in  = _eph(h_in,  s_in)
            e_out = _eph(h_out, s_out)

            W_comp_k = m_in * (h_out - h_in)      # kW  (= P.val/1000 for ideal)
            E_P_k    = m_in * (e_out - e_in)       # kW
            E_D_k    = W_comp_k - E_P_k            # kW
            eps_k    = E_P_k / W_comp_k if W_comp_k > 0 else float('nan')

            entry = {
                'component':     f'Compressor {k}',
                'W_comp [kW]':   round(W_comp_k, 3),
                'E_P [kW]':      round(E_P_k,    3),
                'E_D [kW]':      round(E_D_k,    3),
                'ε [-]':         round(eps_k,    4) if not (eps_k != eps_k) else None,
                'm_in [kg/s]':   round(m_in,     5),
                'T_in [°C]':     round(inl.T.val,  2),
                'T_out [°C]':    round(outl.T.val, 2),
            }
            stage_data.append(entry)

            # Mixer between stage k and k+1 (water injection)
            if k < self.n_stages and f'merge_{k}' in self.comps:
                mix = self.comps[f'merge_{k}']
                # Outlet of mixer
                m_out  = mix.outl[0].m.val
                h_m_out = mix.outl[0].h.val
                s_m_out = mix.outl[0].s.val
                e_m_out = _eph(h_m_out, s_m_out)

                # Sum exergy of all inlets
                E_mix_in = 0.0
                for inlet_conn in mix.inl:
                    mi = inlet_conn.m.val
                    hi = inlet_conn.h.val
                    si = inlet_conn.s.val
                    E_mix_in += mi * _eph(hi, si)

                E_mix_out = m_out * e_m_out
                E_D_mix   = E_mix_in - E_mix_out

                stage_data.append({
                    'component':    f'Mixer {k}→{k+1}',
                    'W_comp [kW]':  None,
                    'E_P [kW]':     round(E_mix_out, 3),
                    'E_D [kW]':     round(E_D_mix,   3),
                    'ε [-]':        round(E_mix_out / E_mix_in, 4) if E_mix_in > 0 else None,
                    'm_in [kg/s]':  round(m_out, 5),
                    'T_in [°C]':    None,
                    'T_out [°C]':   round(mix.outl[0].T.val, 2),
                })

        # Overall balance
        comp_1_inl  = self.comps['comp_1'].inl[0]
        comp_n_outl = self.comps[f'comp_{self.n_stages}'].outl[0]
        E_F_total   = self.total_power   # kW
        E_P_total   = (comp_n_outl.m.val * _eph(comp_n_outl.h.val, comp_n_outl.s.val)
                       - comp_1_inl.m.val * _eph(comp_1_inl.h.val, comp_1_inl.s.val))
        # Account for injected water exergy (deducted from product)
        for k in range(1, self.n_stages):
            if f'merge_{k}' in self.comps:
                for inlet_conn in self.comps[f'merge_{k}'].inl[1:]:
                    mi = inlet_conn.m.val
                    hi = inlet_conn.h.val
                    si = inlet_conn.s.val
                    E_P_total -= mi * _eph(hi, si)

        E_D_total = E_F_total - E_P_total
        eps_total = E_P_total / E_F_total if E_F_total > 0 else float('nan')

        return {
            'T_amb_C':        T_amb_C,
            'p_amb_bar':      p_amb_bar,
            'h0_kJ_kg':       round(h0, 3),
            's0_kJ_kgK':      round(s0, 5),
            'E_F_total_kW':   round(E_F_total,  3),
            'E_P_total_kW':   round(E_P_total,  3),
            'E_D_total_kW':   round(E_D_total,  3),
            'epsilon_total':  round(eps_total,  4) if not (eps_total != eps_total) else None,
            'stages':         stage_data,
        }

    def get_plotting_states(self, **kwargs):
        """
        Returns state point data in process order for state diagrams:
        [HTHP Condenser (optional)] → comp_1 → merge_1 → comp_2 → ... → outlet

        Optional kwarg ``feedwater`` (dict) adds the isobaric HTHP-condenser
        heating path as the first labelled segment.  Expected keys:
            T_fw_C  : feedwater inlet temperature [°C]
            p_bar   : heating pressure = MVR inlet pressure [bar]
            T_end_C : steam temperature at comp-1 inlet [°C]
        The segment is drawn in blue to distinguish it from the orange MVR path.
        """
        from CoolProp.CoolProp import PropsSI as _PSI

        data = {}

        # ── Optional: HTHP condenser path (feedwater → comp-1 inlet) ─────────
        fw = kwargs.get('feedwater')
        # Guard: only proceed when dict exists and all values are non-None numbers
        if fw is not None and all(fw.get(k) is not None for k in ('T_fw_C', 'p_bar', 'T_end_C')):
            try:
                T_fw  = float(fw['T_fw_C'])
                p_bar = float(fw['p_bar'])
                T_end = float(fw['T_end_C'])
                p_Pa  = p_bar * 1e5

                # Saturation temperature at inlet pressure
                T_sat = _PSI('T', 'P', p_Pa, 'Q', 1, 'Water') - 273.15

                h_pts, p_pts, s_pts, T_pts = [], [], [], []

                def _add(T_C, Q=None):
                    T_K = T_C + 273.15
                    if Q is not None:
                        h = _PSI('H', 'P', p_Pa, 'Q', Q, 'Water') / 1000
                        s = _PSI('S', 'P', p_Pa, 'Q', Q, 'Water') / 1000
                        T_C_real = _PSI('T', 'P', p_Pa, 'Q', Q, 'Water') - 273.15
                    else:
                        h = _PSI('H', 'T', T_K, 'P', p_Pa, 'Water') / 1000
                        s = _PSI('S', 'T', T_K, 'P', p_Pa, 'Water') / 1000
                        T_C_real = T_C
                    h_pts.append(h); p_pts.append(p_bar)
                    s_pts.append(s); T_pts.append(T_C_real)

                # ── Always start with the feedwater inlet state ───────────────
                # This point is labeled "Feedwater Inlet" in the legend and is
                # shown as the first numbered state point in the diagram.
                # Clamp T_fw away from exact saturation to avoid CoolProp issues.
                if abs(T_fw - T_sat) < 0.05:
                    T_fw_clamped = T_sat - 0.1
                else:
                    T_fw_clamped = T_fw
                _add(T_fw_clamped)

                # ── Trace heating path from T_fw to T_end ────────────────────
                if T_fw < T_sat - 0.1:
                    # Case A: feedwater is subcooled liquid
                    # 1a. Subcooled liquid: T_fw → saturation boundary (skip T_fw, already added)
                    T_sub_end = min(T_sat - 0.1, T_end)
                    for T_step in np.linspace(T_fw_clamped, T_sub_end, 10)[1:]:
                        _add(T_step)
                    # 1b. Two-phase dome: Q=0 → Q=1
                    if T_end >= T_sat - 0.1:
                        for Q_step in np.linspace(0.0, 1.0, 20):
                            _add(None, Q=Q_step)
                    # 1c. Superheated steam: T_sat → T_end
                    if T_end > T_sat + 0.1:
                        for T_step in np.linspace(T_sat + 0.1, T_end, 10):
                            _add(T_step)

                elif T_fw >= T_sat + 0.1:
                    # Case B: feedwater is already superheated steam (e.g. flashed steam)
                    # Path runs entirely in superheated region from T_fw to T_end
                    if T_end > T_fw + 0.1:
                        for T_step in np.linspace(T_fw, T_end, 10)[1:]:
                            _add(T_step)

                else:
                    # Case C: feedwater is at or near saturation (Q~1)
                    # Two-phase dome: Q=0 → Q=1
                    for Q_step in np.linspace(0.0, 1.0, 20):
                        _add(None, Q=Q_step)
                    # Superheated steam: T_sat → T_end
                    if T_end > T_sat + 0.1:
                        for T_step in np.linspace(T_sat + 0.1, T_end, 10):
                            _add(T_step)

                if h_pts:
                    data['Feedwater Inlet'] = {
                        '_skip_isoline': True,
                        '_color_line':   '#1976D2',   # blue — HTHP/water-side contribution
                        '_color_marker': '#0D47A1',
                        'datapoints': {
                            'h': h_pts,
                            'p': p_pts,
                            's': s_pts,
                            'T': T_pts,
                        },
                    }
            except Exception as _e:
                print(f'  ⚠ feedwater path skipped: {_e}')

        for stage in range(1, self.n_stages + 1):
            # ── Compression segment ───────────────────────────────────────────
            comp = self.comps[f'comp_{stage}']
            comp_label = comp.label
            comp_data = comp.get_plotting_data()[1]
            # Slight offset on starting_point_value prevents TESPy isoline issues
            comp_data['starting_point_value'] *= 0.999999
            data[comp_label] = comp_data

            # ── Water injection / mixing (all stages except the last) ─────────
            if stage < self.n_stages:
                merge_label = self.comps[f'merge_{stage}'].label
                try:
                    merge_raw = self.comps[f'merge_{stage}'].get_plotting_data()
                    if merge_raw and len(merge_raw) > 1:
                        data[merge_label] = merge_raw[1]
                except Exception:
                    pass

        # ── Final outlet of the last compressor ───────────────────────────────
        # The plotting loop marks only the *start* of each segment, so the
        # last discharge point would be missing.  We add a synthetic entry
        # with '_skip_isoline': True so MVRBase skips calc_individual_isoline
        # and uses the pre-filled datapoints directly.
        last_comp = self.comps[f'comp_{self.n_stages}']
        outlet = last_comp.outl[0]
        try:
            # Network units: h in kJ/kg, T in °C, p in bar, s in kJ/(kgK)
            # — .val already returns values in those units, no conversion needed.
            data['Outlet'] = {
                '_skip_isoline': True,
                'datapoints': {
                    'h': [outlet.h.val],
                    'p': [outlet.p.val],
                    's': [outlet.s.val],
                    'T': [outlet.T.val],
                }
            }
        except Exception:
            pass

        return data

    def print_summary(self):
        """Gibt eine detaillierte Zusammenfassung der Ergebnisse aus."""
        print('\n' + '='*80)
        print(f'MVR MULTI-STAGE MIT WASSEREINSPRITZUNG - ZUSAMMENFASSUNG ({self.n_stages} Stufen)')
        print('='*80)
        
        print(f'\nWorking fluid: {self.wf}')
        print(f'Ziel-Ausgangsmassenstrom:      {self.m_target:.4f} kg/s  (Vorgabe)')
        print(f'Eintrittsmassenstrom (rückger.): {self.m_design:.4f} kg/s')
        print(f'Gesamt eingespritzes Wasser:   {self.total_water_injected:.4f} kg/s')
        _check = self.m_design + self.total_water_injected
        print(f'  Kontrollrechnung (m_in + sum m_inj = {_check:.4f} kg/s, Ziel: {self.m_target:.4f} kg/s)')
        
        print('\n--- INLET ---')
        inlet_state = self.get_state_properties('0')
        print(f'  T: {inlet_state["T"]:.2f} °C')
        print(f'  p: {inlet_state["p"]:.3f} bar')
        print(f'  h: {inlet_state["h"]:.2f} kJ/kg')
        
        # Details für jede Stufe
        for stage in range(1, self.n_stages + 1):
            print(f'\n--- STUFE {stage} ---')
            stage_data = self.stage_results[f'stage_{stage}']
            print(f'  Eingang:')
            print(f'    T: {stage_data["T_in"]:.2f} °C')
            print(f'    p: {stage_data["p_in"]:.3f} bar')
            print(f'    m: {stage_data["m"]:.3f} kg/s')
            print(f'  Ausgang:')
            print(f'    T: {stage_data["T_out"]:.2f} °C')
            print(f'    p: {stage_data["p_out"]:.3f} bar')
            print(f'  Performance:')
            print(f'    PR: {stage_data["PR"]:.3f}')
            print(f'    η_s: {stage_data["eta_s"]:.3f}')
            print(f'    Spez. Arbeit: {stage_data["specific_work"]:.2f} kJ/kg')
            print(f'    Power: {stage_data["P"]:.1f} kW')
            
            # Wassereinspritzung nach dieser Stufe (falls vorhanden)
            if stage < self.n_stages:
                print(f'\n--- WASSEREINSPRITZUNG {stage} ---')
                m_water = self.water_injection_rates[stage - 1]
                T_after = self.comps[f'merge_{stage}'].outl[0].T.val
                print(f'  Eingespritzte Wassermenge: {m_water:.4f} kg/s')
                print(f'  T nach Kühlung: {T_after:.2f} °C')
        
        print('\n--- AUSGANG (GESAMT) ---')
        last_conn = f'{self.n_stages * 2 - 1}'
        outlet_state = self.get_state_properties(last_conn)
        print(f'  T: {outlet_state["T"]:.2f} °C')
        print(f'  p: {outlet_state["p"]:.3f} bar')
        print(f'  h: {outlet_state["h"]:.2f} kJ/kg')
        print(f'  m: {outlet_state["m"]:.3f} kg/s')
        
        print('\n--- GESAMT-PERFORMANCE ---')
        print(f'  Kompressionsverhältnis (gesamt): {self.compression_ratio:.3f}')
        print(f'  Spezifische Arbeit (gesamt): {self.specific_work:.2f} kJ/kg')
        print(f'  Leistungsaufnahme (gesamt): {self.total_power:.1f} kW')
        print(f'  Volumetric flow (inlet): {self.volumetric_flow_in:.1f} m³/h')
        
        print('\n--- ENERGETISCHE BEWERTUNG (offenes System) ---')
        print(f'  SEI (Specific Energy Input):')
        print(f'    {self.SEI:.3f} kWh/kg Dampf')
        print(f'    {self.SEI_MJ_per_kg:.3f} MJ/kg Dampf')
        print(f'  Hinweis: SEI gibt die benötigte Energie pro kg erzeugtem Dampf an.')
        print(f'           Niedriger SEI = bessere energetische Effizienz')
        
        print('='*80 + '\n')

    def get_stage_info(self):
        """Gibt detaillierte Info über alle Stufen zurück."""
        info = {
            'n_stages': self.n_stages,
            'total_power': self.total_power,
            'total_compression_ratio': self.compression_ratio,
            'SEI_kWh_per_kg': float(self.SEI),
            'SEI_MJ_per_kg': float(self.SEI_MJ_per_kg),
            'total_water_injected_kg_per_s': self.total_water_injected,
            'm_steam_in': self.m_design,
            'm_total_out': self.m_design + self.total_water_injected,
            'stages': {}
        }
        
        for stage in range(1, self.n_stages + 1):
            info['stages'][f'stage_{stage}'] = self.stage_results[f'stage_{stage}']
        
        info['water_injections'] = {}
        for stage in range(1, self.n_stages):
            info['water_injections'][f'injection_{stage}'] = {
                'm_water': self.water_injection_rates[stage - 1],
                'T_in': self.params['cooling_water']['T_in'],
                'T_out': self.comps[f'merge_{stage}'].outl[0].T.val
            }
        
        return info


if __name__ == '__main__':
    from params_mvr import create_multistage_params
    
    print("="*80)
    print("MEHRSTUFIGER MVR MIT WASSEREINSPRITZUNG - BEISPIEL")
    print("="*80)
    
    # Parameter erstellen
    params = create_multistage_params(
        name='MVR 3-Stufig mit Wassereinspritzung',
        n_stages=3,
        working_fluid='Water',
        T_inlet=100.0,
        p_inlet=1.013,
        m_inlet=1.0,
        PR_total=8.0,
        eta_s=0.85,
        superheat=0.0,
        T_cw_in=15,
        delta_T_cw=25
    )
    
    # MVR-Model erstellen und ausführen
    mvr = MVRMultiStage(params=params)
    mvr.run_model(print_results=False)
    
    # Detaillierte Zusammenfassung
    mvr.print_summary()
    
    # Zustandsdiagramm generieren
    try:
        mvr.generate_state_diagram(diagram_type='logph', savefig=True)
    except Exception as e:
        print(f"⚠ Error in diagram generation: {e}")