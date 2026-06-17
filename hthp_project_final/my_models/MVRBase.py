"""
MVRBase - Basis-Klasse für MVR-Modelle (Mechanical Vapor Recompression)

MVR ist ein OFFENES System (kein Kreislauf):
Eingang (Dampf) → Verdichter → Ausgang (Hochdruck-Dampf)

Hauptkennzahl: SEI (Specific Energy Input) [kWh/kg oder MJ/kg]
- Gibt an, wie viel Energie pro kg Dampf aufgewendet werden muss
- Niedriger SEI = bessere energetische Effizienz
- COP ist für offene Systeme ohne Wärmerückgewinnung nicht sinnvoll

Verwendet TurboCompressor von TESPy.
Erweitert um generate_state_diagram nach HeatPumpBase-Vorbild.
"""

import json
import os
from datetime import datetime
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI as PSI
from fluprodia import FluidPropertyDiagram
from tespy.networks import Network
from tespy.tools import ExergyAnalysis
from tespy.tools.characteristics import CharLine
from tespy.tools.characteristics import load_default_char as ldc


class MVRBase:
    """Basis-Klasse für alle MVR-Modelle."""

    def __init__(self, params):
        """Initialisiert das MVR-Modell und setzt notwendige Attribute."""
        self.params = params

        # TESPy Netzwerk initialisieren
        self.nw = Network(
            T_unit='C', p_unit='bar', h_unit='kJ / kg', m_unit='kg / s'
        )

        self._init_fluids()

        # Komponenten, Verbindungen und Busse
        self.comps = dict()
        self.conns = dict()
        self.buses = dict()

        # Performance-Kennzahlen
        self.compression_ratio = np.nan
        self.specific_work = np.nan  # kJ/kg
        self.power_consumption = np.nan  # kW
        self.eta_s_overall = np.nan
        self.volumetric_flow_in = np.nan  # m³/h
        self.solved_design = False
        
        # SEI und COP
        self.SEI = np.nan           # kWh/kg Dampf
        self.SEI_MJ_per_kg = np.nan # MJ/kg Dampf
        self.COP = np.nan           # Q_nutz / W_el  [-]
        self.Q_heat = np.nan        # Nutzwärme an Prozess [kW]

        # Initialisierungswerte
        self._init_vals = {
            'dh_rel_comp': 1.15,  # Für Startwerte
            'pr_default': 0.99    # Default Druckverlust
        }

        self._init_dir_paths()

    def _init_fluids(self):
        """Initialisiert Fluid-Attribute."""
        self.wf = self.params['fluids']['wf']  # Working fluid (Dampf)

    def _init_dir_paths(self):
        """Initialisiert Verzeichnispfade für Speicherung."""
        # Analog zu HeatPumpBase
        self.subdirname = self.params['setup']['name'].replace(' ', '_')
        
        # Design-Pfad
        self.design_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'stable', f'{self.subdirname}_design'
        ))
        
        # Verzeichnisse erstellen falls nicht vorhanden
        stable_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'stable'
        ))
        if not os.path.exists(stable_dir):
            os.makedirs(stable_dir)
        
        output_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'output'
        ))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_components(self):
        """Initialisiert Komponenten des MVR-Systems."""
        # Muss in abgeleiteter Klasse implementiert werden
        pass

    def generate_connections(self):
        """Initialisiert und fügt Verbindungen und Busse zum Netzwerk hinzu."""
        # Muss in abgeleiteter Klasse implementiert werden
        pass

    def init_simulation(self, **kwargs):
        """Führt initiale Parametrierung mit Startwerten durch."""
        # Muss in abgeleiteter Klasse implementiert werden
        pass

    def design_simulation(self, **kwargs):
        """Führt finale Parametrierung und Design-Simulation durch."""
        # Muss in abgeleiteter Klasse implementiert werden
        pass

    def _solve_model(self, **kwargs):
        """Löst das Modell im Design-Modus."""
        if 'iterinfo' in kwargs:
            self.nw.set_attr(iterinfo=kwargs['iterinfo'])
        
        self.nw.solve('design', print_results=False)

        if 'print_results' in kwargs:
            if kwargs['print_results']:
                self.nw.print_results()
        
        if self.nw.residual[-1] < 1e-3:
            self.solved_design = True
            self.nw.save(self.design_path)

    def calc_performance(self):
        """
        Berechnet Performance-Kennzahlen des MVR-Systems.
        
        Für offene MVR-Systeme ohne Wärmerückgewinnung:
        - SEI (Specific Energy Input) statt COP
        - SEI = P_el / m_dampf [kWh/kg]
        - Niedriger SEI = bessere Effizienz
        """
        # Spezifische Arbeit (kJ/kg)
        if hasattr(self, 'comps') and 'comp' in self.comps:
            comp = self.comps['comp']
            h_in = comp.inl[0].h.val
            h_out = comp.outl[0].h.val
            self.specific_work = h_out - h_in
            
            # Kompressionsverhältnis
            p_in = comp.inl[0].p.val
            p_out = comp.outl[0].p.val
            self.compression_ratio = p_out / p_in
            
            # Leistungsaufnahme (kW)
            m_dot = comp.inl[0].m.val
            self.power_consumption = abs(self.buses['power input'].P.val) / 1000.0  # W → kW
            
            # Volumenstrom am Eintritt (m³/h)
            self.volumetric_flow_in = comp.inl[0].vol.val_SI * 3600
            
            # Isentroper Gesamtwirkungsgrad
            self.eta_s_overall = comp.eta_s.val
            
            # SEI berechnen (Specific Energy Input)
            # SEI = P_el / m_dampf [kWh/kg]
            if m_dot > 0:
                self.SEI = self.power_consumption / m_dot / 3600  # kW/(kg/s)/3600 = kWh/kg
                self.SEI_MJ_per_kg = self.SEI * 3.6  # kWh/kg → MJ/kg
            else:
                self.SEI = np.nan
                self.SEI_MJ_per_kg = np.nan

            # COP berechnen (analog zu HeatPump: Q_nutz / W_el)
            # Q_nutz = m_out * (h_out - h_liq_sat(p_in)) [kW]
            # Referenz: gesättigte Flüssigkeit bei Eintrittsdruck
            h_liq_ref = PSI('H', 'P', p_in * 1e5, 'Q', 0, self.wf) / 1000.0  # kJ/kg
            m_out = comp.outl[0].m.val   # kg/s
            self.Q_heat = max(0.0, m_out * (h_out - h_liq_ref))  # kW
            if self.power_consumption > 0:
                self.COP = self.Q_heat / self.power_consumption
            else:
                self.COP = np.nan

    def run_model(self, print_results=False, exergy_analysis=False, **kwargs):
        """Führt die komplette Initialisierung und Design-Simulation aus."""
        self.generate_components()
        self.generate_connections()
        self.init_simulation(**kwargs)
        self.design_simulation(**kwargs)
        self.check_consistency()
        self.calc_performance()
        
        if exergy_analysis:
            self.perform_exergy_analysis(**kwargs)
        
        if print_results:
            print('\n' + '='*70)
            print('MVR PERFORMANCE RESULTS (Offenes System)')
            print('='*70)
            print(f'Kompressionsverhältnis: {self.compression_ratio:.3f}')
            print(f'Spezifische Arbeit: {self.specific_work:.2f} kJ/kg')
            print(f'Leistungsaufnahme: {self.power_consumption:.1f} kW')
            print(f'Volumenstrom (Eintritt): {self.volumetric_flow_in:.1f} m³/h')
            print(f'Isentroper Wirkungsgrad: {self.eta_s_overall:.3f}')
            print('\nENERGETISCHE BEWERTUNG:')
            print(f'  COP (Q_nutz / W_el): {self.COP:.3f}')
            print(f'  Q_nutz: {self.Q_heat:.1f} kW')
            print(f'  SEI (Specific Energy Input):')
            print(f'    {self.SEI:.3f} kWh/kg Dampf')
            print(f'    {self.SEI_MJ_per_kg:.3f} MJ/kg Dampf')
            print(f'  Hinweis: Niedriger SEI = bessere Effizienz')
            print('='*70)

    def check_consistency(self):
        """Überprüft Konsistenz der Simulationsergebnisse."""
        if not self.solved_design:
            print('⚠ Warnung: Design-Simulation nicht konvergiert!')
            print(f'  Residual: {self.nw.residual[-1]:.2e}')
        else:
            print(f'✓ Design-Simulation erfolgreich (Residual: {self.nw.residual[-1]:.2e})')

    def perform_exergy_analysis(self, **kwargs):
        """Führt Exergie-Analyse durch."""
        try:
            self.ean = ExergyAnalysis(
                network=self.nw,
                E_P=[self.buses['power input']],
                E_F=[self.comps['inlet']]
            )
            self.ean.analyse(pamb=self.params['ambient']['p'], Tamb=self.params['ambient']['T'])
            
            if 'print_results' in kwargs:
                if kwargs['print_results']:
                    print('\n--- EXERGIE-ANALYSE ---')
                    print(self.ean.component_data)
        except Exception as e:
            print(f'⚠ Warnung: Exergie-Analyse fehlgeschlagen: {e}')

    def get_state_properties(self, conn_label):
        """Gibt thermodynamische Eigenschaften einer Verbindung zurück."""
        if conn_label not in self.conns:
            raise ValueError(f'Verbindung {conn_label} nicht gefunden!')
        
        conn = self.conns[conn_label]
        return {
            'T': conn.T.val,
            'p': conn.p.val,
            'h': conn.h.val,
            's': conn.s.val,
            'm': conn.m.val,
            'vol': conn.vol.val_SI,
            'x': conn.x.val if hasattr(conn, 'x') else None
        }

    def get_compressor_results(self):
        """Gibt detaillierte Verdichter-Ergebnisse zurück."""
        results = {}
        
        if 'comp' in self.comps:
            comp = self.comps['comp']
            results['single_stage'] = {
                'P': comp.P.val / 1000.0,  # W → kW
                'eta_s': comp.eta_s.val,
                'PR': comp.pr.val,
                'inlet_T': comp.inl[0].T.val,
                'inlet_p': comp.inl[0].p.val,
                'outlet_T': comp.outl[0].T.val,
                'outlet_p': comp.outl[0].p.val
            }
        
        # Für mehrstufige Systeme
        stage = 1
        while f'comp_{stage}' in self.comps:
            comp = self.comps[f'comp_{stage}']
            results[f'stage_{stage}'] = {
                'P': comp.P.val / 1000.0,  # W → kW
                'eta_s': comp.eta_s.val,
                'PR': comp.outl[0].p.val / comp.inl[0].p.val,
                'inlet_T': comp.inl[0].T.val,
                'inlet_p': comp.inl[0].p.val,
                'outlet_T': comp.outl[0].T.val,
                'outlet_p': comp.outl[0].p.val,
                'm': comp.inl[0].m.val
            }
            stage += 1
        
        return results

    def get_plotting_states(self, **kwargs):
        """
        Erzeugt Daten für Zustandsdiagramm.
        Muss in abgeleiteten Klassen implementiert werden.
        """
        raise NotImplementedError("Muss in abgeleiteter Klasse implementiert werden")

    def generate_state_diagram(
        self, diagram_type='logph', savefig=True, filepath=None,
        legend=True, legend_loc='lower right', fontsize=12,
        isoline_data=None, return_diagram=False, open_file=False, **kwargs
    ):
        """
        Generiert Zustandsdiagramm für MVR-Prozess.
        Analog zu HeatPumpBase.generate_state_diagram()
        
        Args:
            diagram_type (str): 'logph' oder 'Ts'
            savefig (bool): Diagramm speichern
            filepath (str): Speicherpfad (optional)
            legend (bool): Legende anzeigen
            legend_loc (str): Position der Legende
            fontsize (int): Schriftgröße
            isoline_data (str): 'default' oder None
            return_diagram (bool): FluidPropertyDiagram zurückgeben
            open_file (bool): Datei nach Speichern öffnen
        """
        # Mapping für Kältemittel-Namen (CoolProp vs. TESPy)
        refrigerant_mapping = {
            'R600a': 'IsoButane',
            'R290': 'Propane',
            'R1234ze(E)': 'R1234zeE',
            'Water': 'Water'
        }
        
        refrig = refrigerant_mapping.get(self.wf, self.wf)
        
        # Diagram-Typ Konfiguration
        if diagram_type == 'logph':
            var = {
                'x': 'h',
                'y': 'p',
                'isolines': ['T', 's']
            }
        elif diagram_type == 'Ts':
            var = {
                'x': 's',
                'y': 'T',
                'isolines': ['p', 'h']
            }
        else:
            raise ValueError(f"Unbekannter diagram_type: {diagram_type}. Verwende 'logph' oder 'Ts'")
        
        # Plotting-Daten von Komponenten holen
        result_dict = self.get_plotting_states(**kwargs)
        
        # Figure erstellen
        fig, ax = plt.subplots(1, figsize=(12, 8))
        
        # Pfad für diagram data
        diagram_data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'input', f'{diagram_type}_{refrig}.json'
        ))
        
        # Generiere Isolinien
        path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'input', 'state_diagram_config.json'
        ))
        
        # Prüfe ob config-Datei existiert
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as file:
                config = json.load(file)

            if refrig in config:
                state_props = config[refrig]
            else:
                state_props = config.get('MISC', {})
        else:
            # Fallback wenn keine config vorhanden
            state_props = None

        if os.path.isfile(diagram_data_path):
            diagram = FluidPropertyDiagram.from_json(diagram_data_path)
        else:
            diagram = FluidPropertyDiagram(refrig)
            diagram.set_unit_system(T='°C', p='bar', h='kJ/kg')

            if state_props:
                iso1 = np.arange(
                    state_props[var['isolines'][0]]['isorange_low'],
                    state_props[var['isolines'][0]]['isorange_high'],
                    state_props[var['isolines'][0]]['isorange_step']
                )
                iso2 = np.arange(
                    state_props[var['isolines'][1]]['isorange_low'],
                    state_props[var['isolines'][1]]['isorange_high'],
                    state_props[var['isolines'][1]]['isorange_step']
                )

                diagram.set_isolines(**{
                    var['isolines'][0]: iso1,
                    var['isolines'][1]: iso2
                })
            
            diagram.calc_isolines()
            
            # Speichere diagram data für zukünftige Verwendung
            try:
                os.makedirs(os.path.dirname(diagram_data_path), exist_ok=True)
                diagram.to_json(diagram_data_path)
            except:
                pass  # Ignoriere Fehler beim Speichern

        # Compute process data for each segment.
        # Entries with '_skip_isoline': True already have 'datapoints' pre-filled
        # (e.g. the synthetic Outlet point) and must not go through calc_individual_isoline.
        for compdata in result_dict.values():
            if compdata.get('_skip_isoline'):
                continue
            compdata['datapoints'] = diagram.calc_individual_isoline(**compdata)
        
        diagram.fig = fig
        diagram.ax = ax

        # Setze Achsenlimits
        if 'xlims' in kwargs:
            xlims = kwargs['xlims']
        else:
            if state_props:
                xlims = (
                    state_props[var['x']]['min'], state_props[var['x']]['max']
                )
            else:
                xlims = None
        
        if 'ylims' in kwargs:
            ylims = kwargs['ylims']
        else:
            if state_props:
                ylims = (
                    state_props[var['y']]['min'], state_props[var['y']]['max']
                )
            else:
                ylims = None

        # Zeichne Isolinien
        if xlims and ylims:
            diagram.draw_isolines(
                diagram_type=diagram_type, fig=fig, ax=ax,
                x_min=xlims[0], x_max=xlims[1], y_min=ylims[0], y_max=ylims[1],
                isoline_data=isoline_data
            )

        # Draw process segments and label state points in process order
        n_process = len(result_dict)
        for i, key in enumerate(result_dict.keys()):
            datapoints = result_dict[key]['datapoints']
            xvals = datapoints.get(var['x'], [])
            yvals = datapoints.get(var['y'], [])
            has_data = len(xvals) > 0 and len(yvals) > 0

            if not has_data:
                ax.scatter(0, 0, color='#FFFFFF', s=0, alpha=1.0,
                           label=f'$\\bf{i+1:.0f}$: {key}')
                ax.annotate('Error\nMissing Plotting Data', (0.5, 0.5),
                            xycoords='axes fraction', ha='center', va='center',
                            fontsize=60, color='#B54036')
                continue

            is_outlet = (key == 'Outlet')

            # Per-segment colour overrides (e.g. for the HTHP condenser path)
            seg      = result_dict[key]
            c_line   = seg.get('_color_line',   '#EC6707')
            c_marker = seg.get('_color_marker', '#B54036')

            if not is_outlet:
                # Draw compression / mixing / isobaric heating segment
                ax.plot(xvals[:], yvals[:], color=c_line)

            # Label the point
            # For normal segments: mark the *start* (suction / inlet)
            # For the Outlet entry: mark the single point (last discharge)
            px, py = xvals[0], yvals[0]
            ax.scatter(px, py, color=c_marker,
                       label=f'$\\bf{i+1:.0f}$: {key}',
                       s=14*int(fontsize*0.9), alpha=0.5)
            ax.annotate(f'{i+1:.0f}', (px, py),
                        ha='center', va='center', color='w',
                        fontsize=int(fontsize*0.9))

        # Formatting - add refrigerant name to bottom-left corner
        ax.text(0.02, 0.02, refrig, transform=ax.transAxes,
                fontsize=int(fontsize*1.2), verticalalignment='bottom',
                horizontalalignment='left', bbox=dict(boxstyle='round',
                facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))

        if diagram_type == 'logph':
            ax.set_xlabel('Specific enthalpy in $kJ/kg$', fontsize=fontsize)
            ax.set_ylabel('Pressure in $bar$', fontsize=fontsize)
        elif diagram_type == 'Ts':
            ax.set_xlabel('Specific entropy in $kJ/(kg \\cdot K)$', fontsize=fontsize)
            ax.set_ylabel('Temperature in $°C$', fontsize=fontsize)

        ax.tick_params(axis='both', labelsize=int(fontsize*0.9))

        if legend:
            ax.legend(
                loc='upper left',
                prop={'size': fontsize * (1 - 0.02 * len(result_dict))},
                markerscale=(1 - 0.02 * len(result_dict))
            )

        # Speichern
        if savefig:
            if filepath is None:
                filename = (
                    f'{diagram_type}_{self.params["setup"]["type"]}_{refrig}.pdf'
                )
                filepath = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), 'output', filename
                ))
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=300)
            print(f'✓ Zustandsdiagramm gespeichert: {filepath}')
            
            if open_file:
                import subprocess
                import platform
                if platform.system() == 'Darwin':       # macOS
                    subprocess.call(('open', filepath))
                elif platform.system() == 'Windows':    # Windows
                    os.startfile(filepath)
                else:                                   # linux variants
                    subprocess.call(('xdg-open', filepath))

        if return_diagram:
            return diagram

    def export_results(self, filename=None):
        """Exportiert Ergebnisse als JSON."""
        if filename is None:
            filename = f'{self.subdirname}_results.json'
        
        output_path = os.path.join(
            os.path.dirname(__file__), 'output', filename
        )
        
        # Sammle Ergebnisse
        results = {
            'setup': self.params['setup'],
            'performance': {
                'compression_ratio': float(self.compression_ratio),
                'specific_work_kJ_per_kg': float(self.specific_work),
                'power_consumption_kW': float(self.power_consumption),
                'volumetric_flow_m3_per_h': float(self.volumetric_flow_in),
                'eta_s_overall': float(self.eta_s_overall),
                'SEI_kWh_per_kg': float(self.SEI),
                'SEI_MJ_per_kg': float(self.SEI_MJ_per_kg),
                'solved_design': self.solved_design,
                'note': 'SEI = Specific Energy Input (offenes System ohne COP)'
            },
            'compressor_details': self.get_compressor_results(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Zustände hinzufügen
        results['states'] = {}
        for conn_label in self.conns.keys():
            try:
                results['states'][conn_label] = self.get_state_properties(conn_label)
            except:
                pass
        
        # JSON speichern
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f'✓ Ergebnisse exportiert: {output_path}')
        
        return results