"""
State Diagram Generator für Hochtemperatur-Wärmepumpen
Erzeugt log(p)-h und T-s Diagramme für HeatPumpSimple, HeatPumpIHX und andere Modele.
Für Kaskadensysteme wird zusätzlich ein kombiniertes Diagramm beider Kältemittelkreise
auf gemeinsamer Achse erzeugt (rote vs. blaue Farbfamilie).
"""

import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from importlib import resources as _importlib_resources
from CoolProp.CoolProp import PropsSI as PSI
from fluprodia import FluidPropertyDiagram


class StateDiagramGenerator:
    """Generator für Zustandsdiagramme (log(p)-h und T-s)"""
    
    def __init__(self, heat_pump, results_dir='results/plots', style='light'):
        """
        Initialisiert den Diagram Generator.
        
        Parameters
        ----------
        heat_pump : HeatPumpBase
            Wärmepumpen-Objekt (HeatPumpSimple, HeatPumpIHX, etc.)
        results_dir : str
            Verzeichnis für die Ausgabe
        style : str
            'light' oder 'dark' für den Diagrammstil
        """
        self.hp = heat_pump
        self.results_dir = results_dir
        self.style = style
        
        # Stelle sicher, dass das Verzeichnis existiert
        os.makedirs(results_dir, exist_ok=True)
        
        # Bestimme das Refrigerant
        self.wf = heat_pump.wf if hasattr(heat_pump, 'wf') else None
        
        # Prüfe ob es ein Zweistoff-System ist
        self.nr_refrigs = self._detect_nr_refrigerants()
    
    def _detect_nr_refrigerants(self):
        """Erkennt ob es sich um ein Ein- oder Zweistoff-System handelt"""
        # Prüfe auf wf1 und wf2 Attribute (für cascade systems)
        if hasattr(self.hp, 'wf1') and hasattr(self.hp, 'wf2'):
            return 2
        return 1
    
    def calc_limits(self, wf, prop='h', padding_rel=0.35, scale='linear'):
        """
        Calculates axis limits from the *solved* TESPy network results.

        Reads hp.nw.results['Connection'], filters to rows belonging to
        working fluid `wf`, and derives min/max from the actual solved
        state-point values — identical to the approach in hp_dashboard.py.

        Parameters
        ----------
        wf          : CoolProp fluid name, e.g. 'IsoButane', 'Ammonia'
        prop        : TESPy Connection column: 'h', 'p', 'T', 's'
        padding_rel : Fractional padding around the data range (0.35 = 35 %)
        scale       : 'linear' or 'log'
        """
        if scale not in ('linear', 'log'):
            raise ValueError(f"scale must be 'linear' or 'log', got '{scale}'")

        try:
            conn    = self.hp.nw.results['Connection']
            mask    = conn[wf] == 1.0
            min_val = float(conn.loc[mask, prop].min())
            max_val = float(conn.loc[mask, prop].max())

            if scale == 'linear':
                delta = max_val - min_val
                xmin  = min_val - padding_rel * delta
                xmax  = max_val + padding_rel * delta
            else:  # log — padding applied in log10 space (hp_dashboard approach)
                log_min   = np.log10(min_val)
                log_max   = np.log10(max_val)
                delta_log = log_max - log_min
                xmin = 10 ** (log_min - padding_rel * delta_log)
                xmax = 10 ** (log_max + padding_rel * delta_log)

            return xmin, xmax

        except Exception as e:
            print(f"  ⚠ Warning in calc_limits for prop='{prop}', wf='{wf}': {e}")
            defaults = {
                'h': (0, 800),
                'p': (1, 100),
                'T': (-50, 200),
                's': (0, 4),
            }
            return defaults.get(prop, (0, 100))


    def generate_diagrams(self, scenario_name='scenario', figsize=(12, 7.5)):
        """
        Generiert beide Diagramme (log(p)-h und T-s) und speichert sie.

        Parameters
        ----------
        scenario_name : str
            Name des Szenarios fuer Dateinamen
        figsize : tuple
            Groesse der Figur (width, height) in inches

        Returns
        -------
        dict
            Dictionary mit Pfaden zu den gespeicherten Diagrammen
        """
        saved_files = {}

        try:
            if not hasattr(self.hp, 'solved_design') or not self.hp.solved_design:
                print(f"  Warning: Model {scenario_name} was not successfully solved. Skipping diagrams.")
                return saved_files

            print(f"\n  Generiere Zustandsdiagramme fuer {scenario_name}...")

            logph_path = self._generate_logph_diagram(scenario_name, figsize)
            if logph_path:
                saved_files['logph'] = logph_path
                print(f"    ok log(p)-h diagram: {logph_path}")

            ts_path = self._generate_ts_diagram(scenario_name, figsize)
            if ts_path:
                saved_files['Ts'] = ts_path
                print(f"    ok T-s diagram: {ts_path}")

            return saved_files

        except Exception as e:
            print(f"  Error in diagram generation fuer {scenario_name}: {e}")
            return saved_files
    
    # ── Axis label translations ───────────────────────────────────────────────
    _AXIS_LABELS = {
        'logph': {
            'x': 'Specific enthalpy in $kJ/kg$',
            'y': 'Pressure in $bar$',
        },
        'Ts': {
            'x': 'Specific entropy in $kJ/(kg\\cdot K)$',
            'y': 'Temperature in $°C$',
        },
    }

    @staticmethod
    def _relabel_axes(diagram, diagram_type: str):
        """Overwrite axis labels on a returned FluidPropertyDiagram to English."""
        labels = StateDiagramGenerator._AXIS_LABELS.get(diagram_type)
        if labels is None or not hasattr(diagram, 'ax'):
            return
        ax = diagram.ax
        ax.set_xlabel(labels['x'])
        ax.set_ylabel(labels['y'])

    def _generate_logph_diagram(self, scenario_name, figsize):
        """Generiert log(p)-h Diagramm"""
        try:
            if self.nr_refrigs == 1:
                #Einfaches System mit einem Refrigerant
                xmin, xmax = self.calc_limits(
                    wf=self.wf, prop='h', padding_rel=0.35
                )
                ymin, ymax = self.calc_limits(
                    wf=self.wf, prop='p', padding_rel=0.25, scale='log'
                )
                
                diagram = self.hp.generate_state_diagram(
                    diagram_type='logph',
                    figsize=figsize,
                    xlims=(xmin, xmax),
                    ylims=(ymin, ymax),
                    style=self.style,
                    return_diagram=True,
                    display_info=False,
                    open_file=False,
                    savefig=False
                )
                
                # Speichere Diagramm als PDF und PNG
                filepath_pdf = os.path.join(
                    self.results_dir, f'{scenario_name}_logph.pdf'
                )
                filepath_png = os.path.join(
                    self.results_dir, f'{scenario_name}_logph.png'
                )
                self._relabel_axes(diagram, 'logph')
                diagram.fig.savefig(filepath_pdf, bbox_inches='tight')
                diagram.fig.savefig(filepath_png, bbox_inches='tight')
                diagram.fig.clf()  # Clear figure
                
                return filepath_png
                
            elif self.nr_refrigs == 2:
                # Zweistoff-System (z.B. Cascade)
                xmin1, xmax1 = self.calc_limits(
                    wf=self.hp.wf1, prop='h', padding_rel=0.35
                )
                ymin1, ymax1 = self.calc_limits(
                    wf=self.hp.wf1, prop='p', padding_rel=0.25, scale='log'
                )
                
                xmin2, xmax2 = self.calc_limits(
                    wf=self.hp.wf2, prop='h', padding_rel=0.35
                )
                ymin2, ymax2 = self.calc_limits(
                    wf=self.hp.wf2, prop='p', padding_rel=0.25, scale='log'
                )
                
                diagram1, diagram2 = self.hp.generate_state_diagram(
                    diagram_type='logph',
                    figsize=figsize,
                    xlims=((xmin1, xmax1), (xmin2, xmax2)),
                    ylims=((ymin1, ymax1), (ymin2, ymax2)),
                    style=self.style,
                    return_diagram=True,
                    display_info=False,
                    savefig=False,
                    open_file=False
                )
                
                # Speichere beide Diagramme
                filepath1_pdf = os.path.join(
                    self.results_dir, f'{scenario_name}_logph_cycle1.pdf'
                )
                filepath1_png = os.path.join(
                    self.results_dir, f'{scenario_name}_logph_cycle1.png'
                )
                filepath2_pdf = os.path.join(
                    self.results_dir, f'{scenario_name}_logph_cycle2.pdf'
                )
                filepath2_png = os.path.join(
                    self.results_dir, f'{scenario_name}_logph_cycle2.png'
                )
                
                self._relabel_axes(diagram1, 'logph')
                self._relabel_axes(diagram2, 'logph')
                diagram1.fig.savefig(filepath1_pdf, bbox_inches='tight')
                diagram1.fig.savefig(filepath1_png, bbox_inches='tight')
                diagram2.fig.savefig(filepath2_pdf, bbox_inches='tight')
                diagram2.fig.savefig(filepath2_png, bbox_inches='tight')
                diagram1.fig.clf()
                diagram2.fig.clf()
                
                return [filepath1_png, filepath2_png]
        
        except Exception as e:
            print(f"    ⚠ Error in log(p)-h diagram: {e}")
            return None
    
    def _generate_ts_diagram(self, scenario_name, figsize):
        """Generiert T-s Diagramm"""
        try:
            if self.nr_refrigs == 1:
                # Einfaches System mit einem Refrigerant
                xmin, xmax = self.calc_limits(
                    wf=self.wf, prop='s', padding_rel=0.35
                )
                ymin, ymax = self.calc_limits(
                    wf=self.wf, prop='T', padding_rel=0.25
                )
                
                diagram = self.hp.generate_state_diagram(
                    diagram_type='Ts',
                    figsize=figsize,
                    xlims=(xmin, xmax),
                    ylims=(ymin, ymax),
                    style=self.style,
                    return_diagram=True,
                    display_info=False,
                    open_file=False,
                    savefig=False
                )
                
                # Speichere Diagramm als PDF und PNG
                filepath_pdf = os.path.join(
                    self.results_dir, f'{scenario_name}_Ts.pdf'
                )
                filepath_png = os.path.join(
                    self.results_dir, f'{scenario_name}_Ts.png'
                )
                self._relabel_axes(diagram, 'Ts')
                diagram.fig.savefig(filepath_pdf, bbox_inches='tight')
                diagram.fig.savefig(filepath_png, bbox_inches='tight')
                diagram.fig.clf()  # Clear figure
                
                return filepath_png
                
            elif self.nr_refrigs == 2:
                # Zweistoff-System (z.B. Cascade)
                xmin1, xmax1 = self.calc_limits(
                    wf=self.hp.wf1, prop='s', padding_rel=0.35
                )
                ymin1, ymax1 = self.calc_limits(
                    wf=self.hp.wf1, prop='T', padding_rel=0.25
                )
                
                xmin2, xmax2 = self.calc_limits(
                    wf=self.hp.wf2, prop='s', padding_rel=0.35
                )
                ymin2, ymax2 = self.calc_limits(
                    wf=self.hp.wf2, prop='T', padding_rel=0.25
                )
                
                diagram1, diagram2 = self.hp.generate_state_diagram(
                    diagram_type='Ts',
                    figsize=figsize,
                    xlims=((xmin1, xmax1), (xmin2, xmax2)),
                    ylims=((ymin1, ymax1), (ymin2, ymax2)),
                    style=self.style,
                    return_diagram=True,
                    display_info=False,
                    savefig=False,
                    open_file=False
                )
                
                # Speichere beide Diagramme als PDF und PNG
                filepath1_pdf = os.path.join(
                    self.results_dir, f'{scenario_name}_Ts_cycle1.pdf'
                )
                filepath1_png = os.path.join(
                    self.results_dir, f'{scenario_name}_Ts_cycle1.png'
                )
                filepath2_pdf = os.path.join(
                    self.results_dir, f'{scenario_name}_Ts_cycle2.pdf'
                )
                filepath2_png = os.path.join(
                    self.results_dir, f'{scenario_name}_Ts_cycle2.png'
                )
                
                self._relabel_axes(diagram1, 'Ts')
                self._relabel_axes(diagram2, 'Ts')
                diagram1.fig.savefig(filepath1_pdf, bbox_inches='tight')
                diagram1.fig.savefig(filepath1_png, bbox_inches='tight')
                diagram2.fig.savefig(filepath2_pdf, bbox_inches='tight')
                diagram2.fig.savefig(filepath2_png, bbox_inches='tight')
                diagram1.fig.clf()
                diagram2.fig.clf()
                
                return [filepath1_png, filepath2_png]
        
        except Exception as e:
            print(f"    ⚠ Error in T-s diagram: {e}")
            return None

    def _heatpumps_models_input_dir(self) -> str:
        """
        Resolve the 'models/input' directory of the *installed* heatpumps
        package.

        Strategy: use inspect.getfile() on the concrete hp class — self.hp is
        always an instance of a class defined inside the real heatpumps.models
        package, so its source file is guaranteed to sit inside
        .../heatpumps/models/.  This works even when a local project folder
        named 'heatpumps/' shadows the installed package on sys.path (causing
        heatpumps.__file__ to be None for a namespace package).
        """
        import inspect
        # self.hp is a HeatPumpCascade2IHX / HeatPumpSimple / ... instance
        hp_file   = inspect.getfile(type(self.hp))          # …/models/HeatPumpXxx.py
        models_dir = os.path.dirname(os.path.abspath(hp_file))  # …/models/
        candidate  = os.path.join(models_dir, 'input')
        if os.path.isdir(candidate):
            return candidate
        # Fallback for unusual layouts: walk up two levels from models/
        for steps in (2, 3):
            parent = models_dir
            for _ in range(steps):
                parent = os.path.dirname(parent)
            candidate = os.path.join(parent, 'models', 'input')
            if os.path.isdir(candidate):
                return candidate
        raise FileNotFoundError(
            f"heatpumps models/input not found relative to {hp_file!r}"
        )

    def _build_fluid_diagram(self, refrig: str, diagram_type: str) -> FluidPropertyDiagram:
        """
        Load or compute a FluidPropertyDiagram for *refrig*.
        Uses the heatpumps library config for isoline ranges where available,
        falls back to MISC otherwise.  Caches the computed diagram as JSON
        next to the config file when writable.
        """
        input_dir = self._heatpumps_models_input_dir()

        # ── state_diagram_config ──────────────────────────────────────────────
        cfg_path = os.path.join(input_dir, 'state_diagram_config.json')
        with open(cfg_path, 'r', encoding='utf-8') as fh:
            config = json.load(fh)
        sp = config.get(refrig, config['MISC'])

        # ── axis variable mapping ─────────────────────────────────────────────
        if diagram_type == 'logph':
            var = {'x': 'h', 'y': 'p', 'isolines': ('T', 's')}
        else:
            var = {'x': 's', 'y': 'T', 'isolines': ('h', 'p')}

        # ── try loading cached JSON ───────────────────────────────────────────
        diagrams_dir = os.path.join(input_dir, 'diagrams')
        cache_path   = os.path.join(diagrams_dir, f'{refrig}.json')

        if os.path.isfile(cache_path):
            return FluidPropertyDiagram.from_json(cache_path)

        # ── compute from scratch ──────────────────────────────────────────────
        diag = FluidPropertyDiagram(refrig)
        diag.set_unit_system(T='°C', p='bar', h='kJ/kg')

        iso1_key, iso2_key = var['isolines']
        diag.set_isolines(**{
            iso1_key: np.arange(
                sp[iso1_key]['isorange_low'],
                sp[iso1_key]['isorange_high'],
                sp[iso1_key]['isorange_step'],
            ),
            iso2_key: np.arange(
                sp[iso2_key]['isorange_low'],
                sp[iso2_key]['isorange_high'],
                sp[iso2_key]['isorange_step'],
            ),
        })
        diag.calc_isolines()

        try:
            os.makedirs(diagrams_dir, exist_ok=True)
            diag.to_json(cache_path)
        except Exception:
            pass

        return diag

    def generate_all_diagrams_for_scenarios(self, scenarios_dict, figsize=(12, 7.5)):
        """
        Generiert Diagramme für mehrere Szenarien.
        
        Parameters
        ----------
        scenarios_dict : dict
            Dictionary mit Szenario-Namen als Keys und Heat Pump Objekten als Values
        figsize : tuple
            Größe der Figur
        
        Returns
        -------
        dict
            Dictionary mit allen generierten Diagrammpfaden
        """
        all_diagrams = {}
        
        for scenario_name, hp_obj in scenarios_dict.items():
            # Erstelle neuen Generator für jedes Szenario
            generator = StateDiagramGenerator(
                hp_obj, 
                results_dir=self.results_dir, 
                style=self.style
            )
            
            # Generiere Diagramme
            diagrams = generator.generate_diagrams(
                scenario_name=scenario_name, 
                figsize=figsize
            )
            
            if diagrams:
                all_diagrams[scenario_name] = diagrams
        
        return all_diagrams