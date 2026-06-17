"""
Questionnaire Reader - Liest HTHP Questionnaire Excel-Datei aus
================================================================

Liest alle Parameter aus der HTHP_questionnaire.xlsx und gibt sie
strukturiert mit Beschreibung, Wert und Einheit zurück.

Version: 4.0
"""

import pandas as pd
import os
from typing import Any, Optional
from datetime import datetime


# ── Row-Index basiertes Mapping ──────────────────────────────────────────────
# Jeder Eintrag: key → (row_index, description, section)
ROW_MAPPING = {
    # 1. Project Data
    'company_name':           (1,  'Company / Name',                           '1. Project Data'),
    'contact_phone_email':    (2,  'Phone / Email',                            '1. Project Data'),
    'commissioning_date':     (3,  'Planned Commissioning Date',               '1. Project Data'),
    'contact_person':         (4,  'Contact Person',                           '1. Project Data'),
    'location_country':       (5,  'Plant Location (country)',                 '1. Project Data'),
    'location_city_street':   (6,  'Plant Location (city, street)',            '1. Project Data'),

    # 2. Heat Source
    'source_type':            (9,  'Heat Source Type',                         '2. Heat Source'),
    'source_heat_capacity':   (10, 'Available Heat Capacity',                  '2. Heat Source'),
    'source_temp_in':         (11, 'Source Inlet Temperature',                 '2. Heat Source'),
    'source_temp_out':        (12, 'Source Outlet Temperature (min.)',         '2. Heat Source'),
    'source_pressure':        (13, 'Source Side Pressure',                     '2. Heat Source'),
    'source_medium':          (14, 'Heat Transfer Medium Source',              '2. Heat Source'),
    'source_year_round':      (15, 'Year-Round Availability?',                 '2. Heat Source'),
    'source_seasonal_period': (16, 'Seasonal? From - To',                      '2. Heat Source'),

    # 3. Heat Consumer - General
    'application_type':       (19, 'Application Type',                         '3. Heat Consumer'),

    # 3a. Hot Water
    'hw_temp_inlet':          (22, 'Sink inlet / feed water Temperature',      '3a. Hot Water'),
    'hw_temp_outlet_required':(23, 'Required Outlet Temperature',              '3a. Hot Water'),
    'hw_temp_outlet_min':     (24, 'Minimum Outlet Temperature',               '3a. Hot Water'),
    'hw_mass_flow':           (25, 'Mass Flow',                                '3a. Hot Water'),
    'hw_heat_capacity':       (26, 'Required Heat Capacity',                   '3a. Hot Water'),
    'hw_operating_mode':      (27, 'Operating Mode',                           '3a. Hot Water'),
    'hw_operating_hours':     (28, 'Daily Operating Hours',                    '3a. Hot Water'),

    # 3b. Steam
    'steam_flow_config':      (31, 'System Configuration / Flow System',       '3b. Steam'),
    'steam_temp_inlet':       (32, 'Feed Water / Steam Temperature',           '3b. Steam'),
    'steam_pressure_inlet':   (33, 'Feed Water / Steam Pressure',              '3b. Steam'),
    'steam_mass_flow_inlet':  (34, 'Feed Water / Supply Mass Flow',            '3b. Steam'),
    'steam_pressure_outlet':  (35, 'Steam Target Pressure',                    '3b. Steam'),
    'steam_quality':          (36, 'Output Steam Quality',                     '3b. Steam'),
    'steam_temp_saturation':  (37, 'Saturation Temperature at Steam Pressure', '3b. Steam'),
    'steam_superheat':        (38, 'Required Superheat',                       '3b. Steam'),
    'steam_operating_mode':   (39, 'Steam Operating Mode',                     '3b. Steam'),
    'steam_operating_hours':  (40, 'Daily Operating Hours',                    '3b. Steam'),

    # 3c. Optional Additional Heat Consumer
    'add_hw_temp_inlet':          (43, 'Additional: Sink Inlet / Feed Water Temperature', '3c. Additional Consumer'),
    'add_hw_temp_outlet_required':(44, 'Additional: Required Outlet Temperature',         '3c. Additional Consumer'),
    'add_hw_temp_outlet_min':     (45, 'Additional: Minimum Outlet Temperature',          '3c. Additional Consumer'),
    'add_hw_heat_capacity':       (46, 'Additional: Required Heat Capacity',              '3c. Additional Consumer'),
    'add_hw_operating_mode':      (47, 'Additional: Operating Mode',                      '3c. Additional Consumer'),
    'add_hw_operating_hours':     (48, 'Additional: Daily Operating Hours',               '3c. Additional Consumer'),

    # 4. Infrastructure
    'electrical_power_supply': (51, 'Electrical Power Supply Available',       '4. Infrastructure'),
    'electrical_power_max':    (52, 'Available Power',                         '4. Infrastructure'),
    'cooling_water_available': (53, 'Cooling Water Available',                 '4. Infrastructure'),
    'cooling_water_temp':      (54, 'Cooling Water Temperature',               '4. Infrastructure'),
    'cooling_water_flow':      (55, 'Condenser Cooling Water Volume Flow',     '4. Infrastructure'),
    'water_hardness':          (56, 'Water Hardness',                          '4. Infrastructure'),

    # 5. Optimization
    'waste_heat_count':            (59, 'Number of Additional Waste Heat Flows',   '5. Optimization'),

    'waste_heat_1_variability':    (61, 'Waste Heat 1: Variability',               '5. Optimization'),
    'waste_heat_1_temp_supply':    (62, 'Waste Heat 1: Supply Temperature',        '5. Optimization'),
    'waste_heat_1_temp_outlet':    (63, 'Waste Heat 1: Outlet Temperature',        '5. Optimization'),
    'waste_heat_1_pressure':       (64, 'Waste Heat 1: Supply Pressure',           '5. Optimization'),
    'waste_heat_1_medium':         (65, 'Waste Heat 1: Medium',                    '5. Optimization'),
    'waste_heat_1_mass_flow':      (66, 'Waste Heat 1: Mass Flow',                 '5. Optimization'),

    'waste_heat_2_variability':    (68, 'Waste Heat 2: Variability',               '5. Optimization'),
    'waste_heat_2_temp_supply':    (69, 'Waste Heat 2: Supply Temperature',        '5. Optimization'),
    'waste_heat_2_temp_outlet':    (70, 'Waste Heat 2: Outlet Temperature',        '5. Optimization'),
    'waste_heat_2_pressure':       (71, 'Waste Heat 2: Supply Pressure',           '5. Optimization'),
    'waste_heat_2_medium':         (72, 'Waste Heat 2: Medium',                    '5. Optimization'),
    'waste_heat_2_mass_flow':      (73, 'Waste Heat 2: Mass Flow',                 '5. Optimization'),

    'waste_heat_3_variability':    (75, 'Waste Heat 3: Variability',               '5. Optimization'),
    'waste_heat_3_temp_supply':    (76, 'Waste Heat 3: Supply Temperature',        '5. Optimization'),
    'waste_heat_3_temp_outlet':    (77, 'Waste Heat 3: Outlet Temperature',        '5. Optimization'),
    'waste_heat_3_pressure':       (78, 'Waste Heat 3: Supply Pressure',           '5. Optimization'),
    'waste_heat_3_medium':         (79, 'Waste Heat 3: Medium',                    '5. Optimization'),
    'waste_heat_3_mass_flow':      (80, 'Waste Heat 3: Mass Flow',                 '5. Optimization'),

    'modulation_required':     (81, 'Modulation Required?',                    '5. Optimization'),
    'modulation_range':        (82, 'Modulation Range',                        '5. Optimization'),
    'storage_available':       (83, 'Storage Option Available?',               '5. Optimization'),
    'storage_volume':          (84, 'Storage Volume',                          '5. Optimization'),

    # 6. Location & Environment
    'installation_space':      (87, 'Available Installation Space',            '6. Location & Environment'),
    'room_temp_winter':        (88, 'Room Temperature Winter',                 '6. Location & Environment'),
    'room_temp_summer':        (89, 'Room Temperature Summer',                 '6. Location & Environment'),
    'air_humidity':            (90, 'Air Humidity',                            '6. Location & Environment'),
    'altitude':                (91, 'Altitude',                                '6. Location & Environment'),
    'noise_sensitivity':       (92, 'Vibration / Noise Sensitivity',           '6. Location & Environment'),

    # 7. Safety & Permissions
    'ped_required':            (95,  'Pressure Equipment Directive (PED)?',    '7. Safety & Permissions'),
    'ped_category':            (96,  'PED Category',                           '7. Safety & Permissions'),
    'atex_required':           (97,  'ATEX Requirements',                      '7. Safety & Permissions'),
    'atex_zone':               (98,  'ATEX Zone',                              '7. Safety & Permissions'),
    'safety_valves_required':  (99,  'Safety Valves Required?',                '7. Safety & Permissions'),
    'safety_valve_pressure':   (100, 'Max. Pressure (Safety Valve)',           '7. Safety & Permissions'),
    'sound_protection':        (101, 'Sound Protection Requirement',           '7. Safety & Permissions'),
    'sound_db_distance':       (102, 'dB(A) at Distance',                     '7. Safety & Permissions'),
    'other_permits':           (103, 'Other Permits / Standards',              '7. Safety & Permissions'),

    # 8. Economic Parameters
    'electricity_price':       (106, 'Electricity Price',                      '8. Economic Parameters'),
    'gas_price':               (107, 'Gas Price',                              '8. Economic Parameters'),
    'investment_budget':       (108, 'Available Investment Budget',            '8. Economic Parameters'),
    'payback_period':          (109, 'Required Payback Period',                '8. Economic Parameters'),
    'annual_operating_hours':  (110, 'Planned Annual Operating Hours',         '8. Economic Parameters'),
}


class QuestionnaireReader:
    """
    Liest HTHP Questionnaire Excel-Datei aus und speichert alle Parameter
    mit Beschreibung, Wert und Einheit.

    Verwendung:
        reader = QuestionnaireReader('path/to/HTHP_questionnaire.xlsx')
        params = reader.get_params()

        # Einzelner Parameter:
        print(params['source_temp_in'])
        # → {'description': 'Source Inlet Temperature',
        #    'value': 90.0,
        #    'unit': '°C',
        #    'section': '2. Heat Source'}
    """

    def __init__(self, excel_path: str):
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel-Datei nicht gefunden: {excel_path}")
        self.excel_path = excel_path
        self._params: dict | None = None
        self._raw_df: pd.DataFrame | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def get_params(self) -> dict:
        """
        Gibt alle Parameter als Dictionary zurück.

        Returns:
            {
                'key': {
                    'description': str,
                    'value': int | float | str | None,
                    'unit': str | None,
                    'section': str
                },
                ...
            }
        """
        if self._params is None:
            self._read()
        return self._params

    def get_value(self, key: str) -> Any:
        """Gibt nur den Wert eines Parameters zurück."""
        return self.get_params().get(key, {}).get('value')

    def get_application_case(self) -> str:
        """
        Bestimmt den Anwendungsfall basierend auf application_type.
        
        Returns:
            'hot_water' - für Heißwasseranwendungen
            'steam' - für Dampfanwendungen
            'unknown' - wenn nicht determiniert werden kann
        """
        app_type = self.get_value('application_type')
        
        if app_type is None:
            return 'unknown'
        
        app_type_lower = str(app_type).lower().strip()
        
        # Mapping verschiedener Schreibweisen
        if any(keyword in app_type_lower for keyword in ['hot water', 'hotwater', 'hw', 'heißwasser', 'warmwasser']):
            return 'hot_water'
        elif any(keyword in app_type_lower for keyword in ['steam', 'dampf', 'dampfereugung']):
            return 'steam'
        else:
            return 'unknown'

    def print_summary(self, skip_empty: bool = True):
        """Gibt alle Parameter strukturiert auf der Konsole aus."""
        params = self.get_params()
        current_section = None

        print("\n" + "=" * 80)
        print("QUESTIONNAIRE PARAMETER SUMMARY")
        print(f"File: {self.excel_path}")
        print("=" * 80)

        for key, info in params.items():
            section = info['section']
            if section != current_section:
                print(f"\n── {section} " + "─" * (60 - len(section)))
                current_section = section

            value = info['value']
            unit  = info['unit']

            if skip_empty and value is None:
                continue

            unit_str  = f" [{unit}]" if unit else ""
            value_str = str(value) if value is not None else "—"
            print(f"  {key:<35} {info['description']:<45} {value_str}{unit_str}")

        print("=" * 80 + "\n")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _read(self):
        """Liest die Excel-Datei und füllt self._params."""
        try:
            df = pd.read_excel(self.excel_path, sheet_name=0, header=None)
        except Exception as e:
            raise RuntimeError(f"Fehler beim Öffnen der Excel-Datei: {e}")

        self._raw_df = df
        self._params = {}

        for key, (row_idx, description, section) in ROW_MAPPING.items():
            if row_idx >= len(df):
                self._params[key] = {
                    'description': description,
                    'value': None,
                    'unit': None,
                    'section': section,
                }
                continue

            row = df.iloc[row_idx]

            raw_value = row[1] if len(row) > 1 else None
            raw_unit  = row[2] if len(row) > 2 else None

            value = self._convert_value(raw_value)
            unit  = self._clean_unit(raw_unit)

            self._params[key] = {
                'description': description,
                'value': value,
                'unit': unit,
                'section': section,
            }

    @staticmethod
    def _convert_value(raw: Any) -> int | float | str | None:
        """Konvertiert einen Rohwert aus Excel in den passenden Python-Typ."""
        # Leere / ungültige Werte
        if raw is None:
            return None
        if isinstance(raw, float) and pd.isna(raw):
            return None

        # Datetime direkt aus pandas
        if isinstance(raw, (pd.Timestamp, datetime)):
            return str(raw.date()) if hasattr(raw, 'date') else str(raw)

        value_str = str(raw).strip()

        # Platzhalter / nicht ausgefüllt
        invalid = {'', 'nan', 'select...', '?', '-', 'x', 'tbd', 'n/a', 'na',
                   'none', 'tbc', 'TBD', 'TBC', 'N/A', 'Select...'}
        if value_str in invalid or value_str.lower() in {v.lower() for v in invalid}:
            return None

        # Versuche numerische Konvertierung
        try:
            # Integer
            if '.' not in value_str and 'e' not in value_str.lower():
                return int(value_str)
            return float(value_str)
        except (ValueError, AttributeError):
            pass

        # String zurückgeben
        return value_str

    @staticmethod
    def _clean_unit(raw: Any) -> str | None:
        """Bereinigt die Einheit aus Spalte C."""
        if raw is None:
            return None
        if isinstance(raw, float) and pd.isna(raw):
            return None

        unit_str = str(raw).strip()
        if unit_str in ('', 'nan'):
            return None

        return unit_str


# ── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else 'hthp_project_final\HTHP_questionnaire.xlsx'

    reader = QuestionnaireReader(path)
    reader.print_summary(skip_empty=False)

    # Beispiel Einzelzugriff
    print("Beispiel-Einzelzugriff:")
    print(f"  source_temp_in  → {reader.get_value('source_temp_in')} °C")
    print(f"  application_type → {reader.get_value('application_type')}")
    print(f"  electricity_price → {reader.get_value('electricity_price')} €/kWh")