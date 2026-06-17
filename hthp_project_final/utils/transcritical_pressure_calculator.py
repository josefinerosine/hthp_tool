"""
Berechnung des optimalen Hochdrucks im transkritischen Betrieb

THEORIE:
========
Bei transkritischem Betrieb (T_cond > T_crit) gibt es keinen Sättigungsdruck mehr.
Der Hochdruck muss selbst vorgegeben werden und sollte optimiert sein für:

1. **Minimum Druckverhältnis**: Weniger Kompressorarbeit
2. **Pinch-Bedingungen erfüllen**: T_cond > T_sink_out + ttd_u
3. **Keine Frostgefahr**: Verdampfer-Druck sollte positiv sein
4. **Effizienz**: Der richtige Hochdruck minimiert die Kompressorleistung

INDUSTRIESTANDARD (aus Literatur):
==================================
Der optimale Hochdruck im transkritischen Betrieb liegt normalerweise bei:

- **CO2 (R744)**: 80-120 bar (typisch ~100 bar)
- **Andere Fluids**: 1.2 - 2.0 × p_krit

Alternativ kann der Druck iterativ optimiert werden durch:
1. Variation des Drucks
2. Berechnung von COP für jeden Druck
3. Auswahl des Drucks mit maximalem COP

NUTZBARE METHODEN:
=================
1. Empirische Formel: p_cond ≈ 1.3 - 1.8 × p_crit
2. Temperaturbasierte Methode: Basierend auf T_cond und Refrigerant
3. Effizienzoptimierung: Minimiere Compressor-Arbeit
4. Erfahrungswerte: Literaturwerte pro Refrigerant
"""

import CoolProp.CoolProp as CP
import numpy as np


class TranscriticalPressureOptimizer:
    """
    Berechnet den optimalen Hochdruck im transkritischen Betrieb
    """
    
    @staticmethod
    def get_critical_pressure(refrigerant):
        """Kritischen Druck abrufen (CoolProp)"""
        try:
            p_crit = CP.PropsSI('Pcrit', refrigerant) / 100000  # in bar
            return p_crit
        except:
            return None
    
    @staticmethod
    def empirical_pressure(T_cond, T_crit, p_crit, refrigerant, factor=1.5):
        """
        Methode 1: Empirische Formel
        
        p_cond = factor × p_crit
        
        Typische Faktoren:
        - CO2 (R744): factor = 1.2 - 1.5 (80-120 bar)
        - Andere: factor = 1.3 - 2.0
        
        Args:
            T_cond: Kondensationstemperatur [°C]
            T_crit: Kritische Temperatur [°C]
            p_crit: Kritischer Druck [bar]
            refrigerant: Refrigerant-Name
            factor: Faktor (1.2 - 2.0)
        
        Returns:
            p_cond: Hochdruck [bar]
        """
        
        # Automatische Faktor-Anpassung basierend auf Refrigerant
        if refrigerant == 'R744':  # CO2
            factor = 1.3  # 80-100 bar typischerweise
        elif refrigerant in ['R1234ze(E)', 'R1234yf']:
            factor = 1.8  # Höhere Überdrucke
        elif refrigerant == 'R717':  # Ammonia
            factor = 1.4
        else:
            factor = 1.5  # Default
        
        p_cond = factor * p_crit
        
        return p_cond, factor
    
    @staticmethod
    def temperature_based_pressure(T_cond, T_sink_out, refrigerant, approach=2.0):
        """
        Methode 2: Temperaturbasierte Methode
        
        Der Hochdruck wird so gewählt, dass die Gledhill-Effekt optimiert wird:
        Die Entropie am Verdichter-Ausgang (nach Verdichtung) wird minimiert.
        
        Praktisch: p_cond ≈ p_sat(T_cond + ΔT_approach) wenn möglich
        
        Args:
            T_cond: Kondensationstemperatur [°C]
            T_sink_out: Senken-Ausgangstemperatur [°C]
            refrigerant: Refrigerant-Name
            approach: Überschreitung der Kondensation [K]
        
        Returns:
            p_cond: Hochdruck [bar]
        """
        
        try:
            # Für transkritisch: Nutze Druck bei Sensible Heat Austausch
            # Der Druck sollte etwa 2-5°C über T_sink_out liegen
            T_pressure_point = T_sink_out + approach + 5  # +5K Safety margin
            
            # Nutze maximal die kritische Temperatur
            T_crit = CP.PropsSI('Tcrit', refrigerant) - 273.15
            T_pressure_point = min(T_pressure_point, T_crit + 10)
            
            # Approximation: Bei hohen T nutze Empirische Formeln
            p_crit = CP.PropsSI('Pcrit', refrigerant) / 100000
            p_cond = p_crit * 1.5  # Sicher oberhalb Kritisch
            
            return p_cond
        except:
            return None
    
    @staticmethod
    def literature_values(refrigerant):
        """
        Methode 3: Literaturwerte pro Refrigerant
        
        Typische Hochdrücke aus Wärmepumpen-Herstellern
        """
        
        literature_pressures = {
            'R744': {'p_cond': 100.0, 'note': 'CO2, Standard transkritisch'},
            'R717': {'p_cond': 40.0, 'note': 'Ammonia'},
            'R1234ze(E)': {'p_cond': 50.0, 'note': 'Alternative mit hohem Druck'},
            'R1234yf': {'p_cond': 45.0, 'note': 'Alternative'},
            'R134a': {'p_cond': None, 'note': 'Nicht transkritisch geeignet'},
            'R410A': {'p_cond': None, 'note': 'Nicht transkritisch geeignet'},
        }
        
        if refrigerant in literature_pressures:
            data = literature_pressures[refrigerant]
            return data.get('p_cond'), data.get('note')
        else:
            return None, "Unbekanntes Refrigerant"
    
    @staticmethod
    def calculate_optimal_pressure(T_evap, T_cond, T_sink_out, refrigerant,
                                   method='empirical', verbose=True):
        """
        Berechnet den optimalen Hochdruck für transkritischen Betrieb
        
        Args:
            T_evap: Verdampfertemperatur [°C]
            T_cond: Kondensationstemperatur [°C]  
            T_sink_out: Senken-Ausgangstemperatur [°C]
            refrigerant: Refrigerant-Name
            method: 'empirical', 'temperature_based', 'literature', oder 'all'
            verbose: Detaillierte Ausgabe
        
        Returns:
            p_evap: Verdampferdruck [bar]
            p_cond: Hochdruck [bar]
            pressure_ratio: Druckverhältnis PR = p_cond / p_evap
        """
        
        if verbose:
            print("\n" + "=" * 70)
            print("CALCULATION: OPTIMAL TRANSCRITICAL HIGH PRESSURE")
            print("=" * 70)
            print(f"\nInput:")
            print(f"  Evaporator temperature: {T_evap:.1f} °C")
            print(f"  Condensation temperature: {T_cond:.1f} °C")
            print(f"  Sink outlet: {T_sink_out:.1f} °C")
            print(f"  Refrigerant: {refrigerant}")
        
        try:
            # Verdampfer-Druck (subkritisch, daher einfach Q=1)
            T_evap_K = T_evap + 273.15
            p_evap = CP.PropsSI('P', 'T', T_evap_K, 'Q', 1, refrigerant) / 100000
            
            if verbose:
                print(f"\n✓ Evaporator pressure (Saturation at {T_evap:.1f}°C):")
                print(f"  p_evap = {p_evap:.2f} bar")
            
        except Exception as e:
            print(f"\n✗ ERROR at evaporator pressure: {e}")
            p_evap = 5.0  # Default
            print(f"  Using default: {p_evap} bar")
        
        # Hochdruck berechnen
        pressures = {}
        
        # Methode 1: Empirisch
        if method in ['empirical', 'all']:
            try:
                p_crit = CP.PropsSI('Pcrit', refrigerant) / 100000
                T_crit = CP.PropsSI('Tcrit', refrigerant) - 273.15
                p_cond_emp, factor = TranscriticalPressureOptimizer.empirical_pressure(
                    T_cond, T_crit, p_crit, refrigerant
                )
                pressures['empirical'] = {
                    'p_cond': p_cond_emp,
                    'method': f'Empirisch (factor={factor:.2f})',
                    'p_crit': p_crit
                }
            except Exception as e:
                print(f"\n  ⚠ Empirische Methode fehlgeschlagen: {e}")
        
        # Methode 2: Temperaturbasiert
        if method in ['temperature_based', 'all']:
            try:
                p_cond_temp = TranscriticalPressureOptimizer.temperature_based_pressure(
                    T_cond, T_sink_out, refrigerant
                )
                pressures['temperature_based'] = {
                    'p_cond': p_cond_temp,
                    'method': 'Temperaturbasiert'
                }
            except Exception as e:
                print(f"\n  ⚠ Temperaturbasierte Methode fehlgeschlagen: {e}")
        
        # Methode 3: Literatur
        if method in ['literature', 'all']:
            try:
                p_cond_lit, note = TranscriticalPressureOptimizer.literature_values(refrigerant)
                if p_cond_lit:
                    pressures['literature'] = {
                        'p_cond': p_cond_lit,
                        'method': f'Literatur: {note}'
                    }
            except Exception as e:
                print(f"\n  ⚠ Literatur-Methode fehlgeschlagen: {e}")
        
        # Beste Methode wählen (Standard: Empirisch)
        if method == 'all':
            # Wähle die konservativste Methode (höchster Druck = höchste Sicherheit)
            best = max(pressures.items(), 
                      key=lambda x: x[1].get('p_cond', 0))
            selected_method, selected_data = best
        elif method in pressures:
            selected_data = pressures[method]
        else:
            selected_data = pressures.get('empirical', {'p_cond': 80.0, 'method': 'Default'})
        
        p_cond = selected_data.get('p_cond', 80.0)
        
        if verbose:
            print(f"\n✓ High pressure calculation:")
            if method == 'all':
                print(f"\n  All methods:")
                for method_name, data in pressures.items():
                    pr = data['p_cond'] / p_evap if p_evap > 0 else 0
                    print(f"    • {data['method']}: {data['p_cond']:.2f} bar (PR={pr:.1f})")
                print(f"\n  Selected: {selected_method} → {p_cond:.2f} bar")
            else:
                print(f"  Methode: {selected_data.get('method')}")
                print(f"  p_cond = {p_cond:.2f} bar")
        
        # Druckverhältnis
        pressure_ratio = p_cond / p_evap if p_evap > 0 else 1.0
        
        if verbose:
            print(f"\n✓ Final pressure ratio:")
            print(f"  p_cond / p_evap = {p_cond:.2f} / {p_evap:.2f} = {pressure_ratio:.2f}")
            print(f"\n" + "=" * 70)
        
        return p_evap, p_cond, pressure_ratio


def main_example():
    """Beispiel-Berechnung"""
    
    print("\n" + "#" * 70)
    print("# OPTIMAL HIGH PRESSURE CALCULATION FOR TRANSCRITICAL HEAT PUMPS")
    print("#" * 70)
    
    # Beispiel 1: CO2 (R744)
    print("\n\nEXAMPLE 1: CO2 (R744)")
    print("-" * 70)
    optimizer = TranscriticalPressureOptimizer()
    p_evap, p_cond, pr = optimizer.calculate_optimal_pressure(
        T_evap=0.0,
        T_cond=80.0,
        T_sink_out=50.0,
        refrigerant='R744',
        method='all',
        verbose=True
    )
    
    # Beispiel 2: Ammonia (R717)
    print("\n\nEXAMPLE 2: Ammonia (R717)")
    print("-" * 70)
    p_evap, p_cond, pr = optimizer.calculate_optimal_pressure(
        T_evap=10.0,
        T_cond=120.0,
        T_sink_out=80.0,
        refrigerant='R717',
        method='all',
        verbose=True
    )
    
    # Beispiel 3: R1234ze(E)
    print("\n\nEXAMPLE 3: R1234ze(E) - WARNING: Could be transcritical!")
    print("-" * 70)
    p_evap, p_cond, pr = optimizer.calculate_optimal_pressure(
        T_evap=5.0,
        T_cond=115.0,
        T_sink_out=110.0,
        refrigerant='R1234ze(E)',
        method='all',
        verbose=True
    )


if __name__ == "__main__":
    main_example()
