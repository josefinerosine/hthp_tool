"""
Costsabschätzung für Dampferzeugungssysteme
=============================================

Berechnung von Betriebskosten basierend auf Stromverbrauch und Länder-spezifischen
Strompreisen sowie CO2 Emissions.

Datenquellen:
-------------
1. STROMPREISE (2024):
   - Eurostat: Energy price statistics (2024)
   - IEA: Electricity Market Report 2024
   - National energy statistics

2. CO2-FAKTOREN STROMMIX:
   - IPCC: 2019 Refinement to the 2006 IPCC Guidelines
   - IEA: CO2 Emissions from Fuel Combustion 2023
   - Umweltbundesamt: CO2-Emissionsfaktoren (2024)

3. INVESTITIONSKOSTEN:
   - VDI 2067: Economic efficiency of building installations (2012)
   - Quaschning, V. (2019): Regenerative Energiesysteme, Hanser Verlag
   
Wissenschaftliche Referenzen:
------------------------------
- Bejan, A., et al. (1996): "Thermal Design and Optimization"
  Chapter 9: Economic Analysis
  
- Peters, M.S., Timmerhaus, K.D., West, R.E. (2003):
  "Plant Design and Economics for Chemical Engineers"
  5th Edition, McGraw-Hill
  
- VDI 2067 Blatt 1 (2012):
  "Wirtschaftlichkeit gebäudetechnischer Anlagen -
   Grundlagen und Costsberechnung"
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json


class CostEstimator:
    """
    Klasse zur Costsabschätzung von Dampferzeugungssystemen.
    """
    
    # Strompreise [EUR/kWh] - Industriekunden (Stand 2024)
    # Source: Eurostat, IEA World Energy Outlook 2023
    ELECTRICITY_PRICES = {
        'Germany': {
            'price_EUR_per_kWh': 0.18,  # Industriestrom inkl. Steuern/Abgaben
            'source': 'Eurostat (2024), BDEW Strompreisanalyse',
            'note': 'Mittelwert für Industriekunden 2024'
        },
        'Norway': {
            'price_EUR_per_kWh': 0.08,  # Niedrige Preise durch Wasserkraft
            'source': 'Statistics Norway (SSB), NVE Energy Facts 2024',
            'note': 'Industriestrom, niedriger durch hohen Wasserkraftanteil'
        },
        'France': {
            'price_EUR_per_kWh': 0.12,
            'source': 'RTE France, Eurostat (2024)',
            'note': 'Moderate Preise durch Kernkraft'
        },
        'Denmark': {
            'price_EUR_per_kWh': 0.15,
            'source': 'Energinet, Danish Energy Agency (2024)',
            'note': 'Hoher Anteil erneuerbarer Energien'
        },
        'Poland': {
            'price_EUR_per_kWh': 0.14,
            'source': 'Eurostat (2024)',
            'note': 'Kohlebasierte Stromerzeugung'
        },
        'Sweden': {
            'price_EUR_per_kWh': 0.09,
            'source': 'Statistics Sweden, Eurostat (2024)',
            'note': 'Mix aus Wasser-, Kern- und Windkraft'
        },
        'Netherlands': {
            'price_EUR_per_kWh': 0.16,
            'source': 'CBS Netherlands, Eurostat (2024)',
            'note': 'Gas- und erneuerbare Energien'
        },
        'Austria': {
            'price_EUR_per_kWh': 0.14,
            'source': 'E-Control Austria, Eurostat (2024)',
            'note': 'Hoher Wasserkraftanteil'
        },
        'Switzerland': {
            'price_EUR_per_kWh': 0.10,
            'source': 'BFE Switzerland (2024)',
            'note': 'Wasserkraft und Kernkraft'
        }
    }
    
    # CO2-Emissionsfaktoren Strommix [kg CO2/kWh]
    # Source: IEA CO2 Emissions from Fuel Combustion 2023,
    #         Umweltbundesamt Deutschland (2024)
    CO2_FACTORS = {
        'Germany': {
            'kg_CO2_per_kWh': 0.420,  # Rückläufig durch Kohleausstieg
            'source': 'Umweltbundesamt Deutschland (2024)',
            'year': 2024,
            'note': 'Deutscher Strommix mit erneuerbaren Energien'
        },
        'Norway': {
            'kg_CO2_per_kWh': 0.015,  # Sehr niedrig durch Wasserkraft
            'source': 'NVE Norwegian Energy Statistics (2024)',
            'year': 2024,
            'note': '98% Wasserkraft'
        },
        'France': {
            'kg_CO2_per_kWh': 0.050,  # Niedrig durch Kernkraft
            'source': 'RTE France, ADEME (2024)',
            'year': 2024,
            'note': '~70% Kernkraft, Rest erneuerbare'
        },
        'Denmark': {
            'kg_CO2_per_kWh': 0.150,
            'source': 'Danish Energy Agency (2024)',
            'year': 2024,
            'note': 'Hoher Windkraftanteil'
        },
        'Poland': {
            'kg_CO2_per_kWh': 0.700,  # Hoch durch Kohle
            'source': 'IEA, Polish Energy Statistics (2024)',
            'year': 2024,
            'note': 'Kohledominierter Strommix'
        },
        'Sweden': {
            'kg_CO2_per_kWh': 0.020,
            'source': 'Swedish Energy Agency (2024)',
            'year': 2024,
            'note': 'Wasser-, Kern- und Windkraft'
        },
        'Netherlands': {
            'kg_CO2_per_kWh': 0.350,
            'source': 'CBS Netherlands (2024)',
            'year': 2024,
            'note': 'Gas und erneuerbare Energien'
        },
        'Austria': {
            'kg_CO2_per_kWh': 0.100,
            'source': 'Umweltbundesamt Austria (2024)',
            'year': 2024,
            'note': 'Hoher Wasserkraftanteil'
        },
        'Switzerland': {
            'kg_CO2_per_kWh': 0.030,
            'source': 'BFE Switzerland (2024)',
            'year': 2024,
            'note': 'Wasser- und Kernkraft'
        }
    }
    
    # Typische Investitionskosten [EUR] - Grobe Richtwerte
    # Source: VDI 2067 (2012), Quaschning (2019)
    # Anmerkung: Diese sind sehr variabel und abhängig von Größe, 
    # Standort und spezifischen Anforderungen
    INVESTMENT_COSTS = {
        'Electric Boiler': {
            'specific_cost_EUR_per_kW': 150,  # EUR/kW thermisch
            'lifetime_years': 20,
            'maintenance_percent_per_year': 2.0,
            'source': 'VDI 2067 (2012), Industriekesselhersteller',
            'note': 'Elektrische Dampferzeuger, einfacher Aufbau'
        },
        'Heat Pump': {
            'specific_cost_EUR_per_kW': 800,  # EUR/kW thermisch
            'lifetime_years': 20,
            'maintenance_percent_per_year': 3.5,
            'source': 'VDI 4640 (2019), BWP Marktdaten',
            'note': 'Hochtemperatur-Wärmepumpe, komplex'
        },
        'MVR': {
            'specific_cost_EUR_per_kW': 600,  # EUR/kW thermisch
            'lifetime_years': 20,
            'maintenance_percent_per_year': 4.0,
            'source': 'GEA Process Engineering (2023), VDI',
            'note': 'Mechanical Vapor Recompression'
        }
    }
    
    def __init__(self, country='Germany', operating_hours_per_year=8000):
        """
        Initialisiert den Costsrechner.
        
        Parameters
        ----------
        country : str
            Land für Strompreis und CO2-Faktor
        operating_hours_per_year : float
            Betriebsstunden pro Jahr (default: 8000h = ~91% Auslastung)
        """
        if country not in self.ELECTRICITY_PRICES:
            available = ', '.join(self.ELECTRICITY_PRICES.keys())
            raise ValueError(
                f"Land '{country}' nicht verfügbar. "
                f"Available countries: {available}"
            )
        
        self.country = country
        self.operating_hours = operating_hours_per_year
        self.electricity_price = self.ELECTRICITY_PRICES[country]['price_EUR_per_kWh']
        self.co2_factor = self.CO2_FACTORS[country]['kg_CO2_per_kWh']
    
    def calculate_operating_costs(
        self,
        P_electric_kW,
        technology='Unknown',
        include_maintenance=False,
        Q_thermal_kW=None
    ):
        """
        Berechnet jährliche Betriebskosten.
        
        Parameters
        ----------
        P_electric_kW : float
            Elektrische Leistung [kW]
        technology : str
            Technologie-Name
        include_maintenance : bool
            Wartungskosten einbeziehen (benötigt Q_thermal_kW)
        Q_thermal_kW : float
            Thermische Leistung für Investment estimation [kW]
        
        Returns
        -------
        dict
            Costsaufstellung
        """
        # Jährlicher Stromverbrauch [kWh/a]
        annual_consumption_kWh = P_electric_kW * self.operating_hours
        
        # Jährliche Stromkosten [EUR/a]
        annual_electricity_costs = annual_consumption_kWh * self.electricity_price
        
        # CO2 Emissions [t CO2/a]
        annual_co2_emissions = annual_consumption_kWh * self.co2_factor / 1000
        
        # Spezifische Costs [EUR/MWh thermisch]
        specific_cost_per_MWh = None
        if Q_thermal_kW is not None and Q_thermal_kW > 0:
            annual_thermal_MWh = Q_thermal_kW * self.operating_hours / 1000
            specific_cost_per_MWh = annual_electricity_costs / annual_thermal_MWh
        
        results = {
            'country': self.country,
            'technology': technology,
            'electricity_price_EUR_per_kWh': self.electricity_price,
            'co2_factor_kg_per_kWh': self.co2_factor,
            'operating_hours_per_year': self.operating_hours,
            'consumption': {
                'P_electric_kW': P_electric_kW,
                'annual_consumption_kWh': annual_consumption_kWh,
                'annual_consumption_MWh': annual_consumption_kWh / 1000
            },
            'costs': {
                'annual_electricity_EUR': annual_electricity_costs,
                'monthly_electricity_EUR': annual_electricity_costs / 12,
                'specific_EUR_per_MWh': specific_cost_per_MWh
            },
            'emissions': {
                'annual_CO2_tons': annual_co2_emissions,
                'specific_kg_CO2_per_MWh_thermal': None
            }
        }
        
        # Wartungskosten (optional)
        if include_maintenance and Q_thermal_kW is not None:
            if technology in self.INVESTMENT_COSTS:
                invest_data = self.INVESTMENT_COSTS[technology]
                investment_EUR = Q_thermal_kW * invest_data['specific_cost_EUR_per_kW']
                annual_maintenance = investment_EUR * invest_data['maintenance_percent_per_year'] / 100
                
                results['costs']['annual_maintenance_EUR'] = annual_maintenance
                results['costs']['annual_total_EUR'] = annual_electricity_costs + annual_maintenance
                results['investment'] = {
                    'estimated_investment_EUR': investment_EUR,
                    'specific_EUR_per_kW': invest_data['specific_cost_EUR_per_kW'],
                    'lifetime_years': invest_data['lifetime_years'],
                    'source': invest_data['source']
                }
        
        # Spezifische Emissionen
        if Q_thermal_kW is not None and Q_thermal_kW > 0:
            annual_thermal_MWh = Q_thermal_kW * self.operating_hours / 1000
            results['emissions']['specific_kg_CO2_per_MWh_thermal'] = (
                annual_co2_emissions * 1000 / annual_thermal_MWh
            )
        
        return results
    
    def compare_technologies(self, technology_results):
        """
        Vergleicht mehrere Technologien bezüglich Costs und Emissionen.
        
        Parameters
        ----------
        technology_results : dict
            Dictionary mit Technologie-Namen und deren Leistungsdaten
            Format: {'Tech1': {'P_electric_kW': ..., 'Q_thermal_kW': ...}}
        
        Returns
        -------
        pd.DataFrame
            Vergleichstabelle
        """
        comparison_data = []
        
        for tech_name, tech_data in technology_results.items():
            costs = self.calculate_operating_costs(
                P_electric_kW=tech_data['P_electric_kW'],
                technology=tech_name,
                include_maintenance=False,
                Q_thermal_kW=tech_data.get('Q_thermal_kW')
            )
            
            comparison_data.append({
                'Technologie': tech_name,
                'Elektrische Leistung [kW]': tech_data['P_electric_kW'],
                'yearssverbrauch [MWh/a]': costs['consumption']['annual_consumption_MWh'],
                'Stromkosten [EUR/a]': costs['costs']['annual_electricity_EUR'],
                'Stromkosten [EUR/Monat]': costs['costs']['monthly_electricity_EUR'],
                'CO2 Emissions [t/a]': costs['emissions']['annual_CO2_tons'],
                'Spez. Costs [EUR/MWh_th]': costs['costs'].get('specific_EUR_per_MWh', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Relative Werte zur Baseline (erste Technologie)
        if len(df) > 0:
            baseline_cost = df.iloc[0]['Stromkosten [EUR/a]']
            baseline_co2 = df.iloc[0]['CO2 Emissions [t/a]']
            
            df['Relative Costs [%]'] = (df['Stromkosten [EUR/a]'] / baseline_cost * 100).round(1)
            df['Relative CO2 [%]'] = (df['CO2 Emissions [t/a]'] / baseline_co2 * 100).round(1)
            df['Costsersparnis [EUR/a]'] = baseline_cost - df['Stromkosten [EUR/a]']
            df['CO2-Reduktion [t/a]'] = baseline_co2 - df['CO2 Emissions [t/a]']
        
        return df
    
    def calculate_payback_period(
        self,
        baseline_costs_EUR_per_year,
        alternative_costs_EUR_per_year,
        additional_investment_EUR
    ):
        """
        Berechnet die Amortisationszeit (Simple Payback Period).
        
        Source: VDI 2067 (2012), Peters et al. (2003)
        
        Parameters
        ----------
        baseline_costs_EUR_per_year : float
            Jährliche Costs der Baseline-Technologie
        alternative_costs_EUR_per_year : float
            Jährliche Costs der alternativen Technologie
        additional_investment_EUR : float
            Mehrinvestition gegenüber Baseline
        
        Returns
        -------
        dict
            Amortisationsberechnung
        """
        annual_savings = baseline_costs_EUR_per_year - alternative_costs_EUR_per_year
        
        if annual_savings <= 0:
            return {
                'payback_period_years': np.inf,
                'annual_savings_EUR': annual_savings,
                'note': 'Keine Costsersparnis - keine Amortisation möglich'
            }
        
        payback_years = additional_investment_EUR / annual_savings
        
        return {
            'payback_period_years': payback_years,
            'payback_period_months': payback_years * 12,
            'annual_savings_EUR': annual_savings,
            'additional_investment_EUR': additional_investment_EUR,
            'cumulative_savings_after_10years_EUR': annual_savings * 10 - additional_investment_EUR
        }
    
    def print_cost_summary(self, results):
        """Gibt eine formatierte Costsübersicht aus."""
        print('\n' + '='*80)
        print('COST ESTIMATION - SUMMARY')
        print('='*80)
        
        print(f"\nCountry: {results['country']}")
        print(f"Technology: {results['technology']}")
        print(f"Electricity price: {results['electricity_price_EUR_per_kWh']:.4f} EUR/kWh")
        print(f"CO2 factor: {results['co2_factor_kg_per_kWh']:.3f} kg CO2/kWh")
        print(f"Operating hours: {results['operating_hours_per_year']:.0f} h/a")
        
        print(f"\n{'Consumption':.<40} {''}")
        print(f"  Electrical power: {results['consumption']['P_electric_kW']:.1f} kW")
        print(f"  Annual consumption: {results['consumption']['annual_consumption_MWh']:.1f} MWh/a")
        
        print(f"\n{'Costs':.<40} {''}")
        print(f"  Annual electricity costs: {results['costs']['annual_electricity_EUR']:,.0f} EUR/a")
        print(f"  Monthly electricity costs: {results['costs']['monthly_electricity_EUR']:,.0f} EUR/Monat")
        if results['costs'].get('specific_EUR_per_MWh'):
            print(f"  Spezifische Costs: {results['costs']['specific_EUR_per_MWh']:.2f} EUR/MWh_th")
        
        if 'annual_maintenance_EUR' in results['costs']:
            print(f"  Annual maintenance: {results['costs']['annual_maintenance_EUR']:,.0f} EUR/a")
            print(f"  TOTAL (incl. maintenance): {results['costs']['annual_total_EUR']:,.0f} EUR/a")
        
        print(f"\n{'CO2 Emissions':.<40} {''}")
        print(f"  Annual: {results['emissions']['annual_CO2_tons']:.1f} t CO2/a")
        if results['emissions'].get('specific_kg_CO2_per_MWh_thermal'):
            print(f"  Specific: {results['emissions']['specific_kg_CO2_per_MWh_thermal']:.1f} kg CO2/MWh_th")
        
        if 'investment' in results:
            print(f"\n{'Investment estimation':.<40} {''}")
            print(f"  Estimated investment: {results['investment']['estimated_investment_EUR']:,.0f} EUR")
            print(f"  Specific: {results['investment']['specific_EUR_per_kW']:.0f} EUR/kW")
            print(f"  Lifetime: {results['investment']['lifetime_years']} years")
        
        print('='*80 + '\n')
    
    def export_results(self, results, filepath='cost_estimation_results.json'):
        """Exportiert Ergebnisse als JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"✓ Costsergebnisse exportiert: {filepath}")


# Hilfsfunktionen
def get_available_countries():
    """Gibt Liste verfügbarer Länder zurück."""
    return list(CostEstimator.ELECTRICITY_PRICES.keys())


def quick_cost_comparison(technology_data, country='Germany', operating_hours=8000):
    """
    Schneller Costsvergleich mehrerer Technologien.
    
    Parameters
    ----------
    technology_data : dict
        {'Tech1': {'P_electric_kW': ..., 'Q_thermal_kW': ...}, ...}
    country : str
        Land
    operating_hours : float
        Betriebsstunden/Jahr
    
    Returns
    -------
    pd.DataFrame
        Vergleichstabelle
    """
    estimator = CostEstimator(country=country, operating_hours_per_year=operating_hours)
    return estimator.compare_technologies(technology_data)


# Beispiel
if __name__ == '__main__':
    print("Available countries:", get_available_countries())
    
    # Beispiel-Technologien
    technologies = {
        'Electric Boiler': {
            'P_electric_kW': 500,
            'Q_thermal_kW': 495
        },
        'Heat Pump': {
            'P_electric_kW': 150,
            'Q_thermal_kW': 500
        },
        'MVR': {
            'P_electric_kW': 180,
            'Q_thermal_kW': 500
        }
    }
    
    # Vergleich für Deutschland
    print("\n" + "="*80)
    print("COST COMPARISON - GERMANY")
    print("="*80)
    df = quick_cost_comparison(technologies, country='Germany', operating_hours=8000)
    print(df.to_string(index=False))
    
    # Vergleich für Norwegen
    print("\n" + "="*80)
    print("COST COMPARISON - NORWAY")
    print("="*80)
    df = quick_cost_comparison(technologies, country='Norway', operating_hours=8000)
    print(df.to_string(index=False))