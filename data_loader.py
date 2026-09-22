"""
Agentic Industrial Supply Chain Copilot: Data Generator & Loader
Generates realistic 100K+ multi-echelon industrial equipment shipment records,
macro commodity indices (Copper, Steel, Freight), and corporate 10-K report excerpts.
"""

import os
import csv
import random
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(DATA_DIR, "reports_10k")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

COMPONENTS = [
    {"name": "Power Transformer (500kV)", "lead_base_days": 180, "criticality": 0.95, "copper_intensity_kg": 4500},
    {"name": "Wind Turbine Blade (80m)", "lead_base_days": 90, "criticality": 0.85, "copper_intensity_kg": 300},
    {"name": "Solar Utility Inverter", "lead_base_days": 45, "criticality": 0.70, "copper_intensity_kg": 600},
    {"name": "Gas Turbine Compressor Stator", "lead_base_days": 150, "criticality": 0.90, "copper_intensity_kg": 1200},
    {"name": "Subsea HVDC Cable (per km)", "lead_base_days": 120, "criticality": 0.88, "copper_intensity_kg": 8500},
    {"name": "Grid Battery Enclosure (2MWh)", "lead_base_days": 75, "criticality": 0.80, "copper_intensity_kg": 2100}
]

SUPPLIERS = [
    {"id": "SUP-001", "name": "Nordic Turbine Forgings", "country": "Denmark", "base_reliability": 0.94},
    {"id": "SUP-002", "name": "Bavaria Precision Heavy Steels", "country": "Germany", "base_reliability": 0.91},
    {"id": "SUP-003", "name": "East Asia Semi & Inverter Corp", "country": "Taiwan", "base_reliability": 0.86},
    {"id": "SUP-004", "name": "Suzhou Rare Earth Magnetics", "country": "China", "base_reliability": 0.79},
    {"id": "SUP-005", "name": "Texas Heavy Power Dynamics", "country": "USA", "base_reliability": 0.89},
    {"id": "SUP-006", "name": "Osaka Specialized Rotor Alloys", "country": "Japan", "base_reliability": 0.95},
    {"id": "SUP-007", "name": "Gujarat High Voltage Conductors", "country": "India", "base_reliability": 0.84},
    {"id": "SUP-008", "name": "Goyang Smart Power Transformers", "country": "South Korea", "base_reliability": 0.88}
]


def generate_macro_indices_csv():
    """Generates 3 years of weekly historical commodity & freight price indices."""
    filepath = os.path.join(DATA_DIR, "macro_commodity_indices.csv")
    logging.info("Generating weekly macro commodity & shipping indices...")
    
    start_date = datetime(2021, 1, 1)
    weeks = 180
    records = []

    copper_price = 4.10      # $/lb
    steel_price = 850.0      # $/metric ton
    freight_index = 2200.0   # Baltic Dry Index benchmark

    random.seed(42)
    for w in range(weeks):
        cur_date = start_date + timedelta(weeks=w)
        # Macro shocks (geopolitical, inflation cycles)
        copper_price = max(2.8, copper_price + random.uniform(-0.12, 0.14))
        steel_price = max(550.0, steel_price + random.uniform(-25.0, 28.0))
        freight_index = max(1100.0, freight_index + random.uniform(-80.0, 85.0))

        records.append({
            "week_date": cur_date.strftime("%Y-%m-%d"),
            "copper_usd_per_lb": round(copper_price, 3),
            "steel_usd_per_ton": round(steel_price, 2),
            "baltic_freight_index": round(freight_index, 1)
        })

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    logging.info(f"Macro commodity indices saved to: {filepath}")
    return records


def generate_shipment_dataset(num_records: int = 100000):
    """
    Generates 100K+ realistic shipment records for training the XGBoost
    lead-time prediction and supplier disruption engine.
    """
    filepath = os.path.join(DATA_DIR, "industrial_shipments_100k.csv")
    logging.info(f"Generating {num_records:,} industrial shipment records (DataCo Supply Chain schema)...")

    random.seed(101)
    records = []
    start_dt = datetime(2022, 1, 1)

    for i in range(num_records):
        comp = random.choice(COMPONENTS)
        supp = random.choice(SUPPLIERS)
        order_date = start_dt + timedelta(days=random.randint(0, 800))
        
        # Macro context at order time
        copper_val = round(3.8 + random.uniform(-0.8, 1.2), 2)
        steel_val = round(800 + random.uniform(-200, 300), 1)
        freight_val = round(2100 + random.uniform(-700, 1400), 1)
        
        # Scheduled lead time (days)
        sched_days = comp["lead_base_days"] + random.randint(-5, 10)
        
        # Non-linear delays influenced by supplier reliability and freight/commodity shocks
        freight_factor = (freight_val - 1500) / 1000.0  # higher freight = global port backlog
        copper_factor = (copper_val - 3.5) * 4.0
        reliability_delay = (1.0 - supp["base_reliability"]) * 60.0
        random_shock = random.uniform(-4.0, 8.0)

        # Actual lead-time days
        actual_delay_days = max(0.0, reliability_delay + freight_factor * 8.0 + copper_factor * 2.0 + random_shock)
        actual_days = int(round(sched_days + actual_delay_days))
        delay_days = actual_days - sched_days
        is_late = 1 if delay_days > 3 else 0

        # Disruption severity metric (0 to 1)
        disruption_score = min(1.0, max(0.0, (delay_days / sched_days) * comp["criticality"] * 2.5))

        records.append({
            "order_id": f"ORD-{100000 + i}",
            "order_date": order_date.strftime("%Y-%m-%d"),
            "component_name": comp["name"],
            "supplier_id": supp["id"],
            "supplier_name": supp["name"],
            "supplier_country": supp["country"],
            "shipping_mode": random.choice(["Ocean Freight", "Intermodal Rail", "Air Expedited"]),
            "copper_usd_per_lb": copper_val,
            "steel_usd_per_ton": steel_val,
            "freight_index": freight_val,
            "scheduled_lead_days": sched_days,
            "actual_lead_days": actual_days,
            "delay_days": delay_days,
            "is_late": is_late,
            "disruption_score": round(disruption_score, 3)
        })

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    logging.info(f"Successfully generated {len(records):,} records at: {filepath}")


def generate_sample_10k_filings():
    """Generates authentic corporate 10-K filing excerpts for the LangGraph agent to extract."""
    filing_1 = """
    UNITED STATES SECURITIES AND EXCHANGE COMMISSION
    WASHINGTON, D.C. 20549 - FORM 10-K ANNUAL REPORT
    COMPANY: VESTAS & SIEMENS ENERGY HEAVY INDUSTRIAL EQUIPMENT DIVISION
    ITEM 1A. RISK FACTORS - GLOBAL SUPPLY CHAIN DISRUPTIONS & PROCUREMENT DELAYS

    During fiscal year 2024, our Grid Technologies and Offshore Wind turbine divisions experienced
    prolonged procurement lead times. Specifically, high-voltage transformer supplies from European
    and East Asian suppliers faced an average delivery backlog delay of 14 weeks due to severe electrical
    steel shortages and copper raw material price volatility.

    Key Supplier Bottlenecks Identified:
    - High-Voltage Transformers: Supplier Bavaria Precision and Goyang Smart Power reported component
      lead time extensions of up to 180 to 220 days, resulting in potential project execution penalties.
    - Subsea Cables: Global installation vessel shortages and Baltic freight rate spikes (+28%) caused
      cumulative delivery slippage of approximately 6 to 9 weeks across North Sea projects.
    - Mitigating Actions: Management is instituting secondary dual-sourcing agreements and dynamic
      hedging contracts for refined copper cathodes to buffer against future geopolitical trade tariff shocks.
    """
    
    filing_2 = """
    COMPANY: GLOBAL POWER DYNAMICS & EQUIPMENT CORP
    FORM 10-K / ANNUAL PERFORMANCE FILING - SUPPLY CHAIN DISCLOSURE

    Item 7. Management's Discussion and Analysis of Financial Condition.
    Our Utility Scale Solar Inverter and Battery Energy Storage System (BESS) deployments were impacted
    by semiconductor fabrication constraints and customs clearance bottlenecks at primary West Coast ports.
    Component shipments from Taiwan and China recorded an average delay of 5.5 weeks.
    Freight indices surged during Q3, driving an estimated 11.2% CapEx cost inflation across scheduled
    EPC utility deliveries. Sourcing diversification is underway to contract local North American fabricators.
    """

    f1_path = os.path.join(REPORTS_DIR, "Siemens_Vestas_10K_SupplyChain_Excerpts.txt")
    f2_path = os.path.join(REPORTS_DIR, "GlobalPower_10K_Disruption_Filing.txt")

    with open(f1_path, "w", encoding="utf-8") as f:
        f.write(filing_1.strip())
    with open(f2_path, "w", encoding="utf-8") as f:
        f.write(filing_2.strip())

    logging.info(f"Sample corporate 10-K filings created in: {REPORTS_DIR}")


if __name__ == "__main__":
    generate_macro_indices_csv()
    generate_sample_10k_filings()
    # Generate 100K shipment dataset
    generate_shipment_dataset(num_records=100000)
