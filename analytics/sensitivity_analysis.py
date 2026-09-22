"""
Quantitative Analytics: CapEx Escalation & Sensitivity Simulation Engine.
Simulates total project capital expenditure inflation under custom tariff and shipping freight shocks.
"""

import os
import csv
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SENSITIVITY_CSV = os.path.join(DATA_DIR, "capex_sensitivity_matrix.csv")


def run_capex_sensitivity_simulation(
    base_project_capex_million_usd: float = 250.0,
    import_content_share: float = 0.45,
    ocean_freight_cost_share: float = 0.08
):
    """
    Simulates CapEx inflation across a 2D grid of Tariff increases (0% to +30%)
    and Ocean Freight Rate spikes (0% to +50%).
    """
    logging.info("=" * 70)
    logging.info(f"RUNNING CAPEX SENSITIVITY SIMULATION (Base CapEx: ${base_project_capex_million_usd}M)")
    logging.info(f"Assumptions: Imported Component Share = {import_content_share*100:.0f}%, Freight Share = {ocean_freight_cost_share*100:.0f}%")
    logging.info("=" * 70)

    tariff_shocks = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    freight_shocks = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

    matrix_records = []
    for t_shock in tariff_shocks:
        for f_shock in freight_shocks:
            # Tariff impact on imported hardware
            tariff_escalation_pct = t_shock * import_content_share
            # Freight shock impact on logistics budget
            freight_escalation_pct = f_shock * ocean_freight_cost_share
            
            total_cost_escalation_pct = round((tariff_escalation_pct + freight_escalation_pct) * 100, 2)
            escalated_capex_m = round(base_project_capex_million_usd * (1.0 + (total_cost_escalation_pct / 100.0)), 2)
            capex_variance_m = round(escalated_capex_m - base_project_capex_million_usd, 2)

            matrix_records.append({
                "tariff_shock_pct": int(t_shock * 100),
                "freight_shock_pct": int(f_shock * 100),
                "total_cost_escalation_pct": total_cost_escalation_pct,
                "escalated_capex_m_usd": escalated_capex_m,
                "capex_variance_m_usd": capex_variance_m
            })

    with open(SENSITIVITY_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(matrix_records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matrix_records)

    logging.info(f"Saved 2D Sensitivity Grid ({len(matrix_records)} scenarios) to: {SENSITIVITY_CSV}")

    # Print representative scenarios
    logging.info("\n--- Key Executive What-If Scenarios ---")
    scenarios = [
        (0, 0, "Baseline (No Shock)"),
        (10, 20, "Moderate Shock (+10% Tariff, +20% Freight)"),
        (25, 50, "Severe Geopolitical Bottleneck (+25% Tariff, +50% Freight)")
    ]
    for t, f_rate, label in scenarios:
        match = next(r for r in matrix_records if r["tariff_shock_pct"] == t and r["freight_shock_pct"] == f_rate)
        logging.info(f"{label}:")
        logging.info(f"   -> CapEx Escalation: +{match['total_cost_escalation_pct']}% (+${match['capex_variance_m_usd']}M) -> Total: ${match['escalated_capex_m_usd']}M USD")

    return matrix_records


if __name__ == "__main__":
    run_capex_sensitivity_simulation()
