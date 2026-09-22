"""
Quantitative Risk Engine: Supplier Disruption Index Calculator.
Computes multi-criteria mathematical risk scores (0-100) across industrial equipment suppliers.
"""

import os
import csv
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SHIPMENTS_CSV = os.path.join(DATA_DIR, "industrial_shipments_100k.csv")
RISK_EXPORT_CSV = os.path.join(DATA_DIR, "supplier_risk_scorecard.csv")


def compute_supplier_risk_scorecard():
    logging.info("Computing Quantitative Supplier Disruption Index across 100K+ shipment records...")
    if not os.path.exists(SHIPMENTS_CSV):
        raise FileNotFoundError(f"Shipment dataset not found at {SHIPMENTS_CSV}. Please run data_loader.py first.")

    # Aggregate shipment stats by supplier
    supp_stats = {}
    with open(SHIPMENTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s_id = row["supplier_id"]
            if s_id not in supp_stats:
                supp_stats[s_id] = {
                    "id": s_id,
                    "name": row["supplier_name"],
                    "country": row["supplier_country"],
                    "total_orders": 0,
                    "late_orders": 0,
                    "total_delay_days": 0.0,
                    "delays": []
                }
            st = supp_stats[s_id]
            st["total_orders"] += 1
            delay = float(row["delay_days"])
            st["total_delay_days"] += delay
            st["delays"].append(delay)
            if int(row["is_late"]) == 1:
                st["late_orders"] += 1

    # Multi-criteria scoring weights
    # w1: Late delivery rate (35%)
    # w2: Average delay duration (30%)
    # w3: Variance / Volatility of delays (20%)
    # w4: Geopolitical trade friction weight (15%)
    geo_weights = {
        "Denmark": 0.10, "Germany": 0.15, "Japan": 0.12, "USA": 0.15,
        "South Korea": 0.25, "Taiwan": 0.45, "China": 0.65, "India": 0.35
    }

    results = []
    for s_id, st in supp_stats.items():
        n = st["total_orders"]
        late_rate = st["late_orders"] / n
        avg_delay = st["total_delay_days"] / n
        
        # Variance of delay
        mean = avg_delay
        variance = sum((x - mean) ** 2 for x in st["delays"]) / n
        std_dev = variance ** 0.5

        geo_score = geo_weights.get(st["country"], 0.25)

        # Normalized index components (0 to 1)
        norm_late = min(1.0, late_rate / 0.40)
        norm_delay = min(1.0, avg_delay / 15.0)
        norm_volatility = min(1.0, std_dev / 8.0)

        # Composite Mathematical Index (0 to 100)
        raw_index = (
            0.35 * norm_late +
            0.30 * norm_delay +
            0.20 * norm_volatility +
            0.15 * geo_score
        ) * 100.0
        disruption_index = round(min(100.0, max(0.0, raw_index)), 1)

        if disruption_index >= 65.0:
            tier = "Critical Disruption Risk"
        elif disruption_index >= 40.0:
            tier = "Moderate Risk"
        else:
            tier = "Reliable / Low Risk"

        results.append({
            "supplier_id": s_id,
            "supplier_name": st["name"],
            "country": st["country"],
            "total_orders": n,
            "on_time_delivery_pct": round((1.0 - late_rate) * 100, 1),
            "avg_delay_days": round(avg_delay, 1),
            "delay_volatility_std": round(std_dev, 2),
            "disruption_index": disruption_index,
            "risk_tier": tier
        })

    results.sort(key=lambda x: x["disruption_index"], reverse=True)

    # Export to CSV
    with open(RISK_EXPORT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logging.info("=" * 70)
    logging.info("  QUANTITATIVE SUPPLIER DISRUPTION INDEX (SCORECARD)")
    logging.info("=" * 70)
    for r in results:
        logging.info(f"  {r['supplier_id']} | {r['supplier_name']:<32} | Score: {r['disruption_index']:>4.1f}/100 | Tier: {r['risk_tier']}")
    logging.info(f"Scorecard saved to: {RISK_EXPORT_CSV}")
    return results


if __name__ == "__main__":
    compute_supplier_risk_scorecard()
