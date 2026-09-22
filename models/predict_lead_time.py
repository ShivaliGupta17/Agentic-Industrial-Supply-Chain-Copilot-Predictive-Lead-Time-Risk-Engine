"""
Inference API for Predicting Industrial Equipment Lead Times & Delays.
"""

import os
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def predict_lead_time_delay(
    scheduled_lead_days: float,
    freight_index: float,
    copper_price_usd_lb: float,
    supplier_country: str,
    shipping_mode: str = "Ocean Freight"
) -> dict:
    """
    Returns estimated actual delivery lead-time and predicted delay in days.
    """
    country_coeffs = {
        "Denmark": 3.6, "Germany": 5.4, "Japan": 3.0, "USA": 6.6,
        "South Korea": 7.2, "Taiwan": 8.4, "China": 12.6, "India": 9.6
    }
    mode_coeffs = {"Air Expedited": -4.0, "Intermodal Rail": 0.0, "Ocean Freight": 3.5}

    c_bias = country_coeffs.get(supplier_country, 6.0)
    f_effect = (freight_index - 1500) * 0.0075
    cu_effect = (copper_price_usd_lb - 3.5) * 1.8
    m_effect = mode_coeffs.get(shipping_mode, 0.0)

    predicted_actual_days = round(scheduled_lead_days + c_bias + f_effect + cu_effect + m_effect, 1)
    predicted_delay_days = round(max(0.0, predicted_actual_days - scheduled_lead_days), 1)

    is_late_risk = "HIGH RISK (>7 days delay)" if predicted_delay_days > 7.0 else (
        "MODERATE RISK (3-7 days delay)" if predicted_delay_days > 3.0 else "ON SCHEDULE"
    )

    return {
        "scheduled_days": scheduled_lead_days,
        "predicted_actual_days": predicted_actual_days,
        "predicted_delay_days": predicted_delay_days,
        "risk_classification": is_late_risk,
        "macro_drivers": {
            "freight_impact_days": round(f_effect, 1),
            "commodity_impact_days": round(cu_effect, 1),
            "supplier_origin_bias_days": round(c_bias, 1)
        }
    }


if __name__ == "__main__":
    # Test sample inference
    result = predict_lead_time_delay(
        scheduled_lead_days=180,
        freight_index=2850,
        copper_price_usd_lb=4.65,
        supplier_country="China",
        shipping_mode="Ocean Freight"
    )
    print(json.dumps(result, indent=4))
