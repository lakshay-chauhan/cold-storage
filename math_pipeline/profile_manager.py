from math import tanh

# ---- Single source of truth for base product profiles ----
PRODUCT_PROFILES = {
    "fruit": {
        # General fresh produce: safe at 4–10 °C, faster spoilage above 20 °C
        "q10": 2.5, "ref_temp": 10,
        "z_threshold": 3.0, "ewma_threshold": 2.5,
        "ewma_alpha": 0.35,
        "max_safe_temp": 20, "min_rate": 0.0, "max_rate": 5.0,
        "weights": {"temp":0.35,"humidity":0.25,"door":0.1,"gas":0.2,"interaction":0.05,"outside":0.05},
        "adaptive_q10": True, "adaptive_window": 25, "adaptive_ewma": True,
        "logistic": {"midpoint": 40.0, "slope": 8.0},   # slower onset, gentler slope
        "alerts": {
            "high_temp":"Fruit temperature exceeds safe limit!",
            "rapid_spoilage":"Fruit spoilage rate too high!"
        }
    },

    "vaccine": {
        # CDC/WHO: 2–8 °C strict storage
        "q10": 2.0, "ref_temp": 5,
        "z_threshold": 2.0, "ewma_threshold": 1.8,
        "ewma_alpha": 0.2,
        "max_safe_temp": 8, "min_rate": 0.0, "max_rate": 2.0,
        "weights": {"temp":0.55,"humidity":0.05,"door":0.25,"gas":0.05,"interaction":0.05,"outside":0.05},
        "adaptive_q10": True, "adaptive_window": 40, "adaptive_ewma": True,
        "logistic": {"midpoint": 50.0, "slope": 12.0},  # sharp slope, small margin
        "alerts": {
            "high_temp":"Vaccine temperature exceeds WHO limit!",
            "rapid_spoilage":"Vaccine degradation rate too high!"
        }
    },

    "seafood": {
        # FDA/FAO: keep ≤4 °C, spoilage rapid if above
        "q10": 3.0, "ref_temp": 4,
        "z_threshold": 2.5, "ewma_threshold": 2.2,
        "ewma_alpha": 0.3,
        "max_safe_temp": 5, "min_rate": 0.0, "max_rate": 7.0,
        "weights": {"temp":0.5,"humidity":0.2,"door":0.15,"gas":0.15,"interaction":0.03,"outside":0.02},
        "adaptive_q10": True, "adaptive_window": 35, "adaptive_ewma": True,
        "logistic": {"midpoint": 35.0, "slope": 7.0},   # faster spoilage onset
        "alerts": {
            "high_temp":"Seafood temperature exceeds safe limit!",
            "rapid_spoilage":"Seafood spoilage rate too high!"
        }
    }
}



def _validate_inputs(product, temp_outside, door_open):
    if product not in PRODUCT_PROFILES:
        raise ValueError(f"Unknown product '{product}'")
    if door_open not in (0,1):
        raise ValueError("door_open must be 0 or 1")
    if temp_outside is not None and not (-30 <= temp_outside <= 50):
        raise ValueError("temp_outside out of realistic range (-30..50 °C)")

def get_dynamic_profile(product="fruit", temp_outside=None, door_open=0, variability=None):
    """
    Returns a dynamic profile derived from PRODUCT_PROFILES by adapting to:
      - outside temperature (temp_outside)
      - door state (door_open)
      - recent variability (std) of target signal
    All downstream calculations should use this dynamic profile.
    """
    _validate_inputs(product, temp_outside, door_open)

    base = PRODUCT_PROFILES[product]
    profile = {
        **{k: v for k, v in base.items() if k not in ("weights",)},
        "weights": {**base["weights"]},
    }

    # ---- Dynamic max_safe_temp (smooth ±5°C around ref, bounded) ----
    if temp_outside is not None:
        delta = temp_outside - profile["ref_temp"]
        profile["max_safe_temp"] = base["max_safe_temp"] + 5.0 * tanh(delta / 10.0)
    else:
        profile["max_safe_temp"] = base["max_safe_temp"]

    # Door effect (gentle, smooth)
    profile["max_safe_temp"] += 0.5 * door_open

    # Bound limits relative to ref_temp
    profile["max_safe_temp"] = min(profile["max_safe_temp"], profile["ref_temp"] + 10.0)
    profile["max_safe_temp"] = max(profile["max_safe_temp"], profile["ref_temp"] - 5.0)

    # ---- Dynamic max_rate with headroom when max_safe_temp rises ----
    profile["max_rate_dynamic"] = base["max_rate"] * (1 + 0.05 * (profile["max_safe_temp"] - profile["ref_temp"]))
    profile["max_rate_dynamic"] = max(base["min_rate"], min(profile["max_rate_dynamic"], base["max_rate"] * 1.5))

    # ---- Adaptive Q10 based on variability (clamped) ----
    if base.get("adaptive_q10", False) and variability is not None:
        q10_dyn = base["q10"] * (1 + 0.01 * float(variability))
        profile["q10_dynamic"] = min(q10_dyn, base["q10"] * 1.5)
    else:
        profile["q10_dynamic"] = base["q10"]

    # ---- Adaptive Z/EWMA thresholds from variability (clamped) ----
    if variability is not None:
        z = base["z_threshold"] * (1 + 0.1 * float(variability))
        ew = base["ewma_threshold"] * (1 + 0.1 * float(variability))
        profile["z_threshold_dynamic"] = min(z, base["z_threshold"] * 2.0)
        profile["ewma_threshold_dynamic"] = min(ew, base["ewma_threshold"] * 2.0)
    else:
        profile["z_threshold_dynamic"] = base["z_threshold"]
        profile["ewma_threshold_dynamic"] = base["ewma_threshold"]

    # ---- Adaptive weights ----
    # Outside heat and door increase temperature weight; variability increases gas weight slightly.
    if temp_outside is not None:
        profile["weights"]["temp"] *= 1 + 0.1 * tanh((temp_outside - profile["ref_temp"]) / 10.0)
    if door_open == 1:
        profile["weights"]["door"] *= 1.2
        profile["weights"]["interaction"] *= 1.1
    if variability is not None:
        profile["weights"]["gas"] *= 1 + 0.05 * float(variability)

    # Normalize weights to sum=1
    total = sum(profile["weights"].values()) or 1.0
    profile["weights"] = {k: v/total for k, v in profile["weights"].items()}

    return profile
