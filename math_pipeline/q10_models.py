# q10_models.py
import numpy as np

def q10_spoilage_rate(temp_celsius, ref_temp=25.0, q10=2.0, base_rate=1.0):
    """Simple Q10 rate."""
    return base_rate * (q10 ** ((temp_celsius - ref_temp) / 10.0))

def adaptive_q10_spoilage_rate(
    temp_celsius,
    recent_inside_temps,
    recent_outside_temps,
    recent_humidity,
    recent_door,
    base_q10=2.0,
    base_rate=1.0,
    window=10
):
    """
    Adaptive Q10 spoilage rate with dynamic ref_temp, q10, and modifier.
    Mirrors and unifies your earlier adaptive implementation.
    """
    # Rolling window slices
    inside_hist = np.asarray(recent_inside_temps[-window:], dtype=float)
    outside_hist = np.asarray(recent_outside_temps[-window:], dtype=float)
    humidity_hist = np.asarray(recent_humidity[-window:], dtype=float)
    door_hist = np.asarray(recent_door[-window:], dtype=float)

    if inside_hist.size == 0:  # fallback
        return q10_spoilage_rate(temp_celsius, ref_temp=25.0, q10=base_q10, base_rate=base_rate)

    # Adaptive reference temperature (door-bias towards outside)
    door_factor = float(np.mean(door_hist)) if door_hist.size else 0.0
    ref_temp = float(np.mean(inside_hist)) + 0.5 * door_factor * (float(np.mean(outside_hist)) - float(np.mean(inside_hist)))

    # Adaptive q10 by inside variability
    temp_variability = float(np.std(inside_hist))
    q10 = base_q10 * (1 + temp_variability / 10.0)

    # Outside influence modifier: more effect when inside ~ outside and outside is volatile
    delta = abs(temp_celsius - (float(np.mean(outside_hist)) if outside_hist.size else temp_celsius))
    outside_variability = float(np.std(outside_hist)) if outside_hist.size else 0.0
    modifier = 1 + (0.05 + outside_variability / 100.0) * np.exp(-delta / 5.0)

    # Humidity & door affect bounds
    humidity_avg = (float(np.mean(humidity_hist)) / 100.0) if humidity_hist.size else 0.0
    door_freq = float(np.mean(door_hist)) if door_hist.size else 0.0
    min_rate = 0.0 + 0.5 * humidity_avg
    max_rate = 5.0 * (1 + door_freq + humidity_avg)

    rate = base_rate * (q10 ** ((temp_celsius - ref_temp) / 10.0))
    rate *= modifier
    rate = float(np.clip(rate, min_rate, max_rate))
    return rate
