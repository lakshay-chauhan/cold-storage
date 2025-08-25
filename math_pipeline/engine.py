# engine.py
from collections import deque
from math import exp
from typing import Deque, Dict, Any, Optional
import numpy as np

from profile_manager import get_dynamic_profile
from q10_models import q10_spoilage_rate, adaptive_q10_spoilage_rate

def _sigmoid(x):  # helper
    return 1.0 / (1.0 + np.exp(-x))

class SpoilageEngine:
    """
    Stateful engine that:
      - Builds dynamic profile each update (adapts to outside/door/variability)
      - Computes instant spoilage via weighted factors + logistic mapping (bounded 0..100)
      - Updates cumulative spoilage via multiplicative quality decay
      - Detects anomalies (adaptive Z-score and EWMA)
      - Produces triaged risk (ok | warning | critical) with tunable debouncing
    """

    def __init__(self, product: str = "vaccine", window: int = 30, mode: str = "adaptive"):
        self.product = product
        self.window = window
        self.mode = mode  # "adaptive" or "simple"
        self.quality = 1.0  # multiplicative quality (1.0 = pristine)
        self.history_instant: Deque[float] = deque(maxlen=window)
        self.recent: Dict[str, Deque[float]] = {
            "ti": deque(maxlen=window), "to": deque(maxlen=window),
            "h": deque(maxlen=window), "d": deque(maxlen=window)
        }
        self.last_ts: Optional[float] = None  # seconds since epoch or monotonic
        # Alert behavior knobs
        self.require_consecutive = 2  # breaches required to upgrade risk
        self._breach_count_warn = 0
        self._breach_count_crit = 0

    def _dt_minutes(self, ts_seconds: Optional[float]) -> float:
        if ts_seconds is None or self.last_ts is None:
            self.last_ts = ts_seconds
            return 1.0
        dt = max(0.1, (ts_seconds - self.last_ts) / 60.0)
        self.last_ts = ts_seconds
        return dt

    def _outside_penalty(self, ti: float, to: float) -> float:
        # Smooth penalty based on |ΔT| relative to ~10°C with slope 0.5
        delta_t = abs(ti - to)
        return 1.0 / (1.0 + np.exp((delta_t - 10.0) / 2.0))

    def update(self, reading: Dict[str, Any]) -> Dict[str, Any]:
        """
        reading = {
          "ts": <seconds since epoch or monotonic float>,
          "product": "vaccine",
          "temp_inside_c": 4.8, "temp_outside_c": 31.2,
          "humidity_pct": 62.0, "door_open": 0, "gas_ppm": 520.0
        }
        """
        # ---- Parse input ----
        ts = reading.get("ts", None)
        ti = float(reading["temp_inside_c"])
        to = float(reading["temp_outside_c"])
        h = float(reading["humidity_pct"])
        d = int(reading["door_open"])
        g = float(reading.get("gas_ppm", 0.0))
        if "product" in reading and reading["product"] != self.product:
            self.product = reading["product"]

        dt = self._dt_minutes(ts)

        # ---- Recent variability (from instant history) ----
        variability = np.std(self.history_instant) if len(self.history_instant) > 5 else None

        # ---- Dynamic profile (adapts to outside/door/variability) ----
        profile = get_dynamic_profile(self.product, temp_outside=to, door_open=d, variability=variability)
        weights = profile["weights"]
        # Normalize again for safety
        s = sum(weights.values()) or 1.0
        w = {k: v / s for k, v in weights.items()}

        # ---- Temperature spoilage rate (Q10) ----
        if self.mode == "adaptive" and len(self.recent["ti"]) >= 5:
            tr = adaptive_q10_spoilage_rate(
                ti, list(self.recent["ti"]), list(self.recent["to"]),
                list(self.recent["h"]), list(self.recent["d"]),
                base_q10=profile["q10"], base_rate=1.0,
                window=min(10, len(self.recent["ti"]))
            )
        else:
            tr = q10_spoilage_rate(ti, ref_temp=profile["ref_temp"], q10=profile["q10_dynamic"], base_rate=1.0)
        tr = min(profile["max_rate_dynamic"], max(profile["min_rate"], tr))

        # ---- Other factors ----
        humidity = h / 100.0
        door_penalty = 1.0 if d == 1 else 0.0
        gas = g / 1000.0
        interaction = humidity * (ti / 30.0)
        outside_penalty = self._outside_penalty(ti, to)

        # ---- Weighted raw index ----
        raw = (tr * w.get("temp", 0) +
               humidity * w.get("humidity", 0) +
               door_penalty * w.get("door", 0) +
               gas * w.get("gas", 0) +
               interaction * w.get("interaction", 0) +
               outside_penalty * w.get("outside", 0))

        # ---- Logistic mapping (bounded 0..100) ----
        mid = profile.get("logistic", {}).get("midpoint", 50.0)
        slope = profile.get("logistic", {}).get("slope", 10.0)
        scaled = raw * 100.0
        instant = 100.0 / (1.0 + exp(-(scaled - mid) / max(1e-6, slope)))
        instant = float(np.clip(instant, 0.0, 100.0))

        # ---- Multiplicative cumulative spoilage via quality decay ----
        self.quality *= max(0.0, 1.0 - (instant / 100.0) * dt)
        cumulative = 100.0 * (1.0 - self.quality)
        cumulative = float(np.clip(cumulative, 0.0, 100.0))

        # ---- Maintain histories ----
        self.history_instant.append(instant)
        for k, v in [("ti", ti), ("to", to), ("h", h), ("d", d)]:
            self.recent[k].append(v)

        # ---- Adaptive Z-score anomaly ----
        arr = np.array(self.history_instant, dtype=float)
        z_thr = profile["z_threshold_dynamic"]
        z = 0.0
        z_anom = False
        if len(arr) > 3 and np.std(arr) > 1e-8:
            z = (arr[-1] - np.mean(arr)) / np.std(arr)
            z_anom = abs(z) > z_thr

        # ---- Adaptive EWMA anomaly ----
        alpha = profile.get("ewma_alpha", 0.3)
        L = profile["ewma_threshold_dynamic"]
        ew_anom = False
        dev = 0.0
        if len(arr) > 1:
            ew = arr[0]
            for v in arr[1:]:
                ew = alpha * v + (1 - alpha) * ew
            lam = np.sqrt((alpha / (2 - alpha)) * (1 - (1 - alpha) ** (2 * len(arr))))
            dev = abs(arr[-1] - ew)
            ew_anom = dev > L * (np.std(arr) if np.std(arr) > 0 else 1.0) * lam

        # ---- Dynamic risk thresholds (warn/crit) ----
        if len(arr) >= 5:
            mu, sigma = float(np.mean(arr)), float(np.std(arr))
            warn = min(100.0, max(30.0, mu + 0.5 * sigma))
            crit = min(100.0, max(40.0, mu + 1.0 * sigma))
        else:
            warn, crit = 60.0, 80.0

        # Debounce
        risk = "ok"
        if instant > crit:
            self._breach_count_crit += 1
            self._breach_count_warn = max(self._breach_count_warn - 1, 0)
            if self._breach_count_crit >= self.require_consecutive:
                risk = "critical"
        elif instant > warn:
            self._breach_count_warn += 1
            self._breach_count_crit = max(self._breach_count_crit - 1, 0)
            if self._breach_count_warn >= self.require_consecutive:
                risk = "warning"
        else:
            self._breach_count_warn = 0
            self._breach_count_crit = 0

        contributions = {
            "temp": tr * w.get("temp", 0),
            "humidity": humidity * w.get("humidity", 0),
            "door": door_penalty * w.get("door", 0),
            "gas": gas * w.get("gas", 0),
            "interaction": interaction * w.get("interaction", 0),
            "outside": outside_penalty * w.get("outside", 0),
        }

        return {
            "ts": ts,
            "product": self.product,
            "instant_spoilage_pct": round(instant, 2),
            "cumulative_spoilage_pct": round(cumulative, 2),
            "risk_level": risk,
            "anomalies": {"zscore": bool(z_anom), "ewma": bool(ew_anom)},
            "adaptive_thresholds": {
                "z": float(z_thr),
                "ewma_L": float(L),
                "warn": round(warn, 2),
                "crit": round(crit, 2)
            },
            "contributions": {k: round(float(v), 3) for k, v in contributions.items()},
            "notes": [f"ΔT={round(abs(ti - to),1)}°C", f"z={round(float(z),2)}", f"EW_dev={round(float(dev),2)}"]
        }
