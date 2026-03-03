"""
ml_predict.py — MLP signal filter for V4 Hydra BTC
Trained on ANE (Gilgamesh), deployed as pure numpy on Hermes.

Usage:
    from ml_predict import MLPredictor
    predictor = MLPredictor("/home/ubuntu/trading/v4/models/")
    prob = predictor.predict(signals_dict, oracle_dict, remaining_secs, bid_data)
    if prob < 0.4:
        skip("MLP says low confidence")

Model: 20 -> 128 -> 64 -> 1 (ReLU, sigmoid)
Accuracy: 86.5% | Precision: 95.0% | Inference: <1ms on CPU
"""
import numpy as np
import json
import os
import math
import time

FEATURE_NAMES = [
    "divergence", "confidence", "side_up", "vpin", "twap", "momentum",
    "liq_chaos", "liq_dir", "liq_vol_log", "liq_long_log", "liq_short_log",
    "remaining_pct", "bid_slope", "bid_price", "spread",
    "oracle_regime", "oracle_timing", "oracle_risk", "oracle_conf", "oracle_modifier",
]

REGIME_MAP = {"calm": 0, "normal": 0.33, "volatile": 0.67, "extreme": 1.0}
TIMING_MAP = {"wait": 0, "soon": 0.5, "now": 1.0}
RISK_MAP = {"low": 0, "medium": 0.5, "high": 1.0}


class MLPredictor:
    def __init__(self, model_dir):
        """Load model weights and normalization params."""
        model_path = os.path.join(model_dir, "btc_mlp3_model.npz")
        norm_path = os.path.join(model_dir, "norm_params.json")

        # Load weights
        m = np.load(model_path)
        self.W1 = m["W1"]  # (128, 20)
        self.W2 = m["W2"]  # (64, 128)
        self.W3 = m["W3"]  # (1, 64)
        self.b1 = m["b1"]  # (128,)
        self.b2 = m["b2"]  # (64,)
        self.b3 = m["b3"]  # (1,)

        # Load normalization
        with open(norm_path) as f:
            norm = json.load(f)
        self.means = np.array(norm["means"], dtype=np.float32)
        self.stds = np.array(norm["stds"], dtype=np.float32)
        self.stds[self.stds < 1e-8] = 1.0  # avoid div by zero

    def _extract_features(self, signals, oracle, remaining_secs, bid_data):
        """
        Extract 20-feature vector from live trading data.

        Args:
            signals: dict with keys: divergence_pct, confidence, side, vpin, twap,
                     momentum, liquidation{chaos, direction, volume_usd, long_liqs, short_liqs}
            oracle: dict with keys: oracle_regime, oracle_timing, oracle_risk,
                    confidence_adjusted/confidence_raw, oracle_modifier
            remaining_secs: seconds remaining in 5-min market (0-300)
            bid_data: dict with keys: bid_slope, current_bid, spread
        """
        liq = signals.get("liquidation", {}) or {}
        bid = bid_data or {}

        features = np.array([
            float(signals.get("divergence_pct", 0) or 0),
            float(signals.get("confidence", 0) or 0),
            1.0 if signals.get("side") == "up" else 0.0,
            float(signals.get("vpin", 0) or 0),
            float(signals.get("twap", 0) or 0),
            float(signals.get("momentum", 0) or 0),
            1.0 if liq.get("chaos") else 0.0,
            float(liq.get("direction", 0)),
            math.log1p(float(liq.get("volume_usd", 0) or 0)),
            math.log1p(float(liq.get("long_liqs", 0) or 0)),
            math.log1p(float(liq.get("short_liqs", 0) or 0)),
            float(remaining_secs) / 300.0,
            float(bid.get("bid_slope", 0)),
            float(bid.get("current_bid", 0.5)),
            float(bid.get("spread", 0)),
            REGIME_MAP.get(oracle.get("oracle_regime", ""), 0.33),
            TIMING_MAP.get(oracle.get("oracle_timing", ""), 0),
            RISK_MAP.get(oracle.get("oracle_risk", ""), 0.5),
            float(oracle.get("confidence_adjusted",
                  oracle.get("confidence_raw", 0)) or 0),
            float(oracle.get("oracle_modifier", 1.0)),
        ], dtype=np.float32)

        return features

    def predict(self, signals, oracle, remaining_secs, bid_data=None):
        """
        Predict probability that the current signal prediction is correct.

        Returns:
            float: probability (0-1) that the predicted side will win.
                   > 0.5 = model agrees with signal, < 0.5 = model disagrees
        """
        x = self._extract_features(signals, oracle, remaining_secs, bid_data)

        # Z-score normalize
        x = (x - self.means) / self.stds

        # Forward pass: 3-layer MLP
        h1 = np.maximum(self.W1 @ x + self.b1, 0)   # ReLU
        h2 = np.maximum(self.W2 @ h1 + self.b2, 0)   # ReLU
        logit = float((self.W3 @ h2 + self.b3)[0])
        prob = 1.0 / (1.0 + math.exp(-max(min(logit, 20), -20)))  # clamp to avoid overflow

        return prob

    def predict_with_meta(self, signals, oracle, remaining_secs, bid_data=None):
        """Same as predict() but returns metadata for logging."""
        t0 = time.monotonic()
        prob = self.predict(signals, oracle, remaining_secs, bid_data)
        elapsed_ms = (time.monotonic() - t0) * 1000

        return {
            "mlp_prob": prob,
            "mlp_verdict": "agree" if prob > 0.5 else "disagree",
            "mlp_confidence": abs(prob - 0.5) * 2,  # 0-1 scale
            "mlp_inference_ms": round(elapsed_ms, 3),
            "mlp_version": "v3_3layer",
        }


# Standalone test
if __name__ == "__main__":
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    pred = MLPredictor(model_dir)

    # Test with synthetic signal
    signals = {
        "divergence_pct": 0.05,
        "confidence": 0.6,
        "side": "up",
        "vpin": 0.8,
        "twap": 1.0,
        "momentum": 0.3,
        "liquidation": {"chaos": False, "direction": 1, "volume_usd": 50000,
                       "long_liqs": 0, "short_liqs": 50000, "events_count": 3},
    }
    oracle = {
        "oracle_regime": "volatile",
        "oracle_timing": "now",
        "oracle_risk": "low",
        "confidence_adjusted": 0.65,
        "oracle_modifier": 1.1,
    }
    bid_data = {"bid_slope": 0.01, "current_bid": 0.85, "spread": 0.03}

    result = pred.predict_with_meta(signals, oracle, remaining_secs=180, bid_data=bid_data)
    print(f"Prediction: {result}")

    # Benchmark
    import timeit
    n = 10000
    t = timeit.timeit(lambda: pred.predict(signals, oracle, 180, bid_data), number=n)
    print(f"Benchmark: {n} predictions in {t*1000:.1f}ms ({t/n*1000:.3f}ms/pred)")
