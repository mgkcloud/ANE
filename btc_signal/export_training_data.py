#!/usr/bin/env python3
"""
Export BTC trading signals + outcomes to training-ready binary format.
Joins oracle signals and confidence engine ticks against trade outcomes.
Output: binary file of [features..., label] rows for ANE training.
"""
import json
import struct
import sys
import os
from collections import defaultdict

DATA_DIR = "/home/ubuntu/trading/v4/data/v4_btc_live"

def load_outcomes():
    """Load closed trades, keyed by market ID."""
    trades = [json.loads(l) for l in open(f"{DATA_DIR}/trades.jsonl")]
    closes = {}
    for t in trades:
        if t.get("type") == "close":
            closes[t["market"]] = {
                "won": t.get("won", False),
                "side": t.get("side", ""),
                "winner": t.get("winner", ""),
                "pnl": t.get("pnl", 0),
                "entry_price": t.get("entry_price", 0),
            }
    return closes

def extract_features_from_signal(sig):
    """Extract feature vector from a confidence engine signal."""
    liq = sig.get("liquidation", {})
    return {
        "divergence_pct": sig.get("divergence_pct", 0) or 0,
        "confidence": sig.get("confidence", 0) or 0,
        "side_up": 1.0 if sig.get("side") == "up" else 0.0,
        "vpin": sig.get("vpin", 0) or 0,
        "twap": sig.get("twap", 0) or 0,
        "momentum": sig.get("momentum", 0) or 0,
        "liq_chaos": 1.0 if liq.get("chaos") else 0.0,
        "liq_direction": float(liq.get("direction", 0)),
        "liq_volume": float(liq.get("volume_usd", 0)),
        "liq_long": float(liq.get("long_liqs", 0)),
        "liq_short": float(liq.get("short_liqs", 0)),
    }

def extract_features_from_oracle(oracle):
    """Extract feature vector from an oracle signal."""
    regime_map = {"calm": 0, "normal": 1, "volatile": 2, "extreme": 3}
    timing_map = {"wait": 0, "soon": 1, "now": 2}
    risk_map = {"low": 0, "medium": 1, "high": 2}
    
    return {
        "divergence_pct": float(oracle.get("change_pct", 0) or 0),
        "confidence": float(oracle.get("confidence_adjusted", 0) or oracle.get("confidence_raw", 0) or 0),
        "side_up": 1.0 if oracle.get("side") == "up" else 0.0,
        "vpin": 0.0,  # not in oracle signals
        "twap": 0.0,  # not in oracle signals
        "momentum": 0.0,  # not in oracle signals
        "liq_chaos": 0.0,
        "liq_direction": 0.0,
        "liq_volume": 0.0,
        "liq_long": 0.0,
        "liq_short": 0.0,
        # Oracle-specific features
        "oracle_regime": float(regime_map.get(oracle.get("oracle_regime", ""), 1)),
        "oracle_timing": float(timing_map.get(oracle.get("oracle_timing", ""), 0)),
        "oracle_risk": float(risk_map.get(oracle.get("oracle_risk", ""), 1)),
        "oracle_modifier": float(oracle.get("oracle_modifier", 1.0)),
    }

FEATURE_NAMES = [
    "divergence_pct", "confidence", "side_up", "vpin", "twap", "momentum",
    "liq_chaos", "liq_direction", "liq_volume", "liq_long", "liq_short",
    "oracle_regime", "oracle_timing", "oracle_risk", "oracle_modifier",
]
N_FEATURES = len(FEATURE_NAMES)

def normalize_features(rows):
    """Z-score normalize each feature column."""
    import statistics
    n = len(rows)
    if n == 0:
        return rows, {}, {}
    
    means = {}
    stds = {}
    for i, name in enumerate(FEATURE_NAMES):
        vals = [r[i] for r in rows]
        m = statistics.mean(vals) if vals else 0
        s = statistics.stdev(vals) if len(vals) > 1 else 1
        if s == 0:
            s = 1
        means[name] = m
        stds[name] = s
        for j in range(n):
            rows[j][i] = (rows[j][i] - m) / s
    
    return rows, means, stds

def main():
    print("Loading outcomes...")
    closes = load_outcomes()
    print(f"  {len(closes)} closed trades with outcomes")
    
    rows = []  # Each row: [feature_0, ..., feature_N, label]
    
    # Source 1: signals.jsonl (confidence engine ticks)
    print("Processing signals.jsonl...")
    signals = [json.loads(l) for l in open(f"{DATA_DIR}/signals.jsonl")]
    sig_labeled = 0
    for s in signals:
        ts = s.get("timestamp", 0)
        mkt_ts = int(ts) // 300 * 300
        mkt = f"btc-updown-5m-{mkt_ts}"
        if mkt in closes:
            feats = extract_features_from_signal(s)
            # Fill oracle features with defaults
            for k in ["oracle_regime", "oracle_timing", "oracle_risk", "oracle_modifier"]:
                if k not in feats:
                    feats[k] = 0.0 if k != "oracle_modifier" else 1.0
            row = [feats.get(name, 0.0) for name in FEATURE_NAMES]
            # Label: did the market side that was predicted actually win?
            outcome = closes[mkt]
            # If the signal predicted the correct side, label=1
            predicted_side = s.get("side", "")
            actual_winner = outcome.get("winner", "")
            label = 1.0 if predicted_side == actual_winner else 0.0
            row.append(label)
            rows.append(row)
            sig_labeled += 1
    print(f"  {sig_labeled} labeled signal ticks")
    
    # Source 2: oracle signals from trades.jsonl
    print("Processing oracle signals...")
    trades = [json.loads(l) for l in open(f"{DATA_DIR}/trades.jsonl")]
    oracle_labeled = 0
    for t in trades:
        if t.get("type") != "oracle_signal":
            continue
        mkt = t.get("market", "")
        if mkt in closes:
            feats = extract_features_from_oracle(t)
            row = [feats.get(name, 0.0) for name in FEATURE_NAMES]
            outcome = closes[mkt]
            predicted_side = t.get("side", "")
            actual_winner = outcome.get("winner", "")
            label = 1.0 if predicted_side == actual_winner else 0.0
            row.append(label)
            rows.append(row)
            oracle_labeled += 1
    print(f"  {oracle_labeled} labeled oracle signals")
    
    print(f"\nTotal labeled samples: {len(rows)}")
    
    # Label distribution
    pos = sum(1 for r in rows if r[-1] > 0.5)
    neg = len(rows) - pos
    print(f"Positive (correct prediction): {pos} ({100*pos/len(rows):.1f}%)")
    print(f"Negative (wrong prediction): {neg} ({100*neg/len(rows):.1f}%)")
    
    # Normalize
    print("\nNormalizing features...")
    features = [r[:-1] for r in rows]
    labels = [r[-1] for r in rows]
    features, means, stds = normalize_features(features)
    rows = [f + [l] for f, l in zip(features, labels)]
    
    # Save normalization params
    norm_path = os.path.join(os.path.dirname(sys.argv[1]) if len(sys.argv) > 1 else ".", "norm_params.json")
    json.dump({"means": means, "stds": stds, "feature_names": FEATURE_NAMES}, 
              open(norm_path, "w"), indent=2)
    print(f"Saved normalization params to {norm_path}")
    
    # Write binary: header + float32 rows
    out_path = sys.argv[1] if len(sys.argv) > 1 else "training_data.bin"
    with open(out_path, "wb") as f:
        # Header: magic, version, n_samples, n_features
        f.write(struct.pack("<IIII", 0x42544353, 1, len(rows), N_FEATURES))
        # Data: row-major float32
        for row in rows:
            f.write(struct.pack(f"<{N_FEATURES + 1}f", *row))
    
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nWritten {out_path}: {len(rows)} samples x {N_FEATURES} features = {size_mb:.1f} MB")
    
    # Quick stats
    print(f"\nFeature summary (post-normalization):")
    for i, name in enumerate(FEATURE_NAMES):
        vals = [r[i] for r in rows]
        print(f"  {name:20s}: mean={sum(vals)/len(vals):+.3f} std={max(0.01, (sum((v-sum(vals)/len(vals))**2 for v in vals)/len(vals))**0.5):.3f}")

if __name__ == "__main__":
    main()
