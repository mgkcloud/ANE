# MLP Signal Filter — Integration Guide

## What it does
Predicts probability that V4 Hydra's signal prediction is correct.
Trained on 18K labeled signal ticks from real trading data.

## Performance
- **86.5% accuracy** on training data
- **95.0% precision** (when it agrees, it's right 95%)
- **Mean prob for correct signals: 0.815 vs wrong: 0.566**
- **0.017ms inference** on Hermes t3.micro CPU
- **44KB model file** (numpy, no GPU/ANE needed)

## Files on Hermes
```
/home/ubuntu/trading/v4/models/
  btc_mlp3_model.npz    # Model weights (44KB)
  norm_params.json       # Z-score normalization parameters
  ml_predict.py          # Inference module
```

## Integration into Hydra

### 1. Import
```python
from models.ml_predict import MLPredictor
predictor = MLPredictor("/home/ubuntu/trading/v4/models/")
```

### 2. Call before entry decision
```python
# In the tick handler or entry decision:
mlp_result = predictor.predict_with_meta(
    signals=current_signals,     # dict with vpin, twap, momentum, liquidation, etc.
    oracle=oracle_state,         # dict with oracle_regime, oracle_timing, etc.
    remaining_secs=remaining,    # seconds left in 5-min market
    bid_data=bid_trajectory,     # dict with bid_slope, current_bid, spread
)

# Log always
log_signal(mlp=mlp_result)

# Gate (when ready to go live):
if mlp_result["mlp_prob"] < 0.5:
    skip("MLP disagrees")
```

### 3. Required signal fields
- `divergence_pct`, `confidence`, `side`, `vpin`, `twap`, `momentum`
- `liquidation.chaos`, `liquidation.direction`, `liquidation.volume_usd`,
  `liquidation.long_liqs`, `liquidation.short_liqs`
- `remaining_secs` (0-300)
- `bid_data.bid_slope`, `bid_data.current_bid`, `bid_data.spread`
- `oracle.oracle_regime`, `oracle.oracle_timing`, `oracle.oracle_risk`,
  `oracle.confidence_adjusted`, `oracle.oracle_modifier`

### 4. Shadow mode
Log predictions without acting on them. After 200+ trades, compare:
- Win rate when MLP agreed vs disagreed
- Would PnL improve with the filter?

## Retraining
Export script: `/tmp/ANE-fork/btc_signal/export_training_data.py`
Retrain on Gilgamesh when new data accumulates (every few weeks).
The data pipeline auto-labels by joining signals against trade outcomes.
