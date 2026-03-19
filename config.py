"""
Centralized configuration constants for the regime-trading system.
"""

# HMM defaults
DEFAULT_LOOKBACK = 730
DEFAULT_N_COMPONENTS = 7
N_SEEDS = 6
HMM_N_ITER = 200

# Strategy parameters
DEFAULT_LEVERAGE = 2.5
COOLDOWN_HOURS = 48
ENTRY_THRESHOLD = 6          # out of 8 conditions required for entry
RSI_OVERBOUGHT = 70
STOP_LOSS_PCT = 0.10         # 10% stop-loss
MIN_CONFIDENCE = 0.5         # minimum HMM confidence to enter a trade

# Indicator windows
ATR_MEDIAN_WINDOW = 168      # 1 week of hourly bars

# Data validation
MIN_BARS_AFTER_WARMUP = 200
