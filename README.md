# Regime Trading

A quantitative algorithmic trading system that uses **Gaussian Hidden Markov Models (HMM)** to detect market regimes and execute leveraged long-only strategies based on multi-indicator confirmation.

## How It Works

### Regime Detection

A 7-state Gaussian HMM is trained on hourly OHLCV data to classify market conditions into regimes:

| Regime | Interpretation |
|--------|---------------|
| Bull Run | Strongest bullish state |
| Bullish | Strong uptrend |
| Mild Bull | Weak uptrend |
| Neutral | No clear direction |
| Mild Bear | Weak downtrend |
| Bearish | Strong downtrend |
| Crash | Strongest bearish state |

States are assigned by rank-ordering HMM components by their mean log return. The model outputs both a regime label and a **confidence score** (posterior probability) for each hourly bar.

**HMM Features:** log returns, price range (high-low/close), volume volatility (24h rolling), historical volatility (20h rolling).

### Entry & Exit Rules

**Entry** requires two gates:
1. Regime must be **Bullish** or **Bull Run**
2. At least **7 of 8** technical indicator conditions must pass:

| Condition | Rule |
|-----------|------|
| RSI < 90 | Not overbought |
| Positive Momentum | 10-period ROC > 0 |
| Low Volatility | ATR% below 1-week rolling median |
| Increasing Volume | Volume > 24-period SMA |
| Positive ADX | +DI > -DI |
| Price > SMA | Close > 50-period SMA |
| Bullish MACD | MACD histogram > 0 |
| ADX Strength | ADX > 20 |

**Exit** triggers when regime becomes **Bearish** or **Crash**, followed by a 48-hour cooldown.

### Dynamic Leverage

When enabled, effective leverage scales with HMM confidence:

```
effective_leverage = max_leverage × confidence
```

Default max leverage is 2.5x. A regime detected at 80% confidence yields 2.0x effective leverage.

### Walk-Forward Optimization (WFO)

The standard model trains on the entire dataset (in-sample). WFO provides **out-of-sample** validation by retraining the HMM daily on a sliding window and predicting only the next day's bars. WFO results are generally more trustworthy as they avoid lookahead bias and overfitting.

> **Note:** In-sample and WFO can produce different regime labels for the same bar because `label_regimes()` rank-orders states relative to the training data. A wider history with extreme events compresses the scale; a narrower sliding window may rank the same conditions higher.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.x. Key dependencies: `hmmlearn`, `yfinance`, `streamlit`, `scikit-learn`, `pandas`, `ta`, `plotly`, `mcp`.

## Usage

### Streamlit Dashboard

```bash
streamlit run app.py
```

Interactive UI for configuring parameters, running analysis, visualizing regime timelines, equity curves, and trade logs.

### MCP Server (Claude Code / Claude Desktop)

The MCP server exposes 4 tools for AI-assisted analysis:

| Tool | Description |
|------|-------------|
| `get_recommendation` | Current regime, confidence, conditions checklist, and Long/Cash/Exit recommendation |
| `get_recommendation_wfo` | Same as above but using walk-forward optimization (out-of-sample) |
| `run_backtest` | Full backtest with performance metrics (return, alpha, Sharpe, drawdown, win rate) |
| `sweep_lookback` | Test lookback periods from 60–730 days to find optimal 30-day Sharpe |

#### Claude Code

The MCP server is configured in `.claude/settings.local.json` and available automatically.

#### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "regime-trading": {
      "command": "python3",
      "args": ["/path/to/regime-trading/mcp_server.py"]
    }
  }
}
```

Restart Claude Desktop after saving.

### Direct Python Usage

```python
from regime_engine import run_regime_detection, run_regime_detection_wfo
from strategy import compute_indicators, backtest, performance_metrics

# In-sample
regime_df, model, scaler, mapping, features = run_regime_detection("BTC-USD", period_days=730)
regime_df = compute_indicators(regime_df)
result_df, trade_log = backtest(regime_df)
metrics = performance_metrics(result_df, trade_log)

# Walk-forward (out-of-sample)
regime_df, features = run_regime_detection_wfo("BTC-USD", period_days=730, train_window_days=365)
regime_df = compute_indicators(regime_df)
result_df, trade_log = backtest(regime_df)
```

## Configuration

| Parameter | Range | Default |
|-----------|-------|---------|
| Lookback period | 60–730 days | 730 |
| HMM components | 3–10 | 7 |
| Max leverage | 1.0–10.0x | 2.5 |
| Dynamic leverage | on/off | on |
| WFO training window | 180–730 days | 365 |
| Cooldown after exit | — | 48 hours |
| Entry threshold | — | 7 of 8 conditions |

## Architecture

```
regime_engine.py    HMM training, regime detection, WFO pipeline, lookback sweep
strategy.py         Technical indicators, entry/exit signals, backtesting engine
app.py              Streamlit dashboard UI
mcp_server.py       MCP tool wrappers for Claude Code / Claude Desktop
```

**Data flow:**

```
yfinance (hourly OHLCV)
  → Feature engineering (log returns, range, vol)
    → HMM training → Regime labels + confidence
      → Technical indicators (RSI, MACD, ADX, etc.)
        → Entry/exit signal generation
          → Backtest with dynamic leverage
            → Performance metrics + equity curve + trade log
```
