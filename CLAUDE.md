# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Regime-Trading is a quantitative algorithmic trading system that uses Gaussian Hidden Markov Models (HMM) to detect market regimes and execute leveraged long-only strategies based on multi-indicator confirmation. It works with hourly OHLCV data fetched via yfinance.

## Running the Application

```bash
# Streamlit dashboard
streamlit run app.py

# MCP server (for Claude Code tool integration)
python mcp_server.py
```

There are no tests, linter, or build steps configured.

## Architecture

**Data flow:** yfinance fetch → data validation → feature engineering → HMM training → regime labels + confidence → technical indicators → entry/exit signals (with stop-loss, confidence gate) → backtest → performance metrics

### Core Modules

- **`config.py`** — Centralized configuration constants (DEFAULT_LOOKBACK, DEFAULT_LEVERAGE, ENTRY_THRESHOLD, RSI_OVERBOUGHT, STOP_LOSS_PCT, MIN_CONFIDENCE, etc.). All modules import from here.

- **`regime_engine.py`** — HMM training and regime detection. Dynamically generates regime labels for any n_components (3–10+), always ensuring Bullish/Bull Run map to top states and Bearish/Crash to bottom states. Validates fetched data for quality. Supports WFO with logged exceptions. Lookback sweep supports optional eval_start_date for normalized comparison.

- **`strategy.py`** — Signal generation and backtesting. Entry requires Bullish/Bull Run regime AND configurable threshold of conditions met, plus minimum HMM confidence. Exits on Bearish/Crash regimes, stop-loss, or margin call. Negative equity protection floors equity at zero. Asserts minimum bars after warmup.

- **`app.py`** — Streamlit dashboard UI with `st.cache_data` caching. Sidebar controls for entry threshold, stop-loss %, min confidence, plus existing ticker/period/leverage controls.

- **`mcp_server.py`** — MCP server exposing 4 tools with configurable leverage and entry threshold parameters.

## Key Parameters

| Parameter | Range | Default |
|-----------|-------|---------|
| Lookback period | 60–1460 days | 730 |
| HMM components | 3–10 states | 7 |
| Max leverage | 1.0–10.0x | 2.5x |
| WFO training window | 180–730 days | — |
| Entry threshold | 1–8 conditions | 6 of 8 |
| Stop-loss | 0–50% | 10% |
| Min confidence | 0.0–1.0 | 0.5 |
| RSI overbought | — | 70 |
| Cooldown after exit | — | 48 hours |

## Dependencies

Python 3.x with pip. Key packages: streamlit, yfinance, hmmlearn, scikit-learn, pandas, numpy, plotly, ta, mcp. Install via `pip install -r requirements.txt`.
