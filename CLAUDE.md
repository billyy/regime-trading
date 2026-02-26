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

**Data flow:** yfinance fetch → feature engineering → HMM training → regime labels + confidence → technical indicators → entry/exit signals → backtest → performance metrics

### Core Modules

- **`regime_engine.py`** — HMM training and regime detection. Trains a 7-state Gaussian HMM on features (log returns, price range, volume volatility, historical volatility). Maps HMM states to 7 regimes (Crash → Bull Run) by sorting on mean return. Supports Walk-Forward Optimization (WFO) with daily retraining on sliding windows. Includes lookback period hyperparameter sweep maximizing 30-day Sharpe ratio.

- **`strategy.py`** — Signal generation and backtesting. Computes 8 technical indicators (RSI, Momentum, ATR, Volume trend, ADX, SMA, MACD). Entry requires Bullish/Bull Run regime AND ≥7/8 indicator conditions met. Exits on Bearish/Crash regimes. Dynamic leverage scales by HMM confidence (max 2.5x default). 48-hour cooldown after exits.

- **`app.py`** — Streamlit dashboard UI. Configures ticker, period, HMM components, leverage. Visualizes regime timeline, equity curves, trade logs. Uses `st.session_state` for persistence.

- **`mcp_server.py`** — MCP server exposing 4 tools: `get_recommendation`, `get_recommendation_wfo`, `run_backtest`, `sweep_lookback`. Wraps regime_engine and strategy modules for Claude Code integration.

## Key Parameters

| Parameter | Range | Default |
|-----------|-------|---------|
| Lookback period | 60–1460 days | 730 |
| HMM components | 3–10 states | 7 |
| Max leverage | 1.0–10.0x | 2.5x |
| WFO training window | 180–730 days | — |
| Entry threshold | — | 7 of 8 conditions |
| Cooldown after exit | — | 48 hours |

## Dependencies

Python 3.x with pip. Key packages: streamlit, yfinance, hmmlearn, scikit-learn, pandas, numpy, plotly, ta, mcp. Install via `pip install -r requirements.txt`.
