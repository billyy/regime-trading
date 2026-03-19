"""
Strategy logic: technical indicator conditions, entry/exit rules, and risk management.

Entry: regime == 'Bullish' or 'Bull Run' AND >= ENTRY_THRESHOLD/8 conditions met.
Exit:  regime flips to 'Bearish' or 'Crash', or stop-loss hit, or margin call.
Risk:  dynamic leverage, configurable stop-loss, cooldown after exit.
"""

import numpy as np
import pandas as pd
import ta

from config import (
    DEFAULT_LEVERAGE,
    COOLDOWN_HOURS,
    ENTRY_THRESHOLD,
    RSI_OVERBOUGHT,
    STOP_LOSS_PCT,
    MIN_CONFIDENCE,
    ATR_MEDIAN_WINDOW,
    MIN_BARS_AFTER_WARMUP,
)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all required technical indicators to the DataFrame."""
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"]

    # RSI (14-period)
    out["RSI"] = ta.momentum.rsi(close, window=14)

    # Momentum (10-period rate of change)
    out["Momentum"] = ta.momentum.roc(close, window=10)

    # Volatility (ATR-based, 14-period)
    atr = ta.volatility.average_true_range(high, low, close, window=14)
    out["ATR"] = atr
    out["ATR_pct"] = atr / close
    # "Low volatility" = ATR% below its rolling median
    out["ATR_median"] = out["ATR_pct"].rolling(ATR_MEDIAN_WINDOW).median()

    # Volume trend (24-period SMA of volume)
    out["Vol_SMA"] = volume.rolling(24).mean()

    # ADX (14-period)
    out["ADX"] = ta.trend.adx(high, low, close, window=14)
    out["ADX_pos"] = ta.trend.adx_pos(high, low, close, window=14)
    out["ADX_neg"] = ta.trend.adx_neg(high, low, close, window=14)

    # SMA (50-period)
    out["SMA_50"] = ta.trend.sma_indicator(close, window=50)

    # MACD
    macd = ta.trend.MACD(close)
    out["MACD_line"] = macd.macd()
    out["MACD_signal"] = macd.macd_signal()
    out["MACD_hist"] = macd.macd_diff()

    return out


def evaluate_conditions(row: pd.Series) -> dict:
    """Evaluate the 8 entry conditions for a single bar. Returns dict of condition → bool."""
    return {
        f"RSI < {RSI_OVERBOUGHT}": row["RSI"] < RSI_OVERBOUGHT,
        "Positive Momentum": row["Momentum"] > 0,
        "Low Volatility": row["ATR_pct"] < row["ATR_median"],
        "Increasing Volume": row["Volume"] > row["Vol_SMA"],
        "Positive ADX": row["ADX_pos"] > row["ADX_neg"],
        "Price > SMA": row["Close"] > row["SMA_50"],
        "Bullish MACD": row["MACD_hist"] > 0,
        "ADX Strength": row["ADX"] > 20,
    }


def should_enter(row: pd.Series, entry_threshold: int = ENTRY_THRESHOLD) -> tuple[bool, dict]:
    """Return (enter, conditions_dict) based on regime + threshold rule."""
    if row["regime"] not in ("Bullish", "Bull Run"):
        return False, {}
    conds = evaluate_conditions(row)
    passed = sum(conds.values())
    return passed >= entry_threshold, conds


def should_exit(row: pd.Series) -> bool:
    """Exit if regime is Bearish or Crash."""
    return row["regime"] in ("Bearish", "Crash")


def backtest(
    df: pd.DataFrame,
    leverage: float = DEFAULT_LEVERAGE,
    dynamic_leverage: bool = True,
    entry_threshold: int = ENTRY_THRESHOLD,
    stop_loss_pct: float = STOP_LOSS_PCT,
    min_confidence: float = MIN_CONFIDENCE,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Run the strategy backtest on a regime-labelled, indicator-enriched DataFrame.

    Args:
        leverage: Maximum leverage multiplier.
        dynamic_leverage: Scale leverage by HMM confidence when True.
        entry_threshold: Minimum number of conditions (out of 8) to enter.
        stop_loss_pct: Force exit when leveraged loss exceeds this fraction (e.g. 0.10 = 10%).
            Set to None to disable stop-loss.
        min_confidence: Minimum HMM confidence required to enter a trade.

    Returns (equity_df, trade_log).
    """
    df = df.dropna().copy()

    if len(df) < MIN_BARS_AFTER_WARMUP:
        raise ValueError(
            f"Only {len(df)} bars remain after dropping NaN (need {MIN_BARS_AFTER_WARMUP}). "
            f"Increase period_days or check data quality."
        )

    equity = 1.0
    position = False
    entry_price = 0.0
    entry_time = None
    entry_leverage = 0.0
    entry_confidence = 0.0
    cooldown_until = None
    blown_up = False
    trade_log = []

    equities = []
    signals = []

    for i in range(len(df)):
        row = df.iloc[i]
        ts = df.index[i]

        # If equity hit zero (margin call), stay at zero
        if blown_up:
            equities.append(0.0)
            signals.append("BLOWN_UP")
            continue

        # Check cooldown
        in_cooldown = cooldown_until is not None and ts < cooldown_until

        if position:
            # Mark-to-market using the leverage locked at entry
            pnl_pct = (row["Close"] - entry_price) / entry_price
            current_equity = equity * (1 + pnl_pct * entry_leverage)

            # Negative equity protection (margin call)
            if current_equity <= 0:
                equity = 0.0
                trade_log.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "entry_price": entry_price,
                    "exit_price": row["Close"],
                    "return_pct": -100.0,
                    "regime_at_exit": "Margin Call",
                    "leverage": round(entry_leverage, 2),
                    "confidence": round(entry_confidence * 100, 1),
                })
                position = False
                blown_up = True
                equities.append(0.0)
                signals.append("EXIT")
                continue

            # Stop-loss check
            if stop_loss_pct is not None and pnl_pct * entry_leverage <= -stop_loss_pct:
                exit_return = pnl_pct * entry_leverage
                equity *= (1 + exit_return)
                trade_log.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "entry_price": entry_price,
                    "exit_price": row["Close"],
                    "return_pct": exit_return * 100,
                    "regime_at_exit": "Stop Loss",
                    "leverage": round(entry_leverage, 2),
                    "confidence": round(entry_confidence * 100, 1),
                })
                position = False
                cooldown_until = ts + pd.Timedelta(hours=COOLDOWN_HOURS)
                equities.append(equity)
                signals.append("EXIT")
                continue

            if should_exit(row):
                # Close position on regime change
                exit_return = pnl_pct * entry_leverage
                equity *= (1 + exit_return)
                trade_log.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "entry_price": entry_price,
                    "exit_price": row["Close"],
                    "return_pct": exit_return * 100,
                    "regime_at_exit": row["regime"],
                    "leverage": round(entry_leverage, 2),
                    "confidence": round(entry_confidence * 100, 1),
                })
                position = False
                cooldown_until = ts + pd.Timedelta(hours=COOLDOWN_HOURS)
                equities.append(equity)
                signals.append("EXIT")
            else:
                equities.append(current_equity)
                signals.append("LONG")
        else:
            if not in_cooldown:
                enter, conds = should_enter(row, entry_threshold=entry_threshold)
                if enter:
                    confidence = row.get("confidence", 1.0)
                    # Minimum confidence gate
                    if confidence >= min_confidence:
                        position = True
                        entry_price = row["Close"]
                        entry_time = ts
                        entry_confidence = confidence
                        entry_leverage = leverage * confidence if dynamic_leverage else leverage
                        equities.append(equity)
                        signals.append("ENTER")
                    else:
                        equities.append(equity)
                        signals.append("CASH")
                else:
                    equities.append(equity)
                    signals.append("CASH")
            else:
                equities.append(equity)
                signals.append("COOLDOWN")

    # Close any open position at end
    if position:
        last = df.iloc[-1]
        pnl_pct = (last["Close"] - entry_price) / entry_price
        exit_return = pnl_pct * entry_leverage
        equity *= (1 + exit_return)
        # Floor at zero
        if equity < 0:
            equity = 0.0
        trade_log.append({
            "entry_time": entry_time,
            "exit_time": df.index[-1],
            "entry_price": entry_price,
            "exit_price": last["Close"],
            "return_pct": exit_return * 100,
            "regime_at_exit": "End of Data",
            "leverage": round(entry_leverage, 2),
            "confidence": round(entry_confidence * 100, 1),
        })
        equities[-1] = equity
        signals[-1] = "EXIT"

    df["equity"] = equities
    df["signal"] = signals
    return df, trade_log


def performance_metrics(df: pd.DataFrame, trade_log: list[dict]) -> dict:
    """Compute summary performance statistics."""
    total_return = (df["equity"].iloc[-1] / df["equity"].iloc[0] - 1) * 100

    # Buy & hold return
    bh_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

    alpha = total_return - bh_return

    wins = [t for t in trade_log if t["return_pct"] > 0]
    win_rate = len(wins) / len(trade_log) * 100 if trade_log else 0

    # Max drawdown from equity curve
    cummax = df["equity"].cummax()
    drawdown = (df["equity"] - cummax) / cummax
    max_dd = drawdown.min() * 100

    return {
        "Total Return (%)": round(total_return, 2),
        "Buy & Hold Return (%)": round(bh_return, 2),
        "Alpha (%)": round(alpha, 2),
        "Win Rate (%)": round(win_rate, 2),
        "Max Drawdown (%)": round(max_dd, 2),
        "Total Trades": len(trade_log),
        "Winning Trades": len(wins),
        "Losing Trades": len(trade_log) - len(wins),
    }
