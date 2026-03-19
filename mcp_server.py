"""
HMM Regime Detection MCP Server.

Exposes regime detection, walk-forward optimization, backtesting, and
hyperparameter sweep as MCP tools for Claude Code (or any MCP client).
"""

import sys
import os

# Ensure the project directory is on the path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

from config import DEFAULT_LEVERAGE, ENTRY_THRESHOLD
from regime_engine import (
    run_regime_detection,
    run_regime_detection_wfo,
    sweep_lookback_periods,
    compute_sharpe_from_equity,
)
from strategy import (
    compute_indicators,
    backtest,
    performance_metrics,
    evaluate_conditions,
)

mcp = FastMCP("regime-trading")


@mcp.tool()
def get_recommendation(
    ticker: str,
    period_days: int = 730,
    n_components: int = 7,
    leverage: float = DEFAULT_LEVERAGE,
    entry_threshold: int = ENTRY_THRESHOLD,
) -> str:
    """Get the current HMM regime and trading recommendation for a ticker.

    Runs the full HMM pipeline (fetch data, compute features, train model,
    classify regimes) and returns the current regime, confidence, entry
    conditions checklist, and a recommendation (Long / Cash / Exit).

    Args:
        ticker: Stock/crypto symbol (e.g. "SPY", "BTC-USD")
        period_days: Lookback period in days (max 730 for hourly data)
        n_components: Number of HMM states (default 7)
        leverage: Maximum leverage multiplier (default 2.5)
        entry_threshold: Minimum conditions to pass for entry (default 6 of 8)
    """
    try:
        regime_df, model, scaler, mapping, features = run_regime_detection(
            ticker, period_days, n_components
        )
        regime_df = compute_indicators(regime_df)
        last = regime_df.dropna().iloc[-1]

        regime = last["regime"]
        confidence = last["confidence"]
        price = last["Close"]

        # Evaluate entry conditions
        conds = evaluate_conditions(last)
        passed = sum(conds.values())
        total = len(conds)

        # Determine recommendation
        if regime in ("Bearish", "Crash"):
            recommendation = "EXIT / STAY CASH"
        elif regime in ("Bullish", "Bull Run") and passed >= entry_threshold:
            eff_lev = round(leverage * confidence, 2)
            recommendation = f"LONG (suggested leverage: {eff_lev}x)"
        else:
            recommendation = "CASH (conditions not met)"

        # Format conditions checklist
        cond_lines = []
        for name, met in conds.items():
            mark = "PASS" if met else "FAIL"
            cond_lines.append(f"  [{mark}] {name}")

        return "\n".join([
            f"=== {ticker} Regime Report ===",
            f"Current Price:  {price:.2f}",
            f"Regime:         {regime}",
            f"HMM Confidence: {confidence:.1%}",
            f"Recommendation: {recommendation}",
            f"",
            f"Entry Conditions ({passed}/{total} passed, need >= {entry_threshold}):",
            *cond_lines,
        ])

    except Exception as e:
        return f"Error analysing {ticker}: {e}"


@mcp.tool()
def get_recommendation_wfo(
    ticker: str,
    period_days: int = 730,
    n_components: int = 7,
    train_window_days: int = 365,
    n_seeds: int = 3,
    leverage: float = DEFAULT_LEVERAGE,
    entry_threshold: int = ENTRY_THRESHOLD,
) -> str:
    """Get regime recommendation using Walk-Forward Optimization (out-of-sample).

    Same as get_recommendation but retrains the HMM daily on a sliding window
    so the most recent regime call is truly out-of-sample. More robust but slower.

    Args:
        ticker: Stock/crypto symbol (e.g. "SPY", "BTC-USD")
        period_days: Total lookback period in days (max 730)
        n_components: Number of HMM states (default 7)
        train_window_days: Sliding training window in days
        n_seeds: Number of random seeds per HMM fit
        leverage: Maximum leverage multiplier (default 2.5)
        entry_threshold: Minimum conditions to pass for entry (default 6 of 8)
    """
    try:
        regime_df, features = run_regime_detection_wfo(
            ticker, period_days, n_components, train_window_days, n_seeds
        )
        regime_df = compute_indicators(regime_df)
        last = regime_df.dropna().iloc[-1]

        regime = last["regime"]
        confidence = last["confidence"]
        price = last["Close"]

        conds = evaluate_conditions(last)
        passed = sum(conds.values())
        total = len(conds)

        if regime in ("Bearish", "Crash"):
            recommendation = "EXIT / STAY CASH"
        elif regime in ("Bullish", "Bull Run") and passed >= entry_threshold:
            eff_lev = round(leverage * confidence, 2)
            recommendation = f"LONG (suggested leverage: {eff_lev}x)"
        else:
            recommendation = "CASH (conditions not met)"

        cond_lines = []
        for name, met in conds.items():
            mark = "PASS" if met else "FAIL"
            cond_lines.append(f"  [{mark}] {name}")

        return "\n".join([
            f"=== {ticker} WFO Regime Report ===",
            f"(Out-of-sample validated, train window: {train_window_days}d)",
            f"Current Price:  {price:.2f}",
            f"Regime:         {regime}",
            f"HMM Confidence: {confidence:.1%}",
            f"Recommendation: {recommendation}",
            f"",
            f"Entry Conditions ({passed}/{total} passed, need >= {entry_threshold}):",
            *cond_lines,
        ])

    except Exception as e:
        return f"Error analysing {ticker} (WFO): {e}"


@mcp.tool()
def run_backtest(
    ticker: str,
    period_days: int = 730,
    n_components: int = 7,
    leverage: float = 2.5,
    dynamic_leverage: bool = True,
    use_wfo: bool = False,
    train_window_days: int = 365,
) -> str:
    """Run a full backtest of the HMM regime strategy and return performance metrics.

    Fetches data, detects regimes, computes indicators, runs the backtest, and
    returns total return, alpha, win rate, max drawdown, Sharpe, and trade count.

    Args:
        ticker: Stock/crypto symbol (e.g. "SPY", "BTC-USD")
        period_days: Lookback period in days (max 730)
        n_components: Number of HMM states (default 7)
        leverage: Maximum leverage (default 2.5)
        dynamic_leverage: Scale leverage by HMM confidence (default True)
        use_wfo: Use walk-forward optimization instead of in-sample (default False)
        train_window_days: WFO training window in days (only used if use_wfo=True)
    """
    try:
        if use_wfo:
            regime_df, features = run_regime_detection_wfo(
                ticker, period_days, n_components, train_window_days
            )
        else:
            regime_df, model, scaler, mapping, features = run_regime_detection(
                ticker, period_days, n_components
            )

        regime_df = compute_indicators(regime_df)
        result_df, trade_log = backtest(
            regime_df, leverage=leverage, dynamic_leverage=dynamic_leverage
        )
        metrics = performance_metrics(result_df, trade_log)
        sharpe = compute_sharpe_from_equity(result_df["equity"])

        # Regime distribution
        regime_counts = regime_df["regime"].value_counts()
        regime_lines = []
        for r, count in regime_counts.items():
            pct = count / len(regime_df) * 100
            regime_lines.append(f"  {r}: {pct:.1f}%")

        mode_label = "WFO (out-of-sample)" if use_wfo else "In-sample"
        lev_label = f"dynamic (max {leverage}x)" if dynamic_leverage else f"static {leverage}x"

        return "\n".join([
            f"=== {ticker} Backtest Results ===",
            f"Mode:            {mode_label}",
            f"Lookback:        {period_days} days",
            f"Leverage:        {lev_label}",
            f"",
            f"Performance:",
            f"  Total Return:    {metrics['Total Return (%)']:.2f}%",
            f"  Buy & Hold:      {metrics['Buy & Hold Return (%)']:.2f}%",
            f"  Alpha:           {metrics['Alpha (%)']:.2f}%",
            f"  Win Rate:        {metrics['Win Rate (%)']:.1f}%",
            f"  Max Drawdown:    {metrics['Max Drawdown (%)']:.2f}%",
            f"  30-day Sharpe:   {sharpe:.4f}",
            f"",
            f"Trades:",
            f"  Total:   {metrics['Total Trades']}",
            f"  Wins:    {metrics['Winning Trades']}",
            f"  Losses:  {metrics['Losing Trades']}",
            f"",
            f"Regime Distribution:",
            *regime_lines,
        ])

    except Exception as e:
        return f"Error backtesting {ticker}: {e}"


@mcp.tool()
def sweep_lookback(
    ticker: str,
    n_components: int = 7,
    leverage: float = 2.5,
    dynamic_leverage: bool = True,
) -> str:
    """Sweep lookback periods to find the optimal one for a ticker.

    Tests lookback periods from 60 to 730 days (in 30-day steps) and returns
    the one that maximizes the 30-day Sharpe ratio.

    Args:
        ticker: Stock/crypto symbol (e.g. "SPY", "BTC-USD")
        n_components: Number of HMM states (default 7)
        leverage: Maximum leverage for backtest (default 2.5)
        dynamic_leverage: Scale leverage by HMM confidence (default True)
    """
    try:
        best_lookback, best_sharpe, all_results = sweep_lookback_periods(
            ticker, n_components, leverage, dynamic_leverage
        )

        # Format results table
        table_lines = [f"{'Lookback':>10} {'Sharpe':>10} {'Return':>10} {'Trades':>8}"]
        table_lines.append("-" * 42)
        for r in all_results:
            sharpe_str = f"{r['sharpe_30d']:.4f}" if r["sharpe_30d"] == r["sharpe_30d"] else "ERROR"
            ret_str = f"{r['total_return_pct']:.2f}%" if r["total_return_pct"] == r["total_return_pct"] else "N/A"
            marker = " <-- BEST" if r["lookback_days"] == best_lookback else ""
            table_lines.append(
                f"{r['lookback_days']:>8}d {sharpe_str:>10} {ret_str:>10} {r['num_trades']:>8}{marker}"
            )

        return "\n".join([
            f"=== {ticker} Lookback Sweep ===",
            f"Optimal Lookback: {best_lookback} days",
            f"Best 30-day Sharpe: {best_sharpe:.4f}",
            f"",
            *table_lines,
        ])

    except Exception as e:
        return f"Error sweeping {ticker}: {e}"


if __name__ == "__main__":
    mcp.run()
