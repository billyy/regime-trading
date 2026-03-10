#!/usr/bin/env python3
"""Batch sweep_lookback for all symbols, saving results to JSON."""
import json, time, sys
sys.path.insert(0, "/Users/billy/.openclaw/workspace/regime-trading")
from regime_engine import sweep_lookback_periods

SYMBOLS = [
    "NVDA", "AMD", "AMZN", "META", "MSFT", "GOOGL", "AAPL", "TSLA",
    "NFLX", "INTU", "TSM", "TMO", "UNH", "ORCL", "MA", "V",
    "CRM", "GDX", "SLV", "GLD", "QQQ", "SPY"
]

FAST_CANDIDATES = list(range(90, 361, 30))  # 10 candidates instead of 23

results = {}
for i, sym in enumerate(SYMBOLS):
    t0 = time.time()
    print(f"[{i+1}/22] Sweeping {sym}...", flush=True)
    try:
        best_days, best_sharpe, details = sweep_lookback_periods(sym, candidates=FAST_CANDIDATES)
        # Find the best entry for num_trades and total_return
        best_entry = next((d for d in details if d["lookback_days"] == best_days), {})
        results[sym] = {
            "optimal_lookback": best_days,
            "best_sharpe": best_sharpe,
            "total_return": best_entry.get("total_return_pct", 0),
            "num_trades": best_entry.get("num_trades", 0),
            "time_s": round(time.time() - t0, 1)
        }
        print(f"    {sym}: lookback={best_days}, sharpe={best_sharpe}, return={results[sym]['total_return']}%, trades={results[sym]['num_trades']} ({results[sym]['time_s']}s)", flush=True)
    except Exception as e:
        results[sym] = {"error": str(e)}
        print(f"    {sym}: ERROR - {e}", flush=True)

with open("/Users/billy/.openclaw/workspace/regime-trading/sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone! Results saved to sweep_results.json", flush=True)
