"""
Gaussian Hidden Markov Model regime detection engine.

Trains a 7-component HMM on hourly price data to classify market regimes
(Bull Run, Bullish, Neutral, Bearish, Crash, etc.).

Supports Walk-Forward Optimization (WFO) for true out-of-sample evaluation,
historical volatility as a 4th HMM feature, and automated hyperparameter sweep.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)

REGIME_COLORS = {
    "Bull Run": "#00c853",
    "Bullish": "#66bb6a",
    "Mild Bull": "#a5d6a7",
    "Neutral": "#9e9e9e",
    "Mild Bear": "#ef9a9a",
    "Bearish": "#ef5350",
    "Crash": "#b71c1c",
}

REGIME_NAMES_ORDERED = [
    "Crash", "Bearish", "Mild Bear", "Neutral", "Mild Bull", "Bullish", "Bull Run"
]


def fetch_data(ticker: str, period_days: int = 730) -> pd.DataFrame:
    """Download hourly OHLCV data for the given ticker."""
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=period_days)
    # yfinance caps hourly data at 730 days; download in chunks of 59 days
    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=59), end)
        chunk = yf.download(
            ticker,
            start=cursor.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
            interval="1h",
            progress=False,
            auto_adjust=True,
        )
        if not chunk.empty:
            chunks.append(chunk)
        cursor = chunk_end
    if not chunks:
        raise ValueError(f"No data returned for {ticker}.")
    df = pd.concat(chunks).sort_index()
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[~df.index.duplicated(keep="first")]

    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute HMM input features: log returns, price range, volume volatility, and historical volatility."""
    feat = pd.DataFrame(index=df.index)
    feat["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    feat["price_range"] = (df["High"] - df["Low"]) / df["Close"]
    feat["volume_vol"] = df["Volume"].rolling(24).std() / (df["Volume"].rolling(24).mean() + 1e-10)
    feat["hist_vol"] = feat["log_return"].rolling(20).std()
    feat.dropna(inplace=True)
    return feat


def train_hmm(features: pd.DataFrame, n_components: int = 7, n_iter: int = 200, n_seeds: int = 6) -> tuple:
    """Train a Gaussian HMM and return (model, scaler, state assignments)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    best_model = None
    best_score = -np.inf
    for seed in range(n_seeds):
        model = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=n_iter,
            random_state=seed,
            verbose=False,
        )
        model.fit(X)
        score = model.score(X)
        if score > best_score:
            best_score = score
            best_model = model

    states = best_model.predict(X)
    proba = best_model.predict_proba(X)
    return best_model, scaler, states, proba


def label_regimes(features: pd.DataFrame, states: np.ndarray) -> dict:
    """Map numeric state IDs to regime labels sorted by mean log return."""
    state_returns = {}
    for s in np.unique(states):
        mask = states == s
        state_returns[s] = features["log_return"].values[mask].mean()

    sorted_states = sorted(state_returns, key=state_returns.get)
    mapping = {}
    for rank, state_id in enumerate(sorted_states):
        mapping[state_id] = REGIME_NAMES_ORDERED[rank]
    return mapping


def build_regime_df(df: pd.DataFrame, features: pd.DataFrame, states: np.ndarray, proba: np.ndarray, mapping: dict) -> pd.DataFrame:
    """Merge regime labels and confidence back onto the price DataFrame."""
    regime_series = pd.Series(states, index=features.index, name="state_id")
    regime_labels = regime_series.map(mapping).rename("regime")
    # Confidence = posterior probability of the assigned state for each bar
    confidence = np.array([proba[i, states[i]] for i in range(len(states))])
    merged = df.loc[features.index].copy()
    merged["state_id"] = regime_series
    merged["regime"] = regime_labels
    merged["confidence"] = confidence
    return merged


def run_regime_detection(ticker: str, period_days: int = 730, n_components: int = 7):
    """Full pipeline: fetch → features → HMM → labelled DataFrame."""
    df = fetch_data(ticker, period_days)
    features = compute_features(df)
    model, scaler, states, proba = train_hmm(features, n_components)
    mapping = label_regimes(features, states)
    regime_df = build_regime_df(df, features, states, proba, mapping)
    return regime_df, model, scaler, mapping, features


def run_regime_detection_wfo(
    ticker,
    period_days=730,
    n_components=7,
    train_window_days=365,
    n_seeds=3,
    progress_callback=None,
):
    """
    Walk-Forward Optimization pipeline.

    Retrains the HMM daily on a sliding window, predicting the *next* day's bars
    (true out-of-sample evaluation).

    Returns (regime_df, features) — no model/scaler/mapping since they change each day.
    """
    df = fetch_data(ticker, period_days)
    features = compute_features(df)

    # Get unique calendar dates from the feature index
    dates = features.index.normalize().unique().sort_values()

    if len(dates) <= train_window_days:
        raise ValueError(
            f"Not enough data for WFO: {len(dates)} days available, "
            f"but training window is {train_window_days} days. "
            f"Increase period_days or decrease training window."
        )

    # Collect out-of-sample predictions
    oos_records = []
    total_steps = len(dates) - train_window_days
    completed = 0

    for i in range(train_window_days, len(dates)):
        # Training window: dates[i - train_window_days] to dates[i-1]
        train_start = dates[i - train_window_days]
        train_end = dates[i - 1]
        eval_date = dates[i]

        # Select training features (all bars in training date range)
        train_mask = (features.index.normalize() >= train_start) & (features.index.normalize() <= train_end)
        train_feat = features[train_mask]

        # Select evaluation features (bars on the eval date)
        eval_mask = features.index.normalize() == eval_date
        eval_feat = features[eval_mask]

        if train_feat.empty or eval_feat.empty:
            completed += 1
            if progress_callback:
                progress_callback(completed / total_steps)
            continue

        try:
            model, scaler, train_states, _ = train_hmm(train_feat, n_components, n_seeds=n_seeds)
            mapping = label_regimes(train_feat, train_states)

            # Predict on evaluation day (out-of-sample)
            X_eval = scaler.transform(eval_feat.values)
            eval_states = model.predict(X_eval)
            eval_proba = model.predict_proba(X_eval)

            for j in range(len(eval_feat)):
                state_id = eval_states[j]
                confidence = eval_proba[j, state_id]
                regime = mapping.get(state_id, "Neutral")
                oos_records.append({
                    "timestamp": eval_feat.index[j],
                    "state_id": state_id,
                    "regime": regime,
                    "confidence": confidence,
                })
        except Exception:
            # If HMM fails on this window, skip
            pass

        completed += 1
        if progress_callback:
            progress_callback(completed / total_steps)

    if not oos_records:
        raise ValueError("WFO produced no out-of-sample predictions.")

    # Assemble regime_df with same schema as non-WFO path
    oos_df = pd.DataFrame(oos_records).set_index("timestamp")
    regime_df = df.loc[oos_df.index].copy()
    regime_df["state_id"] = oos_df["state_id"]
    regime_df["regime"] = oos_df["regime"]
    regime_df["confidence"] = oos_df["confidence"]

    return regime_df, features


def compute_sharpe_from_equity(equity_series, last_n_days=30):
    """
    Compute annualized Sharpe ratio from the last N days of an hourly equity curve.

    Resamples to daily, takes last N days, computes daily returns, then Sharpe.
    """
    daily_equity = equity_series.resample("1D").last().dropna()
    if len(daily_equity) < 2:
        return 0.0
    tail = daily_equity.iloc[-last_n_days:]
    daily_returns = tail.pct_change().dropna()
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(365))


def sweep_lookback_periods(
    ticker,
    n_components=7,
    leverage=2.5,
    dynamic_leverage=True,
    candidates=None,
    progress_callback=None,
):
    """
    Sweep over lookback periods to find the one maximizing 30-day Sharpe ratio.

    Returns (best_lookback, best_sharpe, all_results).
    """
    # Import here to avoid circular imports
    from strategy import compute_indicators, backtest

    if candidates is None:
        candidates = list(range(60, 721, 30)) + [730]

    all_results = []
    total = len(candidates)

    for idx, lookback in enumerate(candidates):
        try:
            regime_df, _, _, _, _ = run_regime_detection(ticker, lookback, n_components)
            regime_df = compute_indicators(regime_df)
            result_df, trade_log = backtest(regime_df, leverage=leverage, dynamic_leverage=dynamic_leverage)
            sharpe = compute_sharpe_from_equity(result_df["equity"])
            all_results.append({
                "lookback_days": lookback,
                "sharpe_30d": round(sharpe, 4),
                "total_return_pct": round((result_df["equity"].iloc[-1] / result_df["equity"].iloc[0] - 1) * 100, 2),
                "num_trades": len(trade_log),
            })
        except Exception as e:
            all_results.append({
                "lookback_days": lookback,
                "sharpe_30d": float("nan"),
                "total_return_pct": float("nan"),
                "num_trades": 0,
                "error": str(e),
            })

        if progress_callback:
            progress_callback((idx + 1) / total)

    # Find best
    valid = [r for r in all_results if not np.isnan(r["sharpe_30d"])]
    if not valid:
        return None, float("nan"), all_results

    best = max(valid, key=lambda r: r["sharpe_30d"])
    return best["lookback_days"], best["sharpe_30d"], all_results
