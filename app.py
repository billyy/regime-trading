"""
Streamlit dashboard for the regime-based trading system.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from regime_engine import (
    run_regime_detection,
    run_regime_detection_wfo,
    sweep_lookback_periods,
    REGIME_COLORS,
)
from strategy import compute_indicators, backtest, performance_metrics, evaluate_conditions, COOLDOWN_HOURS

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Regime Trading System", layout="wide", page_icon="📈")

st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #3a3a5c;
        text-align: center;
    }
    .metric-card h3 { margin: 0; color: #9e9e9e; font-size: 0.85rem; }
    .metric-card p  { margin: 0.3rem 0 0; font-size: 1.6rem; font-weight: 700; }
    .green { color: #00c853; }
    .red   { color: #ff1744; }
    .gray  { color: #bdbdbd; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="BTC-USD")

# Use session_state to allow sweep to auto-set period_days
if "optimal_lookback" not in st.session_state:
    st.session_state["optimal_lookback"] = None

default_period = st.session_state["optimal_lookback"] or 730
period_days = st.sidebar.slider("Training Period (days)", 60, 1460, default_period)
n_components = st.sidebar.slider("HMM Components", 3, 10, 7)
leverage = st.sidebar.slider("Max Leverage", 1.0, 10.0, 2.5, step=0.5)
dynamic_leverage = st.sidebar.checkbox("Dynamic Leverage (scale by HMM confidence)", value=True)

# ── Walk-Forward Optimization section ────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Walk-Forward Optimization")
use_wfo = st.sidebar.checkbox("Enable Walk-Forward (true out-of-sample)", value=False)

if use_wfo:
    train_window = st.sidebar.slider("WFO Training Window (days)", 180, 730, 365)
    wfo_seeds = st.sidebar.slider("WFO Seeds", 1, 6, 3)
    eval_days = period_days - train_window
    st.sidebar.caption(
        f"Total data: **{period_days}** days | "
        f"Training: **{train_window}** days | "
        f"Evaluation: **~{max(eval_days, 0)}** days"
    )
    if period_days <= train_window:
        st.sidebar.warning("Period must be greater than training window for WFO to work.")

run_btn = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ── Sweep button ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Hyperparameter Sweep")
sweep_btn = st.sidebar.button("🔍 Find Optimal Lookback", use_container_width=True)

st.sidebar.markdown("---")
lev_mode = "Dynamic (confidence-scaled)" if dynamic_leverage else f"Static **{leverage}x**"
st.sidebar.markdown(
    f"**Strategy Parameters**\n"
    f"- Max Leverage: **{leverage}x**\n"
    f"- Mode: {lev_mode}\n"
    f"- Cooldown: **{COOLDOWN_HOURS}h**\n"
    f"- Entry rule: **7 of 8** conditions\n"
    f"- Exit: regime → Bear / Crash"
)

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("📊 Regime-Based Trading System")
st.caption("Gaussian HMM regime detection · Multi-indicator confirmation · Leveraged strategy")

if run_btn:
    # ── Step 1: Regime detection ─────────────────────────────────────────────
    if use_wfo:
        st.info(f"Running Walk-Forward Optimization: {train_window}-day training window, {wfo_seeds} seeds")
        progress_bar = st.progress(0, text="Walk-forward optimization in progress...")

        def wfo_progress(pct):
            progress_bar.progress(min(pct, 1.0), text=f"WFO progress: {pct*100:.0f}%")

        regime_df, features = run_regime_detection_wfo(
            ticker, period_days, n_components,
            train_window_days=train_window,
            n_seeds=wfo_seeds,
            progress_callback=wfo_progress,
        )
        progress_bar.progress(1.0, text="Walk-forward optimization complete!")
        regime_df = compute_indicators(regime_df)
    else:
        with st.spinner("Fetching data and training HMM…"):
            regime_df, model, scaler, mapping, features = run_regime_detection(
                ticker, period_days, n_components
            )
            regime_df = compute_indicators(regime_df)

    # ── Step 2: Backtest ─────────────────────────────────────────────────────
    with st.spinner("Running backtest…"):
        result_df, trade_log = backtest(regime_df, leverage=leverage, dynamic_leverage=dynamic_leverage)
        metrics = performance_metrics(result_df, trade_log)

    # ── Current signal ───────────────────────────────────────────────────────
    last_row = result_df.iloc[-1]
    current_regime = last_row["regime"]
    current_signal = last_row["signal"]

    signal_color = {"LONG": "green", "ENTER": "green", "EXIT": "red", "CASH": "gray", "COOLDOWN": "gray"}
    display_signal = "LONG" if current_signal in ("LONG", "ENTER") else ("SHORT / EXIT" if current_signal == "EXIT" else "CASH")

    current_confidence = last_row.get("confidence", 1.0)
    effective_lev = leverage * current_confidence if dynamic_leverage else leverage

    st.markdown("---")
    col_sig, col_reg, col_price, col_conf, col_lev = st.columns(5)
    with col_sig:
        clr = signal_color.get(current_signal, "gray")
        st.markdown(
            f'<div class="metric-card"><h3>Current Signal</h3>'
            f'<p class="{clr}">{display_signal}</p></div>',
            unsafe_allow_html=True,
        )
    with col_reg:
        rc = REGIME_COLORS.get(current_regime, "#9e9e9e")
        st.markdown(
            f'<div class="metric-card"><h3>Current Regime</h3>'
            f'<p style="color:{rc}">{current_regime}</p></div>',
            unsafe_allow_html=True,
        )
    with col_price:
        st.markdown(
            f'<div class="metric-card"><h3>{ticker} Price</h3>'
            f'<p class="gray">${last_row["Close"]:,.2f}</p></div>',
            unsafe_allow_html=True,
        )
    with col_conf:
        st.markdown(
            f'<div class="metric-card"><h3>HMM Confidence</h3>'
            f'<p class="gray">{current_confidence * 100:.1f}%</p></div>',
            unsafe_allow_html=True,
        )
    with col_lev:
        lev_label = f"{effective_lev:.2f}x" if dynamic_leverage else f"{leverage:.1f}x (static)"
        st.markdown(
            f'<div class="metric-card"><h3>Effective Leverage</h3>'
            f'<p class="gray">{lev_label}</p></div>',
            unsafe_allow_html=True,
        )

    # ── WFO badge ────────────────────────────────────────────────────────────
    if use_wfo:
        st.success(f"Walk-Forward results: {len(result_df)} out-of-sample bars across ~{period_days - train_window} evaluation days")

    # ── Performance metrics row ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        clr = "green" if metrics["Total Return (%)"] >= 0 else "red"
        st.markdown(
            f'<div class="metric-card"><h3>Total Return</h3>'
            f'<p class="{clr}">{metrics["Total Return (%)"]:+.2f}%</p></div>',
            unsafe_allow_html=True,
        )
    with m2:
        clr = "green" if metrics["Alpha (%)"] >= 0 else "red"
        st.markdown(
            f'<div class="metric-card"><h3>Alpha vs Buy & Hold</h3>'
            f'<p class="{clr}">{metrics["Alpha (%)"]:+.2f}%</p></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><h3>Win Rate</h3>'
            f'<p class="gray">{metrics["Win Rate (%)"]:.1f}%</p></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-card"><h3>Max Drawdown</h3>'
            f'<p class="red">{metrics["Max Drawdown (%)"]:.2f}%</p></div>',
            unsafe_allow_html=True,
        )

    # ── Price chart with regime overlay ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Price Chart with HMM Regime Overlay")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=("Price & Regimes", "Equity Curve", "Regime Timeline"),
    )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=result_df.index, y=result_df["Close"],
            mode="lines", name="Price",
            line=dict(color="#42a5f5", width=1.2),
        ),
        row=1, col=1,
    )

    # Regime background shading
    for regime_name, color in REGIME_COLORS.items():
        mask = result_df["regime"] == regime_name
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=result_df.index[mask], y=result_df["Close"][mask],
                    mode="markers", name=regime_name,
                    marker=dict(color=color, size=3, opacity=0.6),
                    showlegend=True,
                ),
                row=1, col=1,
            )

    # Entry / exit markers
    entries = result_df[result_df["signal"] == "ENTER"]
    exits = result_df[result_df["signal"] == "EXIT"]
    fig.add_trace(
        go.Scatter(
            x=entries.index, y=entries["Close"],
            mode="markers", name="Entry",
            marker=dict(symbol="triangle-up", size=10, color="#00c853"),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=exits.index, y=exits["Close"],
            mode="markers", name="Exit",
            marker=dict(symbol="triangle-down", size=10, color="#ff1744"),
        ),
        row=1, col=1,
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=result_df.index, y=result_df["equity"],
            mode="lines", name="Equity",
            line=dict(color="#ffd600", width=1.5),
        ),
        row=2, col=1,
    )

    # Regime timeline (numeric encoding)
    regime_to_num = {name: i for i, name in enumerate(
        ["Crash", "Bearish", "Mild Bear", "Neutral", "Mild Bull", "Bullish", "Bull Run"]
    )}
    result_df["regime_num"] = result_df["regime"].map(regime_to_num)
    regime_colors_mapped = result_df["regime"].map(REGIME_COLORS)
    fig.add_trace(
        go.Bar(
            x=result_df.index, y=result_df["regime_num"],
            name="Regime",
            marker=dict(color=regime_colors_mapped.tolist()),
            showlegend=False,
        ),
        row=3, col=1,
    )

    fig.update_layout(
        height=850,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=2, col=1)
    fig.update_yaxes(title_text="Regime", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ── Confirmation checklist ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Entry Condition Checklist (Current Bar)")

    if not result_df.dropna().empty:
        latest = result_df.dropna().iloc[-1]
        conds = evaluate_conditions(latest)
        passed = sum(conds.values())

        cols = st.columns(4)
        for idx, (cond_name, cond_met) in enumerate(conds.items()):
            with cols[idx % 4]:
                icon = "✅" if cond_met else "❌"
                st.markdown(f"**{icon} {cond_name}**")

        st.markdown(f"**Conditions met: {passed} / 8** (need ≥ 7)")

    # ── Trade log ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Trade Log")

    if trade_log:
        tl_df = pd.DataFrame(trade_log)
        tl_df["entry_time"] = pd.to_datetime(tl_df["entry_time"])
        tl_df["exit_time"] = pd.to_datetime(tl_df["exit_time"])
        tl_df["duration"] = tl_df["exit_time"] - tl_df["entry_time"]
        tl_df["return_pct"] = tl_df["return_pct"].round(2)
        tl_df = tl_df.rename(columns={
            "entry_time": "Entry Time",
            "exit_time": "Exit Time",
            "entry_price": "Entry Price",
            "exit_price": "Exit Price",
            "return_pct": "Return (%)",
            "regime_at_exit": "Exit Regime",
            "duration": "Duration",
            "leverage": "Leverage",
            "confidence": "Confidence (%)",
        })
        st.dataframe(tl_df, use_container_width=True, hide_index=True)
    else:
        st.info("No trades triggered during this period.")

    # ── Regime distribution ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Regime Distribution")
    regime_counts = result_df["regime"].value_counts()
    fig_pie = go.Figure(
        go.Pie(
            labels=regime_counts.index,
            values=regime_counts.values,
            marker=dict(colors=[REGIME_COLORS.get(r, "#9e9e9e") for r in regime_counts.index]),
            hole=0.4,
        )
    )
    fig_pie.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        height=400,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

elif sweep_btn:
    # ── Hyperparameter Sweep ─────────────────────────────────────────────────
    st.subheader("🔍 Hyperparameter Sweep: Optimal Lookback Period")
    st.caption("Testing lookback periods from 60 to 730 days, optimizing 30-day Sharpe ratio")

    progress_bar = st.progress(0, text="Starting sweep...")

    def sweep_progress(pct):
        progress_bar.progress(min(pct, 1.0), text=f"Sweep progress: {pct*100:.0f}%")

    best_lookback, best_sharpe, all_results = sweep_lookback_periods(
        ticker,
        n_components=n_components,
        leverage=leverage,
        dynamic_leverage=dynamic_leverage,
        progress_callback=sweep_progress,
    )
    progress_bar.progress(1.0, text="Sweep complete!")

    if best_lookback is not None:
        st.success(f"Optimal lookback: **{best_lookback} days** (30-day Sharpe: **{best_sharpe:.4f}**)")
        st.session_state["optimal_lookback"] = best_lookback

        # Results table
        results_df = pd.DataFrame(all_results).sort_values("sharpe_30d", ascending=False)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Sharpe vs lookback chart
        valid_results = [r for r in all_results if not np.isnan(r["sharpe_30d"])]
        if valid_results:
            vr_df = pd.DataFrame(valid_results).sort_values("lookback_days")
            fig_sweep = go.Figure()
            fig_sweep.add_trace(go.Scatter(
                x=vr_df["lookback_days"],
                y=vr_df["sharpe_30d"],
                mode="lines+markers",
                name="Sharpe Ratio",
                line=dict(color="#42a5f5", width=2),
                marker=dict(size=6),
            ))
            # Vertical line at best
            fig_sweep.add_vline(
                x=best_lookback, line_dash="dash", line_color="#00c853",
                annotation_text=f"Best: {best_lookback}d",
                annotation_position="top right",
            )
            fig_sweep.update_layout(
                title="30-Day Sharpe Ratio vs Lookback Period",
                xaxis_title="Lookback Period (days)",
                yaxis_title="Sharpe Ratio (30d)",
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                height=400,
            )
            st.plotly_chart(fig_sweep, use_container_width=True)

        st.info(f"Training Period slider has been updated to **{best_lookback} days**. Click **Run Analysis** to use the optimal setting.")
    else:
        st.error("Sweep failed — no valid results. Check that data is available for the ticker.")

else:
    st.info("Configure parameters in the sidebar and click **Run Analysis** to start.")
    st.markdown(
        """
        ### How it works
        1. **Data Fetch** — Downloads hourly OHLCV data via yfinance (+ VIX as 4th feature)
        2. **Feature Engineering** — Computes log returns, price range, volume volatility, and VIX
        3. **HMM Training** — Fits a 7-state Gaussian HMM to classify market regimes
        4. **Strategy Execution** — Enters long when Bullish + 7/8 indicator conditions met; exits on Bear/Crash
        5. **Risk Management** — 2.5x leverage with 48-hour cooldown after exits

        ### New Features
        - **Walk-Forward Optimization** — Retrain HMM daily on a sliding window for true out-of-sample evaluation
        - **VIX Feature** — Market fear gauge as 4th HMM input feature
        - **Hyperparameter Sweep** — Find optimal lookback period by maximizing 30-day Sharpe ratio
        """
    )
