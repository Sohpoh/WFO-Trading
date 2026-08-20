import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from data_loader import (
    BARS_PER_YEAR,
    list_local_symbols,
    list_local_timeframes,
    load_data,
)
from metrics import equity_curve, summarize
from session import SESSION_CONFIG
from wfo_engine import build_grid, run_retail_insample, run_walk_forward

st.set_page_config(page_title="Walk-Forward Trading App", layout="wide")


def parse_num_list(text: str, cast=float) -> list:
    vals = sorted({cast(x.strip()) for x in text.split(",") if x.strip()})
    return vals


def fmt_pct(x: float) -> str:
    return f"{x * 100:,.2f}%"


def folds_to_gantt_df(folds) -> pd.DataFrame:
    rows = []
    for f in folds:
        if not f.best_params:
            continue
        label = f"Fold {f.index + 1}"
        rows.append({"Fold": label, "Phase": "Train (In-Sample)", "Start": f.train_start, "Finish": f.train_end})
        rows.append({"Fold": label, "Phase": "Test (Blind OOS)", "Start": f.test_start, "Finish": f.test_end})
    return pd.DataFrame(rows)


def folds_to_table(folds) -> pd.DataFrame:
    rows = []
    for f in folds:
        if not f.best_params:
            continue
        rows.append(
            {
                "Fold": f.index + 1,
                "Train Start": f.train_start.date(),
                "Train End": f.train_end.date(),
                "Test Start": f.test_start.date(),
                "Test End": f.test_end.date(),
                "Range Lookback": f.best_params.get("range_lookback"),
                "Buffer Frac": f.best_params.get("buffer_frac"),
                "Train Sharpe": round(f.train_sharpe, 3),
                "Test Bars": f.n_test_bars,
            }
        )
    return pd.DataFrame(rows)


def param_stability_caption(folds) -> str:
    """Fold-to-fold stability of the winning range_lookback/buffer_frac is a
    stronger robustness signal than any single fold's in-sample Sharpe."""
    chosen = [f.best_params for f in folds if f.best_params]
    if not chosen:
        return "No fold produced a valid parameter set."
    lookback_vals = sorted({p["range_lookback"] for p in chosen})
    buffer_vals = sorted({p["buffer_frac"] for p in chosen})
    return (
        f"Across {len(chosen)} folds: {len(lookback_vals)} distinct range lookback chosen ({lookback_vals}), "
        f"{len(buffer_vals)} distinct buffer frac chosen ({buffer_vals}). "
        f"Fewer distinct values = more stable parameters."
    )


# ---------------------------------------------------------------------------
# Sidebar — data source
# ---------------------------------------------------------------------------
st.sidebar.header("Data")
source_label = st.sidebar.radio("Data Source", ["Local Futures Data", "Yahoo Finance (yfinance)"], index=0)
source = "local" if source_label == "Local Futures Data" else "yfinance"

if source == "local":
    symbols = list_local_symbols()
    default_idx = symbols.index("NQ") if "NQ" in symbols else 0
    symbol = st.sidebar.selectbox("Symbol", symbols, index=default_idx)
    timeframes = list_local_timeframes(symbol)
    tf_default = timeframes.index("1h") if "1h" in timeframes else 0
    timeframe = st.sidebar.selectbox("Bar Size", timeframes, index=tf_default)
else:
    symbol = st.sidebar.text_input("Ticker", value="NQ=F")
    timeframe = st.sidebar.selectbox("Bar Size", ["1d", "1h"], index=0)
    st.sidebar.caption("Yahoo Finance caps intraday history at ~730 days; use 1d bars for long lookbacks.")

try:
    raw_df = load_data(source, symbol, timeframe)
except Exception as e:
    st.error(f"Could not load data for {symbol}: {e}")
    st.stop()

data_min, data_max = raw_df.index.min().date(), raw_df.index.max().date()
date_from, date_to = st.sidebar.date_input(
    "Date Range", value=(data_min, data_max), min_value=data_min, max_value=data_max
)
date_from_ts = pd.Timestamp(date_from, tz="UTC")
date_to_ts = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1)
df = raw_df[(raw_df.index >= date_from_ts) & (raw_df.index < date_to_ts)]
st.sidebar.caption(f"{len(df):,} bars loaded — {df.index.min().date()} to {df.index.max().date()}")

# ---------------------------------------------------------------------------
# Sidebar — strategy parameters
# ---------------------------------------------------------------------------
st.sidebar.header("Strategy: Rolling Range Breakout + Vol-Regime Gate + Failed-Breakout Stop")

session_names = list(SESSION_CONFIG.keys())
session_choice = st.sidebar.selectbox("Day-Trade Session", session_names, index=session_names.index("New York"))
st.sidebar.caption(
    f"Trades only enter during {session_choice} hours "
    f"({SESSION_CONFIG[session_choice]['start']}–{SESSION_CONFIG[session_choice]['end']} "
    f"{SESSION_CONFIG[session_choice]['tz']}) and are force-closed at session end — no overnight holds."
)
session = session_choice
if timeframe == "1d":
    st.sidebar.warning("Daily bars have one price per day, so session filtering doesn't apply — it's ignored at this bar size.")
    session = None

range_lookback_text = st.sidebar.text_input(
    "Breakout Channel Lookback (bars, comma-separated)", value="24, 48, 96, 192",
    help="Rolling high/low window (shifted one bar so the current bar can't define its own level).",
)
buffer_frac_text = st.sidebar.text_input(
    "Breakout Buffer (fraction of channel width, comma-separated)", value="0.0, 0.05, 0.10, 0.15",
    help="Required overshoot past the channel as a fraction of its width (0 = plain touch of the level).",
)
st.sidebar.caption(
    "A zero-parameter volatility-regime gate (96-bar ATR ≥ 384-bar ATR, hardcoded — not searched) "
    "suppresses entries while realized volatility is contracting."
)
st.sidebar.caption(
    "Exit is a failed-breakout stop at half the channel width (hardcoded, not searched): a breakout that "
    "gives back that much is cut. Otherwise the position rides to the opposite breakout or the session's "
    "forced flatten — stops are detected intrabar off High/Low but priced at that bar's Close, so they cap "
    "*when* a trade exits, not the realized loss on the triggering bar."
)

try:
    range_lookbacks = parse_num_list(range_lookback_text, int)
    buffer_fracs = parse_num_list(buffer_frac_text, float)
    grid = build_grid(range_lookbacks, buffer_fracs, session)
except ValueError:
    st.sidebar.error("Could not parse parameter lists — use comma-separated numbers.")
    st.stop()

if not grid:
    st.sidebar.error("Parameter grid is empty — provide at least one range lookback and buffer fraction.")
    st.stop()
st.sidebar.caption(f"Grid size: {len(grid)} range-lookback/buffer-frac parameter combinations per fold")

# ---------------------------------------------------------------------------
# Sidebar — walk-forward settings
# ---------------------------------------------------------------------------
st.sidebar.header("Walk-Forward Settings")
train_weeks = st.sidebar.number_input("Training Window (weeks)", min_value=1, max_value=104, value=12, step=1)
test_weeks = st.sidebar.number_input("Blind Test Window (weeks)", min_value=1, max_value=52, value=3, step=1)

total_weeks = (df.index.max() - df.index.min()).days / 7
est_folds = max(0, int((total_weeks - train_weeks) // test_weeks) + 1) if total_weeks >= train_weeks + test_weeks else 0
st.sidebar.caption(f"~{est_folds} rolling folds over the selected date range")

st.sidebar.caption("Costs (fixed): 0.001% exchange fee + 0.05% slippage, deducted on every trade leg.")

run_clicked = st.sidebar.button("Run Walk-Forward Analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("Walk-Forward Trading App")
st.caption(
    f"{symbol} · {timeframe} bars · Rolling range breakout (lookback ∈ {{{', '.join(str(n) for n in range_lookbacks)}}} bars, "
    f"buffer ∈ {{{', '.join(str(b) for b in buffer_fracs)}}}) with a volatility-regime gate and a "
    f"failed-breakout stop, day trades only ({session_choice if session else 'no session filter — 1d bars'}), "
    f"optimized on a rolling {train_weeks}-week train / {test_weeks}-week blind test walk-forward schedule."
)

if run_clicked:
    ann_factor = BARS_PER_YEAR.get(timeframe, 252)
    progress_bar = st.progress(0, text="Starting walk-forward optimization...")

    def on_fold_done(completed, total, fold):
        pct = completed / total
        progress_bar.progress(
            pct,
            text=(
                f"Fold {completed}/{total} — training "
                f"{fold.train_start.date()} → {fold.train_end.date()}, "
                f"testing {fold.test_start.date()} → {fold.test_end.date()}"
            ),
        )

    try:
        with st.spinner("Optimizing range-breakout parameters on each rolling training window..."):
            oos_returns, oos_trades, folds = run_walk_forward(
                df, train_weeks, test_weeks, grid, ann_factor, progress_callback=on_fold_done
            )
            insample_returns, insample_trades, insample_params = run_retail_insample(df, grid, ann_factor)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    progress_bar.progress(1.0, text=f"Walk-forward complete — {len(folds)} folds processed.")

    st.session_state["results"] = {
        "oos_returns": oos_returns,
        "oos_trades": oos_trades,
        "folds": folds,
        "insample_returns": insample_returns,
        "insample_trades": insample_trades,
        "insample_params": insample_params,
        "ann_factor": ann_factor,
        "symbol": symbol,
        "timeframe": timeframe,
    }

results = st.session_state.get("results")

if not results:
    st.info("Configure parameters in the sidebar and click **Run Walk-Forward Analysis** to begin.")
    st.stop()

folds = results["folds"]
oos_returns = results["oos_returns"]
oos_trades = results["oos_trades"]
insample_returns = results["insample_returns"]
insample_trades = results["insample_trades"]
ann_factor = results["ann_factor"]

# ---------------------------------------------------------------------------
# Gantt chart of train / test windows
# ---------------------------------------------------------------------------
st.subheader("Walk-Forward Fold Schedule")
gantt_df = folds_to_gantt_df(folds)
if not gantt_df.empty:
    fold_order = [f"Fold {f.index + 1}" for f in folds if f.best_params]
    gantt_fig = px.timeline(
        gantt_df,
        x_start="Start",
        x_end="Finish",
        y="Fold",
        color="Phase",
        color_discrete_map={"Train (In-Sample)": "blue", "Test (Blind OOS)": "orange"},
        category_orders={"Fold": fold_order},
    )
    gantt_fig.update_yaxes(autorange="reversed", title=None)
    gantt_fig.update_xaxes(title="Date")
    gantt_fig.update_layout(height=min(1200, max(350, 22 * len(fold_order))), legend_title=None)
    st.plotly_chart(gantt_fig, use_container_width=True)
else:
    st.warning("No folds produced a valid parameter set.")

# ---------------------------------------------------------------------------
# Side-by-side metrics
# ---------------------------------------------------------------------------
st.subheader("Performance Comparison")
insample_stats = summarize(insample_returns, insample_trades, ann_factor)
oos_stats = summarize(oos_returns, oos_trades, ann_factor)

col_left, col_right = st.columns(2)
with col_left:
    st.markdown("##### Retail In-Sample (curve-fit)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Return", fmt_pct(insample_stats["Total Return"]))
    m2.metric("CAGR", fmt_pct(insample_stats["CAGR"]))
    m3.metric("Sharpe Ratio", f"{insample_stats['Sharpe Ratio']:.2f}")
    m4, m5, m6 = st.columns(3)
    m4.metric("Max Drawdown", fmt_pct(insample_stats["Max Drawdown"]))
    m5.metric("Win Rate", fmt_pct(insample_stats["Win Rate"]))
    m6.metric("Trades", insample_stats["Trades"])

with col_right:
    st.markdown("##### Walk-Forward Out-of-Sample (stitched)")
    n1, n2, n3 = st.columns(3)
    n1.metric("Total Return", fmt_pct(oos_stats["Total Return"]))
    n2.metric("CAGR", fmt_pct(oos_stats["CAGR"]))
    n3.metric("Sharpe Ratio", f"{oos_stats['Sharpe Ratio']:.2f}")
    n4, n5, n6 = st.columns(3)
    n4.metric("Max Drawdown", fmt_pct(oos_stats["Max Drawdown"]))
    n5.metric("Win Rate", fmt_pct(oos_stats["Win Rate"]))
    n6.metric("Trades", oos_stats["Trades"])

comparison_df = pd.DataFrame({"Retail In-Sample": insample_stats, "Walk-Forward OOS": oos_stats})
st.dataframe(comparison_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Large equity curve comparison
# ---------------------------------------------------------------------------
st.subheader("Equity Curves: Retail In-Sample vs. Walk-Forward Out-of-Sample")
insample_eq = equity_curve(insample_returns, start=100.0)
oos_eq = equity_curve(oos_returns, start=100.0)

eq_fig = go.Figure()
eq_fig.add_trace(
    go.Scatter(x=insample_eq.index, y=insample_eq.values, name="Retail In-Sample (curve-fit)", line=dict(color="#d62728", width=2))
)
eq_fig.add_trace(
    go.Scatter(x=oos_eq.index, y=oos_eq.values, name="Walk-Forward Out-of-Sample (stitched)", line=dict(color="#2ca02c", width=2))
)
eq_fig.update_layout(
    height=650,
    yaxis_title="Equity (Start = 100)",
    xaxis_title="Date",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(eq_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------
with st.expander("Fold-by-Fold Optimization Detail"):
    st.caption(param_stability_caption(folds))
    st.dataframe(folds_to_table(folds), use_container_width=True)

with st.expander("Retail In-Sample Parameters (single curve-fit set)"):
    st.json(results["insample_params"])

with st.expander(f"Walk-Forward OOS Trade Log ({len(oos_trades)} trades)"):
    st.dataframe(oos_trades, use_container_width=True)

with st.expander(f"Retail In-Sample Trade Log ({len(insample_trades)} trades)"):
    st.dataframe(insample_trades, use_container_width=True)
