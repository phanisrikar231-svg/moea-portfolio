import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from realtime.live_feed import fetch_live_prices, fetch_historical, get_crisis_periods, NIFTY50_TICKERS
from data.fetch_stocks import compute_stats
from engine.nsga2 import setup_nsga2, run_nsga2
from evaluation.metrics import compute_all_metrics, backtest_portfolio
from baselines.equal_weight import equal_weight_portfolio
from baselines.markowitz import markowitz_min_variance
from baselines.monte_carlo import monte_carlo_best
from downloads.export_manager import export_csv, export_excel, export_pdf

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MOEA Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Bloomberg theme ──────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0d0f14}
[data-testid="stSidebar"]{background:#0a0c10;border-right:1px solid #1e2530}
[data-testid="stHeader"]{background:#0a0c10;border-bottom:1px solid #1e2530}
section[data-testid="stSidebar"] *{color:#9ca3af !important}
h1,h2,h3,h4,h5,h6{color:#e5e7eb !important}
p,li,label,div{color:#d1d5db}
.stButton>button{background:#0f2027 !important;border:1px solid #00e5a044 !important;
  color:#00e5a0 !important;border-radius:6px !important;font-weight:500}
.stButton>button:hover{background:#00e5a0 !important;color:#0d0f14 !important}
.stTextInput>div>input,.stSelectbox>div>div{
  background:#111520 !important;border:1px solid #1e2530 !important;color:#e5e7eb !important}
.stSlider>div{color:#9ca3af}
[data-testid="metric-container"]{background:#111520;border:1px solid #1e2530;
  border-radius:8px;padding:12px}
[data-testid="stDataFrame"]{background:#111520}
.stTabs [data-baseweb="tab"]{color:#6b7280}
.stTabs [aria-selected="true"]{color:#00e5a0 !important;border-bottom:2px solid #00e5a0}
hr{border-color:#1e2530}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**📊 Stock Universe**")
    selected_stocks = st.multiselect(
        "Select stocks (min 5)",
        NIFTY50_TICKERS,
        default=NIFTY50_TICKERS[:10],
        key="stocks",
    )

    st.markdown("**📅 Date Range**")
    periods = get_crisis_periods()
    period_choice = st.selectbox("Period", list(periods.keys()))
    if period_choice == "Custom":
        start_d = st.date_input("Start date", datetime(2018, 1, 1))
        end_d   = st.date_input("End date",   datetime(2024, 12, 31))
        start_str, end_str = str(start_d), str(end_d)
    else:
        start_str, end_str = periods[period_choice]

    st.markdown("**🧬 NSGA-II Settings**")
    st.caption("Population size — number of portfolio combinations evolved simultaneously")
    pop_size = st.slider("Population size", 50, 300, 100, 10)
    st.caption("Generations — number of evolution rounds (more = better result, slower)")
    n_gen = st.slider("Generations", 50, 500, 150, 25)

    st.markdown("---")
    run_btn = st.button("▶ Run Optimization", use_container_width=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📈 MOEA Portfolio Optimizer")
st.markdown("*Multi-Objective Portfolio Optimization using NSGA-II — NSE Nifty 50*")
st.markdown("---")

# ── Live prices ───────────────────────────────────────────────────────────────
st.markdown("### 🔴 Live Market Prices")
display_tickers = selected_stocks[:6] if selected_stocks else NIFTY50_TICKERS[:6]
with st.spinner("Fetching live prices..."):
    live = fetch_live_prices(display_tickers)

cols = st.columns(len(live))
for col, (ticker, d) in zip(cols, live.items()):
    label = ticker.replace(".NS", "")
    delta = f"{d['change_pct']}%" if d['price'] > 0 else "N/A"
    col.metric(label, f"₹{d['price']}" if d['price'] > 0 else "—", delta)

if st.button("🔄 Refresh Prices"):
    st.rerun()

st.markdown("---")

# ── Run optimization ──────────────────────────────────────────────────────────
if run_btn:
    if not selected_stocks or len(selected_stocks) < 2:
        st.error("Select at least 2 stocks.")
        st.stop()

    with st.spinner("⬇️ Downloading historical data..."):
        prices = fetch_historical(selected_stocks, start_str, end_str)

    if prices is None or prices.empty or prices.shape[1] < 2:
        st.error("Not enough data returned. Try a different date range or stocks.")
        st.stop()

    # Keep only common columns
    prices = prices.dropna(axis=1, how="all")
    valid_tickers = [t for t in selected_stocks if t in prices.columns]
    prices = prices[valid_tickers].dropna()

    if len(valid_tickers) < 2:
        st.error("Not enough valid stock data. Try different stocks.")
        st.stop()

    returns, mean_ret, cov_mat = compute_stats(prices)
    st.session_state["prices"]    = prices
    st.session_state["returns"]   = returns
    st.session_state["mean_ret"]  = mean_ret
    st.session_state["cov_mat"]   = cov_mat
    st.session_state["tickers"]   = valid_tickers

    # Progress
    prog_bar   = st.progress(0)
    prog_label = st.empty()

    def cb(gen, total):
        prog_bar.progress(gen / total)
        prog_label.text(f"Generation {gen} / {total}")

    with st.spinner("🧬 Running NSGA-II..."):
        toolbox, ps, ng = setup_nsga2(
            mean_ret, cov_mat, returns, len(valid_tickers), pop_size, n_gen
        )
        pop, hof = run_nsga2(toolbox, ps, ng, progress_callback=cb)

    prog_bar.empty()
    prog_label.empty()
    st.session_state["hof"] = hof
    st.success(f"✅ Optimization complete! Found **{len(hof)}** Pareto-optimal portfolios.")

# ── Results ───────────────────────────────────────────────────────────────────
if "hof" in st.session_state:
    hof        = st.session_state["hof"]
    prices     = st.session_state["prices"]
    returns    = st.session_state["returns"]
    mean_ret   = st.session_state["mean_ret"]
    cov_mat    = st.session_state["cov_mat"]
    tickers    = st.session_state["tickers"]

    f1s = [-ind.fitness.values[0] for ind in hof]   # return
    f2s = [ind.fitness.values[1]  for ind in hof]   # risk
    f3s = [ind.fitness.values[2]  for ind in hof]   # drawdown

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Pareto Front", "📊 Portfolio", "⚔️ Comparison", "⬇️ Download"
    ])

    # ── Tab 1: Pareto Front ──────────────────────────────────────────────────
    with tab1:
        st.markdown("### 🎯 Pareto Front — 3D Objective Space")
        st.caption("Each point = one optimal portfolio. Click any point to explore it.")

        fig3d = go.Figure(data=[go.Scatter3d(
            x=f2s, y=f1s, z=f3s,
            mode="markers",
            marker=dict(
                size=5,
                color=f1s,
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="Return", tickfont=dict(color="#e5e7eb")),
            ),
            text=[f"Portfolio {i}<br>Return:{round(f1s[i]*100,1)}%  Risk:{round(f2s[i]*100,1)}%  DD:{round(f3s[i]*100,1)}%"
                  for i in range(len(hof))],
            hovertemplate="%{text}<extra></extra>",
        )])
        fig3d.update_layout(
            scene=dict(
                xaxis_title="Risk (Volatility)",
                yaxis_title="Return",
                zaxis_title="Max Drawdown",
                bgcolor="#0d0f14",
                xaxis=dict(gridcolor="#1e2530", color="#9ca3af"),
                yaxis=dict(gridcolor="#1e2530", color="#9ca3af"),
                zaxis=dict(gridcolor="#1e2530", color="#9ca3af"),
            ),
            paper_bgcolor="#0d0f14",
            font=dict(color="#e5e7eb"),
            height=560,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig3d, use_container_width=True)

        idx = st.slider("🔍 Select portfolio from Pareto front",
                        0, len(hof) - 1, len(hof) // 2, key="pf_slider")
        st.session_state["selected_idx"] = idx

    # ── Tab 2: Portfolio & Backtest ──────────────────────────────────────────
    with tab2:
        idx    = st.session_state.get("selected_idx", len(hof) // 2)
        best_w = np.array(hof[idx])
        best_w = best_w / best_w.sum()
        labels = [t.replace(".NS", "") for t in tickers]
        alloc  = dict(zip(labels, best_w))

        st.markdown("### 📊 Portfolio Allocation")
        c1, c2 = st.columns(2)

        with c1:
            fig_pie = go.Figure(go.Pie(
                labels=labels,
                values=[round(v, 4) for v in best_w],
                hole=0.4,
                marker=dict(colors=[
                    "#00e5a0","#60a5fa","#a78bfa","#f59e0b","#34d399",
                    "#f87171","#38bdf8","#fb923c","#e879f9","#facc15",
                    "#4ade80","#818cf8","#fb7185","#22d3ee","#a3e635",
                    "#fbbf24","#c084fc","#67e8f9","#86efac","#fca5a5",
                ]),
            ))
            fig_pie.update_layout(
                paper_bgcolor="#0d0f14",
                font=dict(color="#e5e7eb"),
                height=340,
                legend=dict(font=dict(color="#9ca3af")),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("#### Performance Metrics")
            metrics = compute_all_metrics(best_w, mean_ret, cov_mat, returns)
            for k, v in metrics.items():
                st.metric(k, v)
            st.session_state["metrics"] = metrics

        # Backtest
        st.markdown("### 📈 Backtesting — Portfolio Value Over Time")
        bt = backtest_portfolio(best_w, prices)
        fig_bt = go.Figure(go.Scatter(
            x=bt.index, y=bt.values,
            fill="tozeroy",
            fillcolor="rgba(0,229,160,0.08)",
            line=dict(color="#00e5a0", width=2),
            name="Portfolio",
        ))
        fig_bt.update_layout(
            paper_bgcolor="#0d0f14",
            plot_bgcolor="#0d0f14",
            font=dict(color="#e5e7eb"),
            xaxis=dict(gridcolor="#1e2530", title="Date"),
            yaxis=dict(gridcolor="#1e2530", title="Value (Base ₹100)"),
            height=360,
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        # Save portfolio name
        st.markdown("### 💾 Save This Portfolio")
        pname = st.text_input("Give this portfolio a name", key="pname")
        if st.button("Save", key="save_btn"):
            if pname:
                saved = st.session_state.get("saved_portfolios", {})
                saved[pname] = {"weights": best_w.tolist(),
                                "tickers": tickers, "metrics": metrics}
                st.session_state["saved_portfolios"] = saved
                st.success(f"Saved as '{pname}'")
            else:
                st.warning("Enter a name first.")

        saved = st.session_state.get("saved_portfolios", {})
        if saved:
            st.markdown("#### 📁 Your Saved Portfolios")
            for name in saved:
                st.write(f"• **{name}**")

    # ── Tab 3: Comparison ────────────────────────────────────────────────────
    with tab3:
        st.markdown("### ⚔️ Algorithm Comparison")
        idx    = st.session_state.get("selected_idx", len(hof) // 2)
        best_w = np.array(hof[idx])
        best_w = best_w / best_w.sum()

        with st.spinner("Computing baselines..."):
            ew  = equal_weight_portfolio(len(tickers))
            mk  = markowitz_min_variance(mean_ret, cov_mat)
            mc  = monte_carlo_best(mean_ret, cov_mat, 5000)

        rows = []
        for label, w in [("NSGA-II", best_w), ("Markowitz", mk),
                          ("Equal Weight", ew), ("Monte Carlo", mc)]:
            m = compute_all_metrics(w, mean_ret, cov_mat, returns)
            m["Algorithm"] = label
            rows.append(m)

        comp_df = pd.DataFrame(rows).set_index("Algorithm")
        st.dataframe(comp_df, use_container_width=True)

        # Sharpe bar chart
        fig_bar = go.Figure(go.Bar(
            x=comp_df.index.tolist(),
            y=comp_df["Sharpe Ratio"].tolist(),
            marker_color=["#00e5a0", "#60a5fa", "#a78bfa", "#f59e0b"],
            text=[str(v) for v in comp_df["Sharpe Ratio"].tolist()],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Sharpe Ratio Comparison",
            paper_bgcolor="#0d0f14",
            plot_bgcolor="#0d0f14",
            font=dict(color="#e5e7eb"),
            xaxis=dict(gridcolor="#1e2530"),
            yaxis=dict(gridcolor="#1e2530", title="Sharpe Ratio"),
            height=340,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Backtest comparison
        st.markdown("#### Backtest Comparison — All Methods")
        fig_comp = go.Figure()
        colors = {"NSGA-II": "#00e5a0", "Markowitz": "#60a5fa",
                  "Equal Weight": "#a78bfa", "Monte Carlo": "#f59e0b"}
        for label, w in [("NSGA-II", best_w), ("Markowitz", mk),
                          ("Equal Weight", ew), ("Monte Carlo", mc)]:
            bt = backtest_portfolio(w, prices)
            fig_comp.add_trace(go.Scatter(
                x=bt.index, y=bt.values,
                name=label, line=dict(color=colors[label], width=1.5)
            ))
        fig_comp.update_layout(
            paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
            font=dict(color="#e5e7eb"),
            xaxis=dict(gridcolor="#1e2530", title="Date"),
            yaxis=dict(gridcolor="#1e2530", title="Value (Base ₹100)"),
            legend=dict(font=dict(color="#9ca3af")),
            height=380,
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.session_state["comp_df"] = comp_df

    # ── Tab 4: Download ──────────────────────────────────────────────────────
    with tab4:
        st.markdown("### ⬇️ Download Results")
        idx    = st.session_state.get("selected_idx", len(hof) // 2)
        best_w = np.array(hof[idx])
        best_w = best_w / best_w.sum()
        labels = [t.replace(".NS", "") for t in tickers]
        alloc  = dict(zip(labels, best_w))
        metrics = st.session_state.get("metrics",
                  compute_all_metrics(best_w, mean_ret, cov_mat, returns))
        comp_df = st.session_state.get("comp_df", pd.DataFrame())

        pareto_df = pd.DataFrame({
            "Return (%)":    [round(v * 100, 2) for v in f1s],
            "Risk (%)":      [round(v * 100, 2) for v in f2s],
            "Drawdown (%)":  [round(v * 100, 2) for v in f3s],
        })

        user_name = st.text_input("Your name for the PDF report", value="Phani Srikar")

        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.download_button(
                "📥 Download CSV",
                data=export_csv(pareto_df),
                file_name="pareto_front.csv",
                mime="text/csv",
            )
        with dc2:
            st.download_button(
                "📥 Download Excel",
                data=export_excel(pareto_df, comp_df.reset_index() if not comp_df.empty else pd.DataFrame(), alloc),
                file_name="moea_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with dc3:
            st.download_button(
                "📥 Download PDF Report",
                data=export_pdf(user_name, metrics, alloc),
                file_name="moea_report.pdf",
                mime="application/pdf",
            )

        st.markdown("---")
        st.markdown("#### 📄 Pareto Front Data Preview")
        st.dataframe(pareto_df, use_container_width=True)
