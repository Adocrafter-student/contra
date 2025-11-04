# contra.py - CONTRA: Sentiment-Alpha Stock Prediction Model
# Production implementation with Wikipedia, Google Trends, and Event Kernels

import os, sys
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
from pytrends.request import TrendReq
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import requests
from urllib.parse import quote

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Stock & Market
TICKER = "TSLA"
MARKET = "^GSPC"
START  = "2018-01-01"
END    = "2020-12-31"

# Sentiment Sources
KEYWORDS = ["Tesla", "Elon Musk"]
WIKI_TITLES = ["Tesla,_Inc.", "Elon_Musk"]  # Use underscores, exact case
EVENTS_CSV = "events.csv"

# Output
OUTPUT_DIR = "outputs3"

# Feature Toggles (turn on/off different sentiment sources)
USE_TRENDS = True        # Google Trends data
USE_WIKIPEDIA = True     # Wikipedia pageview data
USE_EVENT_KERNEL = True  # Event-based shock kernel

# Event-kernel shape: defines sentiment impact around events
# tau = days relative to event (0 = event day)
# Positive events boost sentiment, negative events decrease it
EVENT_KERNEL = {
    "positive": { -2: 0.5, -1: 1.0, 0: 1.5, +1: 1.0, +2: 0.5 },
    "negative": { -2:-0.5, -1:-1.0, 0:-1.5, +1:-1.0, +2:-0.5 }
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def save_html(fig, fname):
    """Save Plotly figure as HTML"""
    ensure_dir(OUTPUT_DIR)
    fig.write_html(os.path.join(OUTPUT_DIR, fname), include_plotlyjs="cdn")
    
def try_load_events():
    """Load events from CSV if available"""
    if os.path.exists(EVENTS_CSV):
        try:
            ev = pd.read_csv(EVENTS_CSV, parse_dates=["date"])
            if "event_type" not in ev.columns:
                raise ValueError("events.csv must have 'event_type' column")
            ev["event_type"] = ev["event_type"].str.lower().str.strip()
            ev = ev.sort_values("date")
            print(f"✓ Loaded {len(ev)} events from {EVENTS_CSV}")
            return ev
        except Exception as e:
            print(f"⚠ Could not load events: {e}")
            return None
    return None

# =============================================================================
# SENTIMENT DATA COLLECTION FUNCTIONS
# =============================================================================

def fetch_trends_daily(start, end, keywords):
    """
    Fetch Google Trends data year-by-year for better granularity.
    Returns DataFrame with 'sent_trends_z' (z-scored sentiment).
    """
    print(f"\n📈 Fetching Google Trends data...")
    pytrends = TrendReq(hl="en-US", tz=0, retries=2, backoff_factor=0.2)
    
    frames = []
    y0, y1 = pd.to_datetime(start).year, pd.to_datetime(end).year
    
    for year in range(y0, y1+1):
        s = f"{year}-01-01"
        e = f"{year}-12-31" if year < y1 else end
        print(f"  Fetching {year}...")
        
        try:
            pytrends.build_payload(keywords, timeframe=f"{s} {e}", geo="")
            part = pytrends.interest_over_time()
            if "isPartial" in part.columns:
                part = part.drop(columns=["isPartial"])
            frames.append(part)
        except Exception as ex:
            print(f"  ⚠ Failed to fetch {year}: {ex}")
            continue
    
    if not frames:
        raise RuntimeError("No Google Trends data retrieved")
    
    tr = pd.concat(frames).sort_index()
    tr.columns = [c.lower().replace(" ","_") for c in tr.columns]
    tr = tr.resample("D").mean().ffill()  # Convert to daily
    tr["sent_trends_raw"] = tr.mean(axis=1)  # Average across keywords
    tr["sent_trends_z"] = (tr["sent_trends_raw"] - tr["sent_trends_raw"].mean()) / tr["sent_trends_raw"].std()
    
    print(f"  ✓ Google Trends: {len(tr)} daily points from {tr.index.min().date()} to {tr.index.max().date()}")
    return tr[["sent_trends_z"]]

def fetch_wiki_daily(start, end, titles):
    """
    Fetch Wikipedia daily pageviews using Wikimedia REST API.
    Returns DataFrame with 'sent_wiki_z' (z-scored pageview sentiment).
    
    Wikipedia pageviews spike dramatically on scandal/news days,
    providing a more reactive signal than Google Trends.
    """
    print(f"\n📚 Fetching Wikipedia pageview data...")
    s = pd.to_datetime(start).strftime("%Y%m%d")
    e = pd.to_datetime(end).strftime("%Y%m%d")
    
    series = []
    for title in titles:
        print(f"  Fetching: {title}")
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{quote(title)}/daily/{s}/{e}"
        
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            js = r.json()
            rows = js.get("items", [])
            
            if not rows:
                print(f"    ⚠ No data returned for {title}")
                continue
                
            df = pd.DataFrame({
                "date": [pd.to_datetime(x["timestamp"][:8]) for x in rows],
                f"pv_{title}": [x["views"] for x in rows]
            }).set_index("date")
            
            series.append(df)
            print(f"    ✓ Got {len(df)} daily pageviews")
            
        except Exception as ex:
            print(f"    ⚠ Failed to fetch {title}: {ex}")
            continue
    
    if not series:
        raise RuntimeError("No Wikipedia data retrieved")
    
    pv = pd.concat(series, axis=1).asfreq("D").ffill()
    pv["sent_wiki_raw"] = pv.mean(axis=1)  # Average across pages
    pv["sent_wiki_z"] = (pv["sent_wiki_raw"] - pv["sent_wiki_raw"].mean()) / pv["sent_wiki_raw"].std()
    
    print(f"  ✓ Wikipedia: {len(pv)} daily points from {pv.index.min().date()} to {pv.index.max().date()}")
    return pv[["sent_wiki_z"]]

def build_event_kernel_index(trading_index, events_df, kernel):
    """
    Build event-kernel sentiment index from curated events.
    
    For each event in events_df, applies a kernel weight pattern
    around the event date (e.g., day -2, -1, 0, +1, +2).
    
    This creates a "shock" signal that peaks on event days,
    representing the expected sentiment impact of known controversies/news.
    
    Returns z-scored Series aligned to trading days.
    """
    print(f"\n⚡ Building event kernel index...")
    idx = pd.Index(trading_index)  # Trading days only
    ek = pd.Series(0.0, index=idx)
    
    event_count = {"positive": 0, "negative": 0}
    
    for _, ev in events_df.iterrows():
        # Find nearest trading day
        d0 = ev["date"]
        if d0 not in idx:
            d0 = idx[idx.get_indexer([d0], method="nearest")[0]]
        
        event_type = ev["event_type"].lower()
        if event_type not in kernel:
            continue
            
        # Apply kernel weights around event
        for tau, weight in kernel[event_type].items():
            target = d0 + pd.tseries.offsets.BDay(tau)  # Business day offset
            if target in ek.index:
                ek.loc[target] += weight
        
        event_count[event_type] += 1
    
    # Z-score the kernel values
    ek_z = (ek - ek.mean()) / ek.std(ddof=0) if ek.std() > 0 else ek
    
    print(f"  ✓ Applied kernel to {event_count['positive']} positive and {event_count['negative']} negative events")
    print(f"  ✓ Non-zero days: {(ek != 0).sum()}")
    
    return ek_z.rename("sent_event_z")

# =============================================================================
# MODELING FUNCTIONS
# =============================================================================

def fit_ols_robust(df, with_sent):
    """Fit OLS regression with robust standard errors"""
    X = df[["ret_mkt"]].copy()
    if with_sent:
        X["sent_composite_z"] = df["sent_composite_z"]  # Use composite sentiment
    X = sm.add_constant(X)
    y = df["ret_stk"]
    model = sm.OLS(y, X).fit(cov_type="HC1")
    return model

def predict(model, df):
    """Generate predictions from fitted model"""
    # Get feature names from model (excluding constant)
    feature_names = [col for col in model.params.index if col != 'const']
    
    # Build feature matrix
    X = df[feature_names].copy()
    X = sm.add_constant(X, has_constant='add')
    
    # Ensure column order matches model
    X = X[model.params.index]
    
    return model.predict(X)

def metrics(y_true, y_pred):
    """Calculate MAE and RMSE"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return mae, rmse

# =============================================================================
# EVENT STUDY FUNCTIONS
# =============================================================================

def market_model_params(data, up_to_date):
    """Estimate market model parameters using 120-day window ending 21 days before event"""
    df = data.loc[:up_to_date].iloc[:-21]  # Cut off 21 days pre-event
    est = df.iloc[-120:] if len(df) >= 120 else df
    X = sm.add_constant(est["ret_mkt"])
    y = est["ret_stk"]
    m = sm.OLS(y, X).fit()
    return m.params["const"], m.params["ret_mkt"]

def window_data(data, center_date, k=5):
    """Extract data window around event date"""
    idx = data.index
    # Find nearest trading day
    if center_date not in idx:
        center_date = idx[idx.get_indexer([center_date], method="nearest")[0]]
    loc = idx.get_loc(center_date)
    lo = max(loc - k, 0)
    hi = min(loc + k, len(idx)-1)
    out = data.iloc[lo:hi+1].copy()
    # tau = days relative to event (0 = event day)
    out["tau"] = np.arange(lo, hi+1) - loc
    return out

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CONTRA: Sentiment-Alpha Stock Prediction Model")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 1) Download Stock Price Data
    # -------------------------------------------------------------------------
    print(f"\n📊 Downloading stock data for {TICKER} and {MARKET}...")
    prices = yf.download([TICKER, MARKET], start=START, end=END, auto_adjust=True)["Close"]
    prices.columns = ["stk", "mkt"]
    ret = prices.pct_change().dropna()
    ret.index = pd.to_datetime(ret.index)
    ret.rename(columns={"stk":"ret_stk", "mkt":"ret_mkt"}, inplace=True)
    print(f"  ✓ Got {len(ret)} trading days of returns")
    
    # -------------------------------------------------------------------------
    # 2) Build Composite Sentiment from Multiple Sources
    # -------------------------------------------------------------------------
    print(f"\n🔍 Building composite sentiment index...")
    print(f"  Sources enabled: Trends={USE_TRENDS}, Wikipedia={USE_WIKIPEDIA}, Events={USE_EVENT_KERNEL}")
    
    parts = []
    
    # (A) Google Trends
    if USE_TRENDS:
        try:
            tr = fetch_trends_daily(START, END, KEYWORDS)
            parts.append(tr)
        except Exception as e:
            print(f"  ⚠ Google Trends failed: {e}")
    
    # (B) Wikipedia Pageviews
    if USE_WIKIPEDIA:
        try:
            pv = fetch_wiki_daily(START, END, WIKI_TITLES)
            parts.append(pv)
        except Exception as e:
            print(f"  ⚠ Wikipedia failed: {e}")
    
    # (C) Event Kernel
    events = try_load_events()
    if USE_EVENT_KERNEL and events is not None and not events.empty:
        try:
            events["date"] = pd.to_datetime(events["date"])
            sent_event = build_event_kernel_index(ret.index, events, EVENT_KERNEL)
            parts.append(sent_event.to_frame())
        except Exception as e:
            print(f"  ⚠ Event kernel failed: {e}")
    
    # Compose final sentiment
    if not parts:
        raise RuntimeError("❌ No sentiment sources available! Enable at least one source.")
    
    print(f"\n🔧 Composing sentiment from {len(parts)} source(s)...")
    
    # Align all sources to trading days
    aligned_parts = []
    for p in parts:
        aligned = p.reindex(ret.index).ffill()  # Forward-fill missing days
        aligned_parts.append(aligned)
    
    sentiment = pd.concat(aligned_parts, axis=1)
    
    # Build composite as equal-weighted average
    sent_cols = [c for c in sentiment.columns if c.startswith("sent_")]
    sentiment["sent_composite_raw"] = sentiment[sent_cols].mean(axis=1)
    sentiment["sent_composite_z"] = (
        (sentiment["sent_composite_raw"] - sentiment["sent_composite_raw"].mean()) / 
        sentiment["sent_composite_raw"].std()
    )
    
    print(f"  ✓ Composite sentiment created from: {', '.join(sent_cols)}")
    
    # -------------------------------------------------------------------------
    # 3) Merge Returns + Sentiment into Modeling Dataset
    # -------------------------------------------------------------------------
    data = ret.join(sentiment, how="inner").dropna()
    print(f"\n📋 Final dataset: {len(data)} observations from {data.index[0].date()} to {data.index[-1].date()}")
    
    # -------------------------------------------------------------------------
    # 4) Train/Test Split + Model Fitting
    # -------------------------------------------------------------------------
    split = int(len(data) * 0.7)
    train, test = data.iloc[:split], data.iloc[split:]
    
    print(f"\n🎯 Training models...")
    print(f"  Train: {len(train)} obs ({train.index[0].date()} to {train.index[-1].date()})")
    print(f"  Test:  {len(test)} obs ({test.index[0].date()} to {test.index[-1].date()})")
    
    base = fit_ols_robust(train, with_sent=False)    # Baseline: market only
    contra = fit_ols_robust(train, with_sent=True)   # CONTRA: market + sentiment
    
    # Generate predictions
    yhat_base_train = predict(base, train)
    yhat_base_test = predict(base, test)
    yhat_contra_train = predict(contra, train)
    yhat_contra_test = predict(contra, test)
    
    # Calculate metrics
    mae_b, rmse_b = metrics(test["ret_stk"], yhat_base_test)
    mae_c, rmse_c = metrics(test["ret_stk"], yhat_contra_test)
    
    # -------------------------------------------------------------------------
    # 5) Display & Save Results
    # -------------------------------------------------------------------------
    print(f"\n" + "=" * 70)
    print("MODEL RESULTS")
    print("=" * 70)
    
    print(f"\n📊 TRAINING SET (Adj R²):")
    print(f"  Baseline (Market only):  {base.rsquared_adj:.6f}")
    print(f"  CONTRA (+ Sentiment):    {contra.rsquared_adj:.6f}")
    print(f"  Improvement:             {(contra.rsquared_adj - base.rsquared_adj):.6f} ({((contra.rsquared_adj - base.rsquared_adj)/base.rsquared_adj*100):+.2f}%)")
    
    print(f"\n📊 TEST SET PERFORMANCE:")
    print(f"  MAE  — Baseline: {mae_b:.6e}  |  CONTRA: {mae_c:.6e}  |  Δ: {((mae_b-mae_c)/mae_b*100):+.2f}%")
    print(f"  RMSE — Baseline: {rmse_b:.6e}  |  CONTRA: {rmse_c:.6e}  |  Δ: {((rmse_b-rmse_c)/rmse_b*100):+.2f}%")
    
    print(f"\n🎯 SENTIMENT-ALPHA COEFFICIENT (θ):")
    if "sent_composite_z" in contra.params:
        theta = contra.params["sent_composite_z"]
        p_val = contra.pvalues["sent_composite_z"]
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
        print(f"  θ = {theta:.6f}  (p={p_val:.4f}) {sig}")
        print(f"  Interpretation: 1 SD ↑ in sentiment → {theta*100:.3f}% Δ in daily return")
    
    # Save detailed results
    ensure_dir(OUTPUT_DIR)
    
    with open(os.path.join(OUTPUT_DIR, "contra_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("CONTRA: Sentiment-Alpha Model Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Ticker: {TICKER}\n")
        f.write(f"Period: {START} to {END}\n")
        f.write(f"Sentiment Sources: {', '.join(sent_cols)}\n\n")
        f.write("TRAINING SET (Adj R²):\n")
        f.write(f"  Baseline:    {base.rsquared_adj:.6f}\n")
        f.write(f"  CONTRA:      {contra.rsquared_adj:.6f}\n")
        f.write(f"  Improvement: {(contra.rsquared_adj - base.rsquared_adj):.6f}\n\n")
        f.write("TEST SET:\n")
        f.write(f"  MAE  — Baseline: {mae_b:.6e}  |  CONTRA: {mae_c:.6e}\n")
        f.write(f"  RMSE — Baseline: {rmse_b:.6e}  |  CONTRA: {rmse_c:.6e}\n\n")
        if "sent_composite_z" in contra.params:
            f.write(f"SENTIMENT-ALPHA (θ): {contra.params['sent_composite_z']:.6f} (p={contra.pvalues['sent_composite_z']:.4f})\n")
    
    with open(os.path.join(OUTPUT_DIR, "contra_regression_summary.txt"), "w", encoding="utf-8") as f:
        f.write(contra.summary().as_text())
    
    # -------------------------------------------------------------------------
    # 6) Visualization: Actual vs Predicted Returns
    # -------------------------------------------------------------------------
    print(f"\n📈 Generating visualizations...")
    
    # Combine train and test predictions
    plot_df = data.copy()
    plot_df["pred_base"] = pd.concat([yhat_base_train, yhat_base_test])
    plot_df["pred_contra"] = pd.concat([yhat_contra_train, yhat_contra_test])
    
    fig1 = go.Figure()
    
    # Actual returns
    fig1.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["ret_stk"],
        mode="lines", name="Actual Return",
        line=dict(color="black", width=2)
    ))
    
    # Baseline predictions
    fig1.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["pred_base"],
        mode="lines", name="Baseline (Market Only)",
        line=dict(color="blue", dash="dot")
    ))
    
    # CONTRA predictions
    fig1.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["pred_contra"],
        mode="lines", name="CONTRA (+ Sentiment)",
        line=dict(color="red", width=2)
    ))
    
    # Mark train/test split
    split_date = test.index[0]
    fig1.add_shape(
        type="line",
        x0=split_date, x1=split_date,
        y0=0, y1=1, yref="paper",
        line=dict(color="green", width=2, dash="dash")
    )
    fig1.add_annotation(
        x=split_date, y=1, yref="paper",
        text="Train/Test Split", showarrow=False,
        yshift=10, font=dict(color="green")
    )
    
    # Mark events if available
    if events is not None and not events.empty:
        for _, ev in events.iterrows():
            ev_date = pd.to_datetime(ev["date"])
            if ev_date in plot_df.index:
                color = "lightgreen" if ev["event_type"] == "positive" else "lightcoral"
                fig1.add_vrect(
                    x0=ev_date - pd.Timedelta(days=1),
                    x1=ev_date + pd.Timedelta(days=1),
                    fillcolor=color, opacity=0.2, line_width=0
                )
    
    fig1.update_layout(
        title=f"{TICKER}: Actual vs Predicted Returns ({data.index[0].date()} to {data.index[-1].date()})",
        xaxis_title="Date",
        yaxis_title="Daily Return",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99)
    )
    save_html(fig1, "contra_predictions.html")
    print(f"  ✓ Saved: contra_predictions.html")
    
    # -------------------------------------------------------------------------
    # 7) Sentiment vs Returns Dual-Axis Plot
    # -------------------------------------------------------------------------
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=data.index, y=data["ret_stk"],
        mode="lines", name="Stock Returns",
        line=dict(color="black", width=1),
        yaxis="y1"
    ))
    
    fig2.add_trace(go.Scatter(
        x=data.index, y=data["sent_composite_z"],
        mode="lines", name="Composite Sentiment",
        line=dict(color="red", width=1.5),
        yaxis="y2"
    ))
    
    # Mark events
    if events is not None and not events.empty:
        for _, ev in events.iterrows():
            ev_date = pd.to_datetime(ev["date"])
            if ev_date in data.index:
                color = "green" if ev["event_type"] == "positive" else "red"
                fig2.add_vline(
                    x=ev_date,
                    line=dict(color=color, width=2, dash="dash"),
                    opacity=0.5
                )
    
    fig2.update_layout(
        title=f"{TICKER}: Stock Returns vs Composite Sentiment",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Daily Return", side="left"),
        yaxis2=dict(title="Sentiment Z-Score", overlaying="y", side="right"),
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99)
    )
    save_html(fig2, "contra_sentiment_overlay.html")
    print(f"  ✓ Saved: contra_sentiment_overlay.html")
    
    # -------------------------------------------------------------------------
    # 8) Granger Causality Test
    # -------------------------------------------------------------------------
    print(f"\n🔬 Running Granger causality tests...")
    gc_df = data[["ret_stk", "sent_composite_z"]].dropna()
    
    try:
        gtest = grangercausalitytests(gc_df, maxlag=5, verbose=False)
        
        rows = []
        for L in range(1, 6):
            p = gtest[L][0]["ssr_ftest"][1]
            rows.append({"lag": L, "p_value": p})
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            print(f"  Lag {L}: p={p:.4f} {sig}")
        
        gc_tbl = pd.DataFrame(rows)
        gc_tbl.to_csv(os.path.join(OUTPUT_DIR, "granger_causality.csv"), index=False)
        
        fig3 = px.bar(gc_tbl, x="lag", y="p_value",
                      title=f"{TICKER}: Granger Causality (Sentiment → Returns)")
        fig3.add_hline(y=0.05, line_dash="dash", annotation_text="α = 0.05")
        save_html(fig3, "contra_granger.html")
        print(f"  ✓ Saved: contra_granger.html")
        
    except Exception as e:
        print(f"  ⚠ Granger test failed: {e}")
    
    # -------------------------------------------------------------------------
    # 9) Event Study: Abnormal Returns Around Events
    # -------------------------------------------------------------------------
    if events is not None and not events.empty:
        print(f"\n📅 Running event study on {len(events)} events...")
        
        rows = []
        for _, ev in events.iterrows():
            try:
                a, b = market_model_params(data, ev["date"])
                W = window_data(data, ev["date"], k=5)
                W["er"] = a + b * W["ret_mkt"]  # Expected return
                W["ar"] = W["ret_stk"] - W["er"]  # Abnormal return
                W["event_type"] = ev["event_type"]
                W["event_date"] = pd.to_datetime(ev["date"]).date()
                rows.append(W[["event_date", "tau", "ar", "event_type"]])
            except Exception as e:
                print(f"  ⚠ Skipped event {ev['date']}: {e}")
        
        if rows:
            ES = pd.concat(rows, ignore_index=True)
            CAR = ES.sort_values(["event_date", "tau"]).copy()
            CAR["car"] = CAR.groupby("event_date")["ar"].transform(np.cumsum)
            
            # Aggregate by event type
            agg = (CAR.groupby(["event_type", "tau"])
                      .agg(mean_AR=("ar", "mean"),
                           mean_CAR=("car", "mean"),
                           n=("ar", "count"))
                      .reset_index())
            
            agg.to_csv(os.path.join(OUTPUT_DIR, "event_study.csv"), index=False)
            
            # Plot CAR for each event type
            for et in agg["event_type"].unique():
                sub = agg[agg["event_type"] == et]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sub["tau"], y=sub["mean_CAR"],
                    mode="lines+markers", name="Mean CAR"
                ))
                fig.add_hline(y=0, line_dash="dash")
                fig.add_vline(x=0, line_dash="dash")
                fig.update_layout(
                    title=f"{TICKER}: CAR [-5,+5] for {et.title()} Events (n={int(sub['n'].max())})",
                    xaxis_title="Days Relative to Event (τ)",
                    yaxis_title="Cumulative Abnormal Return"
                )
                save_html(fig, f"contra_car_{et}.html")
                print(f"  ✓ Saved: contra_car_{et}.html")
    
    # -------------------------------------------------------------------------
    # 10) Summary
    # -------------------------------------------------------------------------
    print(f"\n" + "=" * 70)
    print(f"✅ CONTRA analysis complete!")
    print(f"📁 Results saved in: {OUTPUT_DIR}/")
    print(f"=" * 70)
    print(f"\n💡 Key Findings:")
    print(f"  • Sentiment-alpha coefficient (θ): {contra.params.get('sent_composite_z', 0):.6f}")
    print(f"  • R² improvement: {((contra.rsquared_adj - base.rsquared_adj)/base.rsquared_adj*100):+.2f}%")
    print(f"  • Test RMSE reduction: {((rmse_b - rmse_c)/rmse_b*100):+.2f}%")
    print(f"\n📊 View interactive results:")
    print(f"  • contra_predictions.html - Model comparison")
    print(f"  • contra_sentiment_overlay.html - Sentiment correlation")
    print(f"  • contra_granger.html - Causality tests")
    if events is not None and not events.empty:
        print(f"  • contra_car_*.html - Event study results")
    print()

