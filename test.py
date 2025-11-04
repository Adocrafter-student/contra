# contra_run.py
import os, sys
import pandas as pd, numpy as np
import datetime as dt
import yfinance as yf
from pytrends.request import TrendReq
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# 0) Params (EDIT THESE)
# -----------------------------
TICKER = "TSLA"
MARKET = "^GSPC"
START  = "2018-01-01"     # so we capture 2018–2020 shocks
END    = "2020-12-31"     # keep it tight for the demo
KEYWORDS = ["Tesla", "Elon Musk"]
EVENTS_CSV = "events.csv" # we'll provide one below
OUTPUT_DIR = "outputs"

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_html(fig, fname):
    ensure_dir(OUTPUT_DIR)
    fig.write_html(os.path.join(OUTPUT_DIR, fname), include_plotlyjs="cdn")

def fit_ols_robust(df, with_sent, use_enhanced=False):
    X = df[["ret_mkt"]].copy()
    if with_sent:
        if use_enhanced and "sent_enhanced_z" in df.columns:
            X["sent_enhanced_z"] = df["sent_enhanced_z"]
            # Also add sentiment change for capturing shocks
            if "sent_change_z" in df.columns:
                X["sent_change_z"] = df["sent_change_z"]
        else:
            X["sent_z"] = df["sent_z"]
    X = sm.add_constant(X)
    y = df["ret_stk"]
    model = sm.OLS(y, X).fit(cov_type="HC1")
    return model

def predict(model, df):
    cols = ["ret_mkt"]
    # Check which sentiment variables are in the model
    if "sent_enhanced_z" in model.params.index:
        cols.append("sent_enhanced_z")
    if "sent_change_z" in model.params.index:
        cols.append("sent_change_z")
    if "sent_z" in model.params.index:
        cols.append("sent_z")
    
    # Build feature matrix matching the training data
    X = df[cols].copy()
    X = sm.add_constant(X, has_constant='add')
    
    # Ensure column order matches model params
    X = X[model.params.index]
    
    return model.predict(X)

# -----------------------------
# 1) Prices & returns
# -----------------------------
prices = yf.download([TICKER, MARKET], start=START, end=END, auto_adjust=True)["Close"]
prices.columns = ["stk","mkt"]
ret = prices.pct_change().dropna()
ret.index = pd.to_datetime(ret.index)
ret.rename(columns={"stk":"ret_stk","mkt":"ret_mkt"}, inplace=True)

# -----------------------------
# 2) Google Trends (sentiment proxy)
# -----------------------------
pytrends = TrendReq(hl="en-US", tz=0, retries=2, backoff_factor=0.2)
frames = []
for year in range(2018, 2021):  # 2018, 2019, 2020
    start_y = f"{year}-01-01"
    end_y   = f"{year}-12-31"
    print(f"Fetching Google Trends data for {year}...")
    pytrends.build_payload(KEYWORDS, timeframe=f"{start_y} {end_y}", geo="")
    tr_part = pytrends.interest_over_time().drop(columns=["isPartial"], errors='ignore')
    if not tr_part.empty:
        print(f"  Got {len(tr_part)} data points from {tr_part.index.min()} to {tr_part.index.max()}")
        frames.append(tr_part)
    
if not frames:
    print("Google Trends returned empty data. Try a longer window or different keywords.")
    sys.exit(1)

tr = pd.concat(frames, axis=0)
tr = tr.sort_index()  # Sort by date
tr = tr[~tr.index.duplicated(keep='first')]  # Remove duplicate dates if any
print(f"\nTotal Google Trends data: {len(tr)} points from {tr.index.min()} to {tr.index.max()}")

if tr.empty:
    print("Google Trends returned empty data. Try a longer window or different keywords.")
    sys.exit(1)

sent = tr.copy()
sent.columns = [c.lower().replace(" ","_") for c in sent.columns]
sent = sent.resample("D").mean().ffill()         # daily
sent["sent_raw"] = sent.mean(axis=1)             # average across keywords
sent["sent_z"]   = (sent["sent_raw"] - sent["sent_raw"].mean())/sent["sent_raw"].std()

# Enhanced sentiment features for better shock detection
sent["sent_change"] = sent["sent_z"].diff()  # Day-to-day sentiment change (velocity)
sent["sent_change_z"] = (sent["sent_change"] - sent["sent_change"].mean())/sent["sent_change"].std()
sent["sent_lag1"] = sent["sent_z"].shift(1)  # Yesterday's sentiment
sent["sent_vol_7d"] = sent["sent_z"].rolling(7).std()  # 7-day sentiment volatility

# Create event-amplified sentiment: spike sentiment on known event dates
try:
    events_df = pd.read_csv(EVENTS_CSV, parse_dates=["date"])
    sent["event_boost"] = 0.0
    for _, ev in events_df.iterrows():
        event_date = pd.to_datetime(ev["date"])
        # Amplify sentiment around event dates (±2 days window)
        for offset in range(-2, 3):
            check_date = event_date + pd.Timedelta(days=offset)
            if check_date in sent.index:
                # Boost magnitude: positive events get +2, negative get -2
                multiplier = 2.0 if ev["event_type"].lower() == "positive" else -2.0
                decay = 1.0 - abs(offset) * 0.2  # Decay effect away from event
                sent.loc[check_date, "event_boost"] += multiplier * decay
    print(f"Event boost applied to {(sent['event_boost'] != 0).sum()} days")
except Exception as e:
    print(f"Could not load events for boosting: {e}")
    sent["event_boost"] = 0.0

# Combine standard sentiment with event-amplified signal
sent["sent_enhanced"] = sent["sent_z"] + sent["event_boost"]
sent["sent_enhanced_z"] = (sent["sent_enhanced"] - sent["sent_enhanced"].mean())/sent["sent_enhanced"].std()

# Align to trading days
data = ret.join(sent[["sent_z", "sent_change_z", "sent_lag1", "sent_vol_7d", "sent_enhanced_z"]], how="inner").dropna()

# -----------------------------
# 3) Train / test split + OLS (3 models)
# -----------------------------
split = int(len(data)*0.7)
train, test = data.iloc[:split], data.iloc[split:]

print(f"\nTraining models on {len(train)} observations...")
print(f"Testing on {len(test)} observations...")

base   = fit_ols_robust(train, with_sent=False)  # Baseline: market only
contra = fit_ols_robust(train, with_sent=True, use_enhanced=False)  # Basic sentiment
contra_enhanced = fit_ols_robust(train, with_sent=True, use_enhanced=True)  # Enhanced sentiment with events

print(f"\nBase model features: {list(base.params.index)}")
print(f"CONTRA model features: {list(contra.params.index)}")
print(f"Enhanced model features: {list(contra_enhanced.params.index)}")

yhat_base_test    = predict(base, test)
yhat_contra_test  = predict(contra, test)
yhat_enhanced_test = predict(contra_enhanced, test)

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return mae, rmse

mae_b, rmse_b = metrics(test["ret_stk"], yhat_base_test)
mae_c, rmse_c = metrics(test["ret_stk"], yhat_contra_test)
mae_e, rmse_e = metrics(test["ret_stk"], yhat_enhanced_test)

# Save textual metrics
ensure_dir(OUTPUT_DIR)
with open(os.path.join(OUTPUT_DIR,"metrics.txt"), "w") as f:
    f.write("=" * 60 + "\n")
    f.write("MODEL COMPARISON - TESLA SENTIMENT-ALPHA ANALYSIS\n")
    f.write("=" * 60 + "\n\n")
    f.write("TRAINING SET PERFORMANCE (Adj R-squared):\n")
    f.write(f"  Baseline (Market only):        {base.rsquared_adj:.6f}\n")
    f.write(f"  CONTRA (Basic sentiment):      {contra.rsquared_adj:.6f}\n")
    f.write(f"  CONTRA Enhanced (Event boost): {contra_enhanced.rsquared_adj:.6f}\n")
    f.write(f"  Improvement vs Baseline:       {(contra_enhanced.rsquared_adj - base.rsquared_adj):.6f}\n\n")
    f.write("TEST SET PERFORMANCE:\n")
    f.write(f"  MAE  — BASE: {mae_b:.6e} | CONTRA: {mae_c:.6e} | ENHANCED: {mae_e:.6e}\n")
    f.write(f"  RMSE — BASE: {rmse_b:.6e} | CONTRA: {rmse_c:.6e} | ENHANCED: {rmse_e:.6e}\n")
    f.write(f"  MAE Improvement:  {((mae_b - mae_e) / mae_b * 100):.2f}%\n")
    f.write(f"  RMSE Improvement: {((rmse_b - rmse_e) / rmse_b * 100):.2f}%\n\n")
    f.write("=" * 60 + "\n")
    f.write("SENTIMENT-ALPHA COEFFICIENTS:\n")
    f.write("=" * 60 + "\n\n")
    
with open(os.path.join(OUTPUT_DIR,"ols_contra_basic_summary.txt"), "w") as f:
    f.write(contra.summary().as_text())
with open(os.path.join(OUTPUT_DIR,"ols_contra_enhanced_summary.txt"), "w") as f:
    f.write(contra_enhanced.summary().as_text())
    
print("\nModel Coefficients (Sentiment-Alpha θ):")
if "sent_z" in contra.params:
    print(f"  Basic sentiment θ: {contra.params['sent_z']:.6f} (p={contra.pvalues['sent_z']:.4f})")
if "sent_enhanced_z" in contra_enhanced.params:
    print(f"  Enhanced sentiment θ: {contra_enhanced.params['sent_enhanced_z']:.6f} (p={contra_enhanced.pvalues['sent_enhanced_z']:.4f})")
if "sent_change_z" in contra_enhanced.params:
    print(f"  Sentiment shock θ: {contra_enhanced.params['sent_change_z']:.6f} (p={contra_enhanced.pvalues['sent_change_z']:.4f})")

# -----------------------------
# 4) HTML Figure: Actual vs Pred (FULL timeline)
# -----------------------------
# Get predictions for both train and test
yhat_base_train = predict(base, train)
yhat_contra_train = predict(contra, train)
yhat_enhanced_train = predict(contra_enhanced, train)

# Combine train and test into full timeline
plot_df = data.copy()
plot_df["pred_base"] = pd.concat([yhat_base_train, yhat_base_test])
plot_df["pred_contra"] = pd.concat([yhat_contra_train, yhat_contra_test])
plot_df["pred_enhanced"] = pd.concat([yhat_enhanced_train, yhat_enhanced_test])
plot_df["split"] = ["Train"] * len(train) + ["Test"] * len(test)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=plot_df.index, y=plot_df["ret_stk"],
                          mode="lines", name="Actual Return", line=dict(color="black", width=2)))
fig1.add_trace(go.Scatter(x=plot_df.index, y=plot_df["pred_base"],
                          mode="lines", name="Baseline (Market Only)", line=dict(color="blue", dash="dot")))
fig1.add_trace(go.Scatter(x=plot_df.index, y=plot_df["pred_contra"],
                          mode="lines", name="CONTRA (Basic Sentiment)", line=dict(color="orange")))
fig1.add_trace(go.Scatter(x=plot_df.index, y=plot_df["pred_enhanced"],
                          mode="lines", name="CONTRA Enhanced (Event-Boosted)", line=dict(color="red", width=2)))

# Add vertical line to show train/test split
split_date = test.index[0]
fig1.add_shape(
    type="line",
    x0=split_date, x1=split_date,
    y0=0, y1=1,
    yref="paper",
    line=dict(color="green", width=2, dash="dash")
)
fig1.add_annotation(
    x=split_date, y=1, yref="paper",
    text="Train/Test Split", showarrow=False,
    yshift=10, font=dict(color="green")
)

# Mark event dates on the plot
try:
    events_for_plot = pd.read_csv(EVENTS_CSV, parse_dates=["date"])
    for _, ev in events_for_plot.iterrows():
        ev_date = pd.to_datetime(ev["date"])
        if ev_date in plot_df.index:
            color = "lightgreen" if ev["event_type"].lower() == "positive" else "lightcoral"
            fig1.add_vrect(
                x0=ev_date - pd.Timedelta(days=1), x1=ev_date + pd.Timedelta(days=1),
                fillcolor=color, opacity=0.2, line_width=0,
                annotation_text=ev["event_type"][0].upper(), annotation_position="top left"
            )
except:
    pass

fig1.update_layout(
    title=f"{TICKER}: Actual vs Predicted Returns (Full Timeline: {data.index[0].date()} to {data.index[-1].date()})",
    xaxis_title="Date", 
    yaxis_title="Daily Return",
    hovermode="x unified",
    legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01)
)
save_html(fig1, "actual_vs_pred.html")

# -----------------------------
# 5) Granger causality (1..5 lags) + HTML table
# -----------------------------
gc_df = data[["ret_stk","sent_z"]].dropna()
gtest = grangercausalitytests(gc_df, maxlag=5, verbose=False)

rows = []
for L in range(1,6):
    p = gtest[L][0]["ssr_ftest"][1]
    rows.append({"lag":L, "p_value":p})
gc_tbl = pd.DataFrame(rows)
gc_tbl.to_csv(os.path.join(OUTPUT_DIR,"granger_results.csv"), index=False)

fig2 = px.bar(gc_tbl, x="lag", y="p_value",
              title=f"{TICKER}: Granger Causality p-values (Sentiment ⇒ Returns)")
fig2.add_hline(y=0.05, line_dash="dash", annotation_text="0.05")
save_html(fig2, "granger_pvalues.html")

# New: Sentiment vs Returns Overlay Plot
fig_sent = go.Figure()

# Plot returns on primary y-axis
fig_sent.add_trace(go.Scatter(
    x=data.index, y=data["ret_stk"],
    mode="lines", name="Stock Returns",
    line=dict(color="black", width=1),
    yaxis="y1"
))

# Plot enhanced sentiment on secondary y-axis
fig_sent.add_trace(go.Scatter(
    x=data.index, y=data["sent_enhanced_z"],
    mode="lines", name="Enhanced Sentiment (Event-Boosted)",
    line=dict(color="red", width=1.5),
    yaxis="y2"
))

# Mark events
try:
    ev_mark = pd.read_csv(EVENTS_CSV, parse_dates=["date"])
    for _, ev in ev_mark.iterrows():
        ev_date = pd.to_datetime(ev["date"])
        if ev_date in data.index:
            color = "green" if ev["event_type"].lower() == "positive" else "red"
            fig_sent.add_vline(
                x=ev_date,
                line=dict(color=color, width=2, dash="dash"),
                opacity=0.5
            )
except:
    pass

fig_sent.update_layout(
    title=f"{TICKER}: Stock Returns vs Sentiment Signal with Event Markers",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Daily Return", side="left"),
    yaxis2=dict(title="Sentiment Z-Score", overlaying="y", side="right"),
    hovermode="x unified",
    legend=dict(x=0.01, y=0.99)
)
save_html(fig_sent, "sentiment_vs_returns.html")

# -----------------------------
# 6) Event Study (with auto-events fallback)
# -----------------------------
def try_load_events():
    if os.path.exists(EVENTS_CSV):
        ev = pd.read_csv(EVENTS_CSV, parse_dates=["date"])
        if "event_type" not in ev.columns:
            raise ValueError("events.csv must have 'event_type' column")
        ev["event_type"] = ev["event_type"].str.lower().str.strip()
        ev = ev.sort_values("date")
        return ev
    return None

events = try_load_events()

def auto_events_from_sentiment(n_pos=3, n_neg=3, min_gap_days=5):
    # pick top positive and negative sentiment z-score days, spaced out
    s = sent.loc[data.index, "sent_z"].copy()
    df = pd.DataFrame({"date": s.index, "sent_z": s.values}).dropna()
    df = df.sort_values("sent_z", ascending=False)

    chosen = []
    def spaced(day):
        return all(abs((day - d).days) >= min_gap_days for d in chosen)

    pos = []
    for _,r in df.iterrows():
        d = r["date"]
        if r["sent_z"] > 0 and spaced(d):
            pos.append(d)
            chosen.append(d)
            if len(pos) >= n_pos: break

    df_neg = df.sort_values("sent_z", ascending=True)
    neg = []
    for _,r in df_neg.iterrows():
        d = r["date"]
        if r["sent_z"] < 0 and spaced(d):
            neg.append(d)
            chosen.append(d)
            if len(neg) >= n_neg: break

    rows = []
    for d in neg:
        rows.append({"date": d, "event_type":"negative", "description":"auto: low sentiment day"})
    for d in pos:
        rows.append({"date": d, "event_type":"positive", "description":"auto: high sentiment day"})
    return pd.DataFrame(rows).sort_values("date")

if events is None or events.empty:
    events = auto_events_from_sentiment()

# core market model functions
def market_model_params(up_to_date):
    # estimation window: last 120 trading days ending at -21
    df = data.loc[:up_to_date].iloc[:-21]  # cut off 21 days pre-event
    est = df.iloc[-120:] if len(df) >= 120 else df
    X = sm.add_constant(est["ret_mkt"])
    y = est["ret_stk"]
    m = sm.OLS(y, X).fit()
    return m.params["const"], m.params["ret_mkt"]

def window_data(center_date, k=5):
    idx = data.index
    # nearest trading day
    if center_date not in idx:
        center_date = idx[idx.get_indexer([center_date], method="nearest")[0]]
    loc = idx.get_loc(center_date)
    lo = max(loc - k, 0)
    hi = min(loc + k, len(idx)-1)
    out = data.iloc[lo:hi+1].copy()
    # tau centered on event (0 at event index)
    out["tau"] = np.arange(lo, hi+1) - loc
    return out

# compute AR & CAR around events
rows = []
for _,ev in events.iterrows():
    try:
        a,b = market_model_params(ev["date"])
        W = window_data(ev["date"], k=5)
        W["er"] = a + b*W["ret_mkt"]
        W["ar"] = W["ret_stk"] - W["er"]
        W["event_type"] = ev["event_type"]
        W["event_date"] = pd.to_datetime(ev["date"]).date()
        rows.append(W[["event_date","tau","ar","event_type"]])
    except Exception as e:
        print("Skip event", ev["date"], "->", e)

ES = pd.concat(rows, ignore_index=True)
CAR = (ES.sort_values(["event_date","tau"])
         .assign(car=ES.groupby("event_date")["ar"].transform(np.cumsum)))

# aggregate by event_type
agg = (CAR.groupby(["event_type","tau"])
          .agg(mean_AR=("ar","mean"),
               mean_CAR=("car","mean"),
               n=("ar","count"))
          .reset_index())

agg.to_csv(os.path.join(OUTPUT_DIR,"event_agg.csv"), index=False)

# HTML CAR plots for each polarity
for et in agg["event_type"].unique():
    sub = agg[agg["event_type"]==et]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["tau"], y=sub["mean_CAR"], mode="lines", name="Mean CAR"))
    fig.add_hline(y=0, line_dash="dash")
    fig.add_vline(x=0, line_dash="dash")
    fig.update_layout(title=f"{TICKER}: Mean CAR [-5,+5] (event_type={et}, n={int(sub['n'].max())})",
                      xaxis_title="Days relative to event (τ)", yaxis_title="CAR")
    save_html(fig, f"car_{et}.html")

print(f"Done. HTML files saved in: {OUTPUT_DIR}/")
