"""
app.py - Fyers Options Dashboard (Streamlit)

Features:
- Index selector (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY)
- Expiry selector (auto-populated)
- CE vs PE OI line chart
- CE - PE OI difference bar chart
- CE & PE grouped bars
- Option chain interactive table
- PCR (OI) and Max Pain calculation
- Manual refresh button (safe and predictable)

HOW TO USE:
- Put your FYERS access token into Streamlit secrets:
  st.secrets["FYERS_ACCESS_TOKEN"] = "your_token_here"

Notes:
- This app expects the Fyers "option-chain" data endpoint to return a JSON with a `records.data` list
  where each item may contain "CE" and "PE" objects and "strikePrice"/"expiryDate" fields.
- If Fyers response schema differs, you may need to adapt parsing logic in `parse_option_chain`.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import math
from typing import List, Dict, Any

st.set_page_config(page_title="Options Dashboard (Fyers)", layout="wide")

# -----------------------
# Config / Constants
# -----------------------
INDICES = {
    "NIFTY 50": "NSE:NIFTY50-INDEX",
    "BANK NIFTY": "NSE:NIFTYBANK-INDEX",
    "FIN NIFTY": "NSE:NIFTYFIN-INDEX",
    "MIDCP NIFTY": "NSE:MIDCPNIFTY-INDEX"
}

FYERS_BASE = "https://api.fyers.in/data/option-chain"  # used earlier in examples


# -----------------------
# Helpers
# -----------------------
def get_fyers_token() -> str:
    """
    Read FYERS token from streamlit secrets.
    Streamlit Cloud: store in Secrets as FYERS_ACCESS_TOKEN
    Locally: create .streamlit/secrets.toml with the key FYERS_ACCESS_TOKEN
    """
    token = None
    try:
        token = st.secrets["FYERS_ACCESS_TOKEN"]
    except Exception:
        token = None
    return token


def fetch_option_chain(symbol: str) -> Dict[str, Any]:
    """Call Fyers option-chain endpoint and return parsed JSON or raise error."""
    token = get_fyers_token()
    if not token:
        raise RuntimeError("Missing FYERS_ACCESS_TOKEN in Streamlit secrets. See app instructions.")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    url = f"{FYERS_BASE}?symbol={symbol}"
    resp = requests.get(url, headers=headers, timeout=15)
    # handle errors
    if resp.status_code == 401:
        raise RuntimeError("Unauthorized (401). Your FYERS access token may be invalid/expired.")
    resp.raise_for_status()
    return resp.json()


def parse_option_chain(raw_json: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse returned JSON (Fyers style) into a DataFrame with columns:
    strike, expiry, ce_oi, ce_change_oi, ce_ltp, pe_oi, pe_change_oi, pe_ltp
    This function tries multiple possible keys to be robust.
    """
    # Attempt to find records list
    records = []
    if not raw_json:
        return pd.DataFrame()
    # Many Fyers responses use records.data or optionChain; try common places:
    if isinstance(raw_json.get("records"), dict) and raw_json["records"].get("data"):
        records = raw_json["records"]["data"]
    elif raw_json.get("optionChain"):
        records = raw_json.get("optionChain")
    elif raw_json.get("data"):
        records = raw_json.get("data")
    else:
        # fallback: try top-level list
        if isinstance(raw_json, list):
            records = raw_json
        else:
            # unknown format
            records = []

    rows = []
    for item in records:
        # strike price
        strike = item.get("strikePrice") or item.get("strike") or item.get("strike_price") or item.get("StrikePrice")
        try:
            strike = float(strike) if strike is not None else None
        except Exception:
            strike = None

        # expiry
        expiry = None
        ce = item.get("CE") or item.get("ce") or {}
        pe = item.get("PE") or item.get("pe") or {}
        expiry = ce.get("expiryDate") or pe.get("expiryDate") or item.get("expiryDate") or item.get("expiry")

        def safe_get_int(o, *keys):
            for k in keys:
                if o and (k in o):
                    try:
                        return int(o.get(k) or 0)
                    except Exception:
                        try:
                            return int(float(o.get(k)))
                        except Exception:
                            return 0
            return 0

        def safe_get_float(o, *keys):
            for k in keys:
                if o and (k in o):
                    try:
                        return float(o.get(k) or 0)
                    except Exception:
                        return 0
            return 0.0

        ce_oi = safe_get_int(ce, "openInterest", "oi", "openInterestValue")
        ce_chg = safe_get_int(ce, "changeinOpenInterest", "changeOI", "change_in_oi")
        ce_ltp = safe_get_float(ce, "lastPrice", "ltp", "lastTradedPrice")
        pe_oi = safe_get_int(pe, "openInterest", "oi", "openInterestValue")
        pe_chg = safe_get_int(pe, "changeinOpenInterest", "changeOI", "change_in_oi")
        pe_ltp = safe_get_float(pe, "lastPrice", "ltp", "lastTradedPrice")

        rows.append({
            "strike": strike,
            "expiry": expiry,
            "ce_oi": ce_oi,
            "ce_change_oi": ce_chg,
            "ce_ltp": ce_ltp,
            "pe_oi": pe_oi,
            "pe_change_oi": pe_chg,
            "pe_ltp": pe_ltp
        })

    df = pd.DataFrame(rows)
    # drop rows without strike
    df = df[df["strike"].notna()].copy()
    # sort by strike
    df = df.sort_values("strike").reset_index(drop=True)
    # calculate diff
    df["diff"] = df["ce_oi"].fillna(0) - df["pe_oi"].fillna(0)
    return df


def compute_pcr(df: pd.DataFrame) -> float:
    total_ce = int(df["ce_oi"].sum()) if not df.empty else 0
    total_pe = int(df["pe_oi"].sum()) if not df.empty else 0
    if total_ce == 0:
        return float("nan")
    return round(total_pe / (total_ce if total_ce else 1), 3)


def compute_max_pain(df: pd.DataFrame) -> float:
    """
    Simple max pain calculation:
    For each candidate settlement price (strike), compute sum of (intrinsic value * OI) for calls and puts.
    Return strike with minimum total pain.
    """
    if df.empty:
        return None
    strikes = df["strike"].unique()
    min_pain = None
    min_strike = None
    for s in strikes:
        pain = 0
        for _, r in df.iterrows():
            # call loss = max(0, strike_of_option - s) * ce_oi
            call_loss = max(0, r["strike"] - s) * r["ce_oi"]
            put_loss = max(0, s - r["strike"]) * r["pe_oi"]
            pain += call_loss + put_loss
        if min_pain is None or pain < min_pain:
            min_pain = pain
            min_strike = s
    return min_strike


# -----------------------
# Streamlit UI
# -----------------------
st.title("📈 Options Dashboard — Fyers (Streamlit)")

with st.sidebar:
    st.header("Controls")
    idx_name = st.selectbox("Select Index", list(INDICES.keys()), index=0)
    symbol = INDICES[idx_name]
    st.markdown(f"**Symbol:** `{symbol}`")

    refresh = st.button("Refresh data")
    st.write("If charts look stale, click Refresh.")
    st.markdown("---")
    show_only_near = st.checkbox("Show strikes within ± 20 strikes of ATM", value=True)
    max_strikes = st.slider("Number of strikes to show around ATM (each side)", min_value=5, max_value=100, value=20, step=5)
    show_table = st.checkbox("Show full option chain table", value=True)

st.write("---")

# Token check
token = get_fyers_token()
if not token:
    st.error("Missing FYERS_ACCESS_TOKEN. Add it to Streamlit secrets (see instructions below).")
    st.info("Locally: create .streamlit/secrets.toml with:\n\nFYERS_ACCESS_TOKEN = 'your_token_here'\n")
    st.stop()

# Fetch data
try:
    raw = fetch_option_chain(symbol)
except Exception as e:
    st.error(f"Error fetching option chain: {e}")
    st.stop()

df = parse_option_chain(raw)

if df.empty:
    st.warning("No option chain data found (empty dataframe). The response schema might differ. Check raw JSON in debug below.")
    st.write(raw)
    st.stop()

# Expiry selector
expiries = sorted([d for d in df["expiry"].unique() if d is not None])
selected_expiry = None
if expiries:
    selected_expiry = st.selectbox("Select expiry (auto)", expiries, index=0)
    # filter by expiry
    df = df[df["expiry"] == selected_expiry].copy()

# find ATM (closest strike to underlying LTP if available in the raw; fallback to midpoint)
# Fyers may not provide underlying price in option chain; we compute approximate ATM as median strike with largest combined OI
if not df.empty:
    # try to find a good ATM candidate: strike with max total OI
    df["total_oi"] = df["ce_oi"].fillna(0) + df["pe_oi"].fillna(0)
    atm_row = df.loc[df["total_oi"].idxmax()] if not df["total_oi"].isna().all() else None
    atm_strike = atm_row["strike"] if atm_row is not None else df["strike"].median()
else:
    atm_strike = None

# limit strikes relative to ATM if requested
if show_only_near and atm_strike is not None:
    lower = atm_strike - max_strikes * (df["strike"].diff().median() if len(df) > 1 else 50)
    upper = atm_strike + max_strikes * (df["strike"].diff().median() if len(df) > 1 else 50)
    # fallback simpler: select nearest N strikes on either side
    strikes_sorted = sorted(df["strike"].unique())
    try:
        pos = strikes_sorted.index(atm_strike)
    except ValueError:
        # if atm_strike not in list (rare), approximate by nearest
        pos = min(range(len(strikes_sorted)), key=lambda i: abs(strikes_sorted[i] - atm_strike))
    start_idx = max(0, pos - max_strikes)
    end_idx = min(len(strikes_sorted), pos + max_strikes + 1)
    selected_strikes = set(strikes_sorted[start_idx:end_idx])
    df = df[df["strike"].isin(selected_strikes)].copy()

# compute PCR and Max Pain
pcr = compute_pcr(df)
max_pain = compute_max_pain(df)

# Stats row
col1, col2, col3 = st.columns([1, 1, 1])
col1.metric("Displayed strikes", len(df))
col2.metric("PCR (OI)", f"{pcr}")
col3.metric("Max Pain (strike)", f"{max_pain}")

# Charts: CE vs PE line
chart_df = df.copy()
chart_df = chart_df.sort_values("strike")

fig_line = px.line(chart_df, x="strike", y=["ce_oi", "pe_oi"],
                   labels={"value": "Open Interest", "strike": "Strike"},
                   title="CE vs PE Open Interest",
                   markers=True)
fig_line.update_layout(legend_title_text="Series", height=400, margin=dict(t=60, b=20, l=40, r=20))

# Difference chart
fig_diff = px.bar(chart_df, x="strike", y="diff", title="CE - PE OI (Positive = CE > PE)",
                  labels={"diff": "CE - PE OI", "strike": "Strike"})
fig_diff.update_layout(height=360, margin=dict(t=50, b=20, l=40, r=20))

# Grouped bars chart
group_df = chart_df.melt(id_vars=["strike"], value_vars=["ce_oi", "pe_oi"], var_name="type", value_name="oi")
fig_group = px.bar(group_df, x="strike", y="oi", color="type", barmode="group", title="CE & PE OI (grouped)")
fig_group.update_layout(height=400, margin=dict(t=50, b=20, l=40, r=20))

# Render charts in layout
left_col, right_col = st.columns([2, 1])
with left_col:
    st.plotly_chart(fig_line, use_container_width=True)
    st.plotly_chart(fig_diff, use_container_width=True)

with right_col:
    st.plotly_chart(fig_group, use_container_width=True)
    st.markdown("### Top strikes by combined OI")
    topoi = df.sort_values("total_oi", ascending=False).head(10)[["strike", "ce_oi", "pe_oi", "total_oi"]]
    st.table(topoi)

# Show table
if show_table:
    st.markdown("### Full Option Chain (displayed)")
    # format large numbers with thousands separator
    df_display = df.copy()
    df_display["ce_oi"] = df_display["ce_oi"].map("{:,.0f}".format)
    df_display["pe_oi"] = df_display["pe_oi"].map("{:,.0f}".format)
    df_display["diff"] = df_display["diff"].map("{:,.0f}".format)
    df_display["ce_ltp"] = df_display["ce_ltp"].map(lambda x: f"{x:.2f}")
    df_display["pe_ltp"] = df_display["pe_ltp"].map(lambda x: f"{x:.2f}")
    st.dataframe(df_display[["strike", "ce_oi", "ce_ltp", "pe_oi", "pe_ltp", "diff"]], height=400)

# Debug raw JSON collapse
with st.expander("Raw JSON (for debugging)"):
    st.json(raw)

st.markdown("---")
st.caption("Notes: keep your FYERS access token valid. If you need auto-refresh, you can add `st.experimental_rerun()` on a timer or deploy a hosted websocket approach (I can add that for you).")
