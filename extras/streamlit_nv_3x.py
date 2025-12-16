import streamlit as st
import yfinance as yf
import time
from datetime import datetime

st.set_page_config(page_title="NVIDIA Pair Strategy Dashboard", layout="wide")

st.title("🔁 Range-Bound NVIDIA Strategy (3x Leverage)")

# Sidebar Inputs
st.sidebar.header("🔧 Strategy Parameters")
symbol_long = st.sidebar.text_input("📈 Long (Bull) Ticker", value="3LNV.L")
symbol_short = st.sidebar.text_input("📉 Short (Bear) Ticker", value="3SNV.L")

stop_loss_percent = st.sidebar.slider("🚨 Stop-Loss %", min_value=1, max_value=20, value=5)
take_profit_multiple = st.sidebar.slider("🎯 Take-Profit Multiple", min_value=1, max_value=5, value=2)

run_strategy = st.sidebar.button("▶️ Start Strategy")

placeholder = st.empty()

@st.cache_data(ttl=1)
def get_price(symbol):
    data = yf.Ticker(symbol)
    df = data.history(period='1d', interval='1m')
    if df.empty:
        return None
    return df['Close'][-1]

if run_strategy:
    entry_long = get_price(symbol_long)
    entry_short = get_price(symbol_short)

    if not entry_long or not entry_short:
        st.error("❌ Failed to fetch initial prices. Check the ticker symbols.")
    else:
        stop_loss_long = entry_long * (1 - stop_loss_percent / 100)
        stop_loss_short = entry_short * (1 - stop_loss_percent / 100)
        target_profit = stop_loss_percent / 100 * entry_long * take_profit_multiple

        st.success(f"✅ Strategy started at {datetime.now().strftime('%H:%M:%S')}")
        st.info(f"Entry Long: {entry_long:.2f}, Stop: {stop_loss_long:.2f}")
        st.info(f"Entry Short: {entry_short:.2f}, Stop: {stop_loss_short:.2f}")

        while True:
            current_long = get_price(symbol_long)
            current_short = get_price(symbol_short)

            if not current_long or not current_short:
                placeholder.warning("⚠️ Price fetch failed. Retrying...")
                time.sleep(1)
                continue

            pnl_long = current_long - entry_long
            pnl_short = current_short - entry_short
            total_pnl = pnl_long + pnl_short

            stop_triggered = "No"
            suggestion = "Hold Both"

            if current_long < stop_loss_long:
                pnl_long = -stop_loss_percent / 100 * entry_long
                stop_triggered = f"Stop Loss Triggered for Long"
                suggestion = "❌ Sell Long"
            elif current_short < stop_loss_short:
                pnl_short = -stop_loss_percent / 100 * entry_short
                stop_triggered = f"Stop Loss Triggered for Short"
                suggestion = "❌ Sell Short"
            elif total_pnl >= target_profit:
                suggestion = "🎯 Take Profit (Close Both)"

            with placeholder.container():
                st.subheader("📊 Live Position Tracking")
                st.metric("Long Price", f"{current_long:.2f}", f"{pnl_long:.2f}")
                st.metric("Short Price", f"{current_short:.2f}", f"{pnl_short:.2f}")
                st.metric("Total P&L", f"{total_pnl:.2f}")
                st.write(f"⏱️ Last updated: {datetime.now().strftime('%H:%M:%S')}")
                st.write(f"🚨 Status: {stop_triggered}")
                st.success(f"🔄 Suggestion: {suggestion}")

            if suggestion != "Hold Both":
                st.balloons()
                break

            time.sleep(1)
