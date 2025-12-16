import yfinance as yf
import time
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
debugpy.breakpoint()
# Symbols (replace with real tickers if using another API)
symbol_long = "3LNV.L"  # Example: +3x NVIDIA (London)
symbol_short = "3SNV.L"  # Example: -3x NVIDIA (London)

# Strategy parameters
STOP_LOSS_PERCENT = 5
TAKE_PROFIT_MULTIPLE = 2

def get_price(symbol):
    data = yf.Ticker(symbol)
    df = data.history(period='1d', interval='1m')
    if df.empty:
        return None
    return df['Close'][-1]

# Entry prices
entry_long = get_price(symbol_long)
entry_short = get_price(symbol_short)

if not entry_long or not entry_short:
    print("Failed to fetch initial prices.")
    exit()

stop_loss_long = entry_long * (1 - STOP_LOSS_PERCENT / 100)
stop_loss_short = entry_short * (1 - STOP_LOSS_PERCENT / 100)

print(f"Entry (Long): {entry_long} | Stop: {stop_loss_long}")
print(f"Entry (Short): {entry_short} | Stop: {stop_loss_short}")

while True:
    current_long = get_price(symbol_long)
    current_short = get_price(symbol_short)

    if not current_long or not current_short:
        print("Price fetch error. Retrying...")
        time.sleep(60)
        continue

    pnl_long = current_long - entry_long
    pnl_short = current_short - entry_short

    # Exit if either hit stop loss
    if current_long < stop_loss_long:
        print("Long position stopped out.")
        pnl_long = -STOP_LOSS_PERCENT / 100 * entry_long
    if current_short < stop_loss_short:
        print("Short position stopped out.")
        pnl_short = -STOP_LOSS_PERCENT / 100 * entry_short

    total_pnl = pnl_long + pnl_short

    # Exit if total profit > 2x stop loss
    target_profit = STOP_LOSS_PERCENT / 100 * entry_long * TAKE_PROFIT_MULTIPLE
    if total_pnl >= target_profit:
        print(f"Profit target hit! Total P&L: {total_pnl:.2f}")
        break

    print(f"Current P&L | Long: {pnl_long:.2f}, Short: {pnl_short:.2f}, Total: {total_pnl:.2f}")
    time.sleep(60)  # Wait 1 minute
