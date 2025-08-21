import yfinance as yf
import numpy as np
from scipy.stats import norm
import time

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European call option.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call

def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European put option.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put

def compute_volatility(ticker, period='1y'):
    """
    Compute annualized historical volatility from log returns.
    """
    data = yf.download(ticker, period=period)
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    vol = data['Log_Return'].std() * np.sqrt(252)  # 252 trading days/year
    return vol

# User-configurable parameters
ticker = 'AAPL'  # Example: Apple stock; change to your desired ticker
K = 150.0        # Strike price
T = 0.5          # Time to expiration in years (e.g., 6 months)
r = 0.05         # Risk-free rate (5%; adjust based on current rates)
update_interval = 60  # Seconds between updates

# Compute initial volatility
sigma = compute_volatility(ticker)
print(f"Computed volatility (sigma): {sigma:.4f}")

# Monitoring loop
print(f"Monitoring {ticker} option prices (Call/Put) with K={K}, T={T}, r={r}, sigma={sigma:.4f}")
while True:
    try:
        # Fetch current stock price
        stock = yf.Ticker(ticker)
        current_data = stock.history(period='1d')
        if not current_data.empty:
            S = current_data['Close'].iloc[-1]  # Latest closing price
        else:
            print("Failed to fetch data. Retrying...")
            time.sleep(update_interval)
            continue

        # Compute option prices
        call_price = black_scholes_call(S, K, T, r, sigma)
        put_price = black_scholes_put(S, K, T, r, sigma)

        # Output
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Current stock price: {S:.2f} | Predicted Call: {call_price:.2f} | Predicted Put: {put_price:.2f}")

        # Wait for next update
        time.sleep(update_interval)
    except Exception as e:
        print(f"Error: {e}. Retrying in {update_interval} seconds...")
        time.sleep(update_interval)
