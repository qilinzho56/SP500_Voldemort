import matplotlib
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_and_plot_prices(stocks, start_date, end_date, visualization_dir):
    """
    Fetches stock prices, plots open and close price comparison for each stock separately
    
    Parameters:
    - stocks (list of strings): List of stocks
    - start_date (string): Start date (YYYY-MM-DD)
    - end_date (string): End date (YYYY-MM-DD)
    - visualization_dir (string): Directory path to save the output plot.
    """
    stock_data = yf.download(stocks, start=start_date, end=end_date)
    stock_data.index = pd.to_datetime(stock_data.index)
    stock_data = stock_data[stock_data.index.dayofweek < 5]
    
    for stock in stocks:
        plt.figure(figsize=(10, 6))
        if stock in stock_data.columns.get_level_values(1):
            plt.plot(stock_data['Adj Close'][stock], label=f"{stock} Close", color='green')
            plt.plot(stock_data['Open'][stock], label=f"{stock} Open", color='red')

        plt.title(f"{stock} Opening and Closing Prices")
        plt.xlabel("Date")
        plt.ylabel("Price in USD ($)")
        plt.legend()
        plt.tight_layout()

        last_inclusive_date = (pd.to_datetime(end_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        output_filename = f"{stock}_open_close_price_{start_date}_to_{last_inclusive_date}.png"
        output_path = os.path.join(visualization_dir, output_filename)

        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
        plt.close()

def main():
    stocks = ["AAPL", "AMZN", "BA", "GOOG", "NVDA"]
    start_date = "2024-01-24"
    end_date = "2024-01-30"

    visualization_dir = os.path.join(os.path.dirname(__file__), 'visualization')
    os.makedirs(visualization_dir, exist_ok=True)
    fetch_and_plot_prices(stocks, start_date, end_date, visualization_dir)

if __name__ == "__main__":
    main()
