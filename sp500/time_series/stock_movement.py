from sp500.time_series.time_series import load_ticker_data, ema_macd, obv
import pandas as pd
import mplfinance as mpf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def add_macd(df, window_signal=9):
    # reference from https://plainenglish.io/blog/plot-stock-chart-using-mplfinance-in-python-9286fc69689
    df = df.copy()
    df.loc[:, "signal"] = df["MACD"].ewm(span=window_signal).mean()
    df.loc[:, "diff"] = df["MACD"] - df["signal"]
    df.loc[:, "bar_positive"] = df["diff"].apply(lambda x: x if x > 0 else 0)
    df.loc[:, "bar_negative"] = df["diff"].apply(lambda x: x if x < 0 else 0)
    return df


def plot_stock_data(company):
    """
    Plots real-time stock data for a given company using mplfinance

    Parameters:
    - company: (string) the ticker symbol of the company
    """
    ticker_df = load_ticker_data(company)
    ticker_df.index = pd.to_datetime(ticker_df.index)
    start_date = datetime.now() - timedelta(days=126)
    halfyear_data = ticker_df[ticker_df.index >= start_date]

    df = ema_macd(halfyear_data)
    macd = add_macd(df)
    obv_data = obv(halfyear_data)
    plots = [
        mpf.make_addplot(
            (macd["MACD"]), type="line", color="blue", panel=2, ylabel="MACD"
        ),
        mpf.make_addplot((macd["signal"]), type="line", color="grey", panel=2),
        mpf.make_addplot((macd["bar_positive"]), type="bar", color="green", panel=2),
        mpf.make_addplot((macd["bar_negative"]), type="bar", color="red", panel=2),
        mpf.make_addplot(
            (obv_data["OBV"]), type="line", color="pink", panel=3, ylabel="OBV"
        ),
    ]

    plot_color = mpf.make_marketcolors(up="green", down="red", volume="inherit")
    plot_style = mpf.make_mpf_style(marketcolors=plot_color)
    plot_args = {
        "figratio": (5, 2),
        "panel_ratios": (5, 1, 2, 1),
        "figscale": 1.5,
        "type": "candle",
        "mav": (3, 6, 9),
        "volume": True,
        "volume_panel": 1,
        "addplot": plots,
        "tight_layout": True,
        "style": plot_style,
    }

    fig, axes = mpf.plot(halfyear_data, **plot_args, returnfig=True)

    ax = axes[0]
    red_patch = mpatches.Patch(color="red", label="Price Down")
    green_patch = mpatches.Patch(color="green", label="Price Up")
    ax.legend(handles=[green_patch, red_patch], loc="upper right")

    latest_price = halfyear_data["Close"].iloc[-1]
    title_text = (
        f"Stock Movement for {company.upper()}\nLatest Price: ${latest_price:.2f}"
    )
    box_props = dict(boxstyle="square", facecolor="white", edgecolor="black", alpha=0.8)
    fig.suptitle(
        title_text,
        fontsize=11,
        fontweight="bold",
        ha="left",
        x=0.13,
        y=0.95,
        bbox=box_props,
    )

    plt.show()
