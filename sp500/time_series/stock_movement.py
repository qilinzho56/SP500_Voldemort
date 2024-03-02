import plotly.graph_objects as go
from sp500.time_series.time_series import load_ticker_data, ema_macd, obv
import pandas as pd
from datetime import datetime, timedelta


def add_macd(df, window_signal=9):
    # reference from https://plainenglish.io/blog/plot-stock-chart-using-mplfinance-in-python-9286fc69689
    df = df.copy()
    df.loc[:, "signal"] = df["MACD"].ewm(span=window_signal).mean()
    df.loc[:, "diff"] = df["MACD"] - df["signal"]
    df.loc[:, "bar_positive"] = df["diff"].apply(lambda x: x if x > 0 else 0)
    df.loc[:, "bar_negative"] = df["diff"].apply(lambda x: x if x < 0 else 0)
    return df


def add_sma(df):
    for window in [20, 50, 200]:
        sma_column = f"SMA_{window}"
        df[sma_column] = df["Adj Close"].rolling(window).mean()
    return df


def plot_stock_data_interactive(company):
    ticker_df = load_ticker_data(company)
    ticker_df.index = pd.to_datetime(ticker_df.index)
    start_date = datetime.now() - timedelta(days=400)
    df = ticker_df[ticker_df.index >= start_date]
    df_macd = ema_macd(df)
    df_with_indicators = add_macd(df_macd)
    df_with_indicators2 = obv(df)
    df_with_indicators3 = add_sma(df)

    candlestick = go.Candlestick(
        x=df_with_indicators.index,
        open=df_with_indicators["Open"],
        high=df_with_indicators["High"],
        low=df_with_indicators["Low"],
        close=df_with_indicators["Close"],
        name="Price",
    )

    macd_line = go.Scatter(
        x=df_with_indicators.index,
        y=df_with_indicators["MACD"],
        mode="lines",
        name="MACD",
        yaxis="y4",
    )
    signal_line = go.Scatter(
        x=df_with_indicators.index,
        y=df_with_indicators["signal"],
        mode="lines",
        name="Signal",
        yaxis="y4",
    )

    obv_line = go.Scatter(
        x=df_with_indicators2.index,
        y=df_with_indicators2["OBV"],
        mode="lines",
        name="OBV",
        yaxis="y2",
    )

    sma_20_line = go.Scatter(
        x=df_with_indicators3.index,
        y=df_with_indicators3["SMA_20"],
        mode="lines",
        name="SMA 20",
        line=dict(color="blue", width=2),
    )

    sma_50_line = go.Scatter(
        x=df_with_indicators3.index,
        y=df_with_indicators3["SMA_50"],
        mode="lines",
        name="SMA 50",
        line=dict(color="green", width=2),
    )

    sma_200_line = go.Scatter(
        x=df_with_indicators3.index,
        y=df_with_indicators3["SMA_200"],
        mode="lines",
        name="SMA 200",
        line=dict(color="yellow", width=2),
    )

    volume_bars = go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume",
        marker=dict(color="rgba(128, 128, 128, 0.5)"),
        yaxis="y3",
    )

    layout = go.Layout(
        title=f"Stock Movement for {company}",
        xaxis=dict(
            rangeslider=dict(visible=True), type="category", tickmode="auto", nticks=20
        ),
        yaxis=dict(title="Stock Price", domain=[0.2, 1]),
        yaxis2=dict(
            title="OBV", side="right", overlaying="y", position=0.95, domain=[0.1, 0.2]
        ),
        yaxis3=dict(
            title="Volume", side="right", overlaying="y", position=0.97, domain=[0, 0.1]
        ),
        yaxis4=dict(title="MACD", side="right", overlaying="y", position=0.999),
    )

    fig = go.Figure(
        data=[
            candlestick,
            macd_line,
            signal_line,
            obv_line,
            volume_bars,
            sma_20_line,
            sma_50_line,
            sma_200_line,
        ],
        layout=layout,
    )
    fig.show()

