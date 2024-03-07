import pandas as pd
import requests
import numpy as np
import yfinance as yf
from fredapi import Fred
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

HORIZON = {"2D": 2, "5D": 5, "3M": 63, "6M": 126, "1Y": 252}


# Import Macro Factors
def db_scrapeindicator(url):
    """
    Scrape and extract related macro-economic indicators and values
    using DBnomics API

    Parameters
    ----------
    url: url link for the specific macro observation

    Returns
    -------
    indicator_df: a dataframe with DBnomics macro indicator values and time as index
    """
    response = requests.get(url)
    r_json = response.json()
    periods = r_json["series"]["docs"][0]["period"]
    values = r_json["series"]["docs"][0]["value"]
    dataset = r_json["series"]["docs"][0]["dataset_name"]
    indicator_df = pd.DataFrame(values, index=periods)
    indicator_df.columns = [dataset]
    indicator_df.index = pd.to_datetime(indicator_df.index)

    return indicator_df


def fred_scrapeindicator(fred_key, series_name):
    """
    Scrape and extract related macro-economic indicators and values
    using FRED API

    Parameters
    ----------
    fred_key: FRED API key
    series_name: FRED series name

    Returns
    -------
    indicator_df: a dataframe with FRED macro indicator values and time as index
    """
    fred = Fred(fred_key)
    series = fred.get_series(series_name)

    indicator_df = pd.DataFrame(series)

    # For monthly data, create a year-month column to facilitate merging with data of daily frequency
    if fred.search(series_name)["frequency_short"][0] == "M":
        indicator_df["Year_Month"] = indicator_df.index.to_period("M")

    return indicator_df


def fetch_macro_indicators():
    """
    Fetch macro indicators using the scrapeindicator help function

    Parameters
    ----------
    fred_key: FRED API key

    Returns
    -------
    macro_indicators: a dictionary that maps each dataframe to its corresponding indicator
    """
    db_macro_indicators = {
        "us_1m_rate": "https://api.db.nomics.world/v22/series/FED/H15/RIFLGFCM01_N.B?observations=1",
        "us_3m_rate": "https://api.db.nomics.world/v22/series/FED/H15/RIFLGFCM03_N.B?observations=1",
        "us_1y_rate": "https://api.db.nomics.world/v22/series/FED/H15/RIFLGFCY01_N.B?observations=1",
        "us_dollar_exch": "https://api.db.nomics.world/v22/series/FED/H10/JRXWTFB_N.B?observations=1",
    }

    macro_indicators = {}
    for key, url in db_macro_indicators.items():
        macro_indicators[key] = db_scrapeindicator(url)

    fred_key = input("What is your FRED key? ")
    fred_macro_indicators = ["T10YIE", "UNRATE"]

    for key in fred_macro_indicators:
        macro_indicators[key] = fred_scrapeindicator(fred_key=fred_key, series_name=key)

    return macro_indicators


def preprocess_macro_data(macro_indicators):
    """
    Updates dataframe columns in macro_indicators

    Parameters
    ----------
    macro_indicators: a indicator-dataframe dictionary
    """
    macro_indicators["us_1m_rate"].rename(
        columns={"Selected Interest Rates": "US_1M_Interest_Rate"}, inplace=True
    )
    macro_indicators["us_3m_rate"].rename(
        columns={"Selected Interest Rates": "US_3M_Interest_Rate"}, inplace=True
    )
    macro_indicators["us_1y_rate"].rename(
        columns={"Selected Interest Rates": "US_1Y_Interest_Rate"}, inplace=True
    )
    macro_indicators["T10YIE"].rename(
        columns={0: "US_TC_10Y_Inflation_Rate"}, inplace=True
    )
    macro_indicators["UNRATE"].rename(columns={0: "US_Unemployment_Rate"}, inplace=True)


# Company Past Stock Performance
def load_ticker_data(company):
    """
    Load company ticker data

    Parameters
    ----------
    company: a string of company name

    Returns
    -------
    ticker_df: a dataframe with ticker's daily general price and volume information
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    ticker_df = yf.download(company, start="2000-01-01", end=end_date, interval="1d")

    return ticker_df


# Financial indices: sma, ema_macd, rsi, obv
def sma(ticker_df):
    for horizon, span in HORIZON.items():
        sma = ticker_df["Adj Close"].rolling(span).mean()
        sma_column = f"SMA_Ratio_{horizon}"
        ticker_df[sma_column] = ticker_df["Adj Close"] / sma

    return ticker_df


def ema_macd(ticker_df):
    for horizon, span in HORIZON.items():
        ema = ticker_df["Adj Close"].ewm(span=span, adjust=False).mean()
        ema_column = f"EMA_Ratio_False_{horizon}"
        ticker_df[ema_column] = ticker_df["Adj Close"] / ema

    ticker_df["MACD"] = (
        ticker_df["Adj Close"].ewm(span=26, adjust=False).mean()
        - ticker_df["Adj Close"].ewm(span=12, adjust=False).mean()
    )

    return ticker_df


# referenced from https://medium.com/@farrago_course0f/using-python-and-rsi-to-generate-trading-signals-a56a684fb1
def rsi(ticker_df):
    for horizon, span in HORIZON.items():
        delta = ticker_df["Adj Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(com=span - 1, min_periods=span).mean()
        avg_loss = loss.ewm(com=span - 1, min_periods=span).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_column = f"RSI_{horizon}"
        ticker_df[rsi_column] = rsi

    return ticker_df


def obv(ticker_df):
    ticker_df["OBV"] = (
        np.sign(ticker_df["Adj Close"].diff()) * ticker_df["Volume"]
    ).cumsum()
    return ticker_df


def update_ticker_price_indicators(ticker_df):
    sma(ticker_df)
    ema_macd(ticker_df)
    rsi(ticker_df)
    obv(ticker_df)


def ticker_macro_merge(ticker_df, macro_indicators):
    """
    Merger ticker_df with dataframes in macro_indicators

    Parameters
    ----------
    ticker_df: ticker dataframe
    macro_indicators: a dictionary with a macro_indicator as the key, the dataframe as the value

    Returns
    -------
    ticker_df: a merged ticker dataframe
    """
    for indicator, dataframe in macro_indicators.items():
        # Special Case: merging with data on a monthly basis
        if indicator == "UNRATE":
            ticker_df["Year_Month"] = ticker_df.index.to_period("M")
            index_reset = ticker_df.reset_index()
            ticker_df = index_reset.merge(dataframe, on="Year_Month", how="left")
            ticker_df.drop(columns=["Year_Month"], inplace=True)
            ticker_df = ticker_df.set_index("Date")
        else:
            ticker_df = ticker_df.merge(
                dataframe, left_index=True, right_index=True, how="left"
            )

    return ticker_df


def test_train_prep(company, news_data=None):
    """
     Prepare time-series training and testing data for the stock movement prediction model

     Parameters
     ----------
     company: a string name

     Returns
     -------
    all_data, X_train, X_test, y_train, y_test:  a 5-tuple containing a scaled and preprocessed dataframe
    and the training/testing numpy arrays
    """
    ticker_df = load_ticker_data(company)
    macro_indicators = fetch_macro_indicators()
    preprocess_macro_data(macro_indicators)
    combined_df = ticker_macro_merge(ticker_df, macro_indicators)

    combined_df = combined_df.apply(pd.to_numeric, errors="coerce")

    combined_df["Tomorrow"] = combined_df["Adj Close"].shift(-1)
    combined_df["Target"] = (combined_df["Tomorrow"] > combined_df["Adj Close"]).astype(
        int
    )

    combined_df = combined_df.fillna(
        combined_df.rolling(window=30, min_periods=1).mean()
    )

    combined_df = combined_df.dropna()

    y = combined_df["Target"]

    X = combined_df[
        combined_df.columns.difference(
            ["Target", "Close Adj Close", "Adj Close", "Tomorrow", "Volume"]
        )
    ]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

    all_data = pd.concat([X_scaled_df, y], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42
    )

    return all_data, X_train, X_test, y_train, y_test

