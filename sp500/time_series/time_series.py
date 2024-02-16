import pandas as pd
import requests
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


HORIZON = {
        "2D": 2,  
        "5D": 5,
        "3M": 63,
        "6M": 126,
        "1Y": 252
    }

# Import Macro Factors
def scrapeindicator(url):
    response = requests.get(url)
    r_json = response.json()
    periods = r_json["series"]["docs"][0]["period"]
    values = r_json["series"]["docs"][0]["value"]
    dataset = r_json["series"]["docs"][0]["dataset_name"]
    indicators_df = pd.DataFrame(values,index=periods)
    indicators_df.columns = [dataset]
    return indicators_df


def fetch_macro_indicators():
    macro_indicators = {
        "us_1m_rate": "https://api.db.nomics.world/v22/series/FED/H15/RIFLGFCM01_N.B?observations=1",
        "us_3m_rate": "https://api.db.nomics.world/v22/series/FED/H15/RIFLGFCM03_N.B?observations=1",
        "us_1y_rate": "https://api.db.nomics.world/v22/series/FED/H15/RIFLGFCY01_N.B?observations=1",
        "us_dollar_exch": "https://api.db.nomics.world/v22/series/FED/H10/JRXWTFB_N.B?observations=1",
       # "qtr_inflation": "https://api.db.nomics.world/v22/series/OECD/MEI/USA.CSINFT02.STSA.Q?observations=1",
       # "qtr_unemployment": "https://api.db.nomics.world/v22/series/OECD/MEI/USA.LRUN64TT.STSA.Q?observations=1",
    }

    for key, url in macro_indicators.items():
        macro_indicators[key] = scrapeindicator(url)
    return macro_indicators


def quarter_to_date(quarter):
    year, qtr = quarter.split('-')
    if qtr == 'Q1':
        return f'{year}-01-01'
    elif qtr == 'Q2':
        return f'{year}-04-01'
    elif qtr == 'Q3':
        return f'{year}-07-01'
    elif qtr == 'Q4':
        return f'{year}-10-01'
    else:
        return None
    

def preprocess_macro_data(macro_indicators):
   # macro_indicators["qtr_unemployment"].index = macro_indicators["qtr_unemployment"].index.map(quarter_to_date)
    macro_indicators["us_1m_rate"].rename(columns={'Selected Interest Rates': 'US_1M_Interest_Rate'}, inplace=True)
    macro_indicators["us_3m_rate"].rename(columns={'Selected Interest Rates': 'US_3M_Interest_Rate'}, inplace=True)
    macro_indicators["us_1y_rate"].rename(columns={'Selected Interest Rates': 'US_1Y_Interest_Rate'}, inplace=True)
   # macro_indicators["qtr_inflation"].rename(columns={'Main Economic Indicators Publication': 'Quarterly_Inflation'}, inplace=True)
   # macro_indicators["qtr_unemployment"].rename(columns={"Main Economic Indicators Publication": "Quarterly_UnEmploy"}, inplace=True)

    for dataframe in macro_indicators.values():
        dataframe.index = pd.to_datetime(dataframe.index)
    
    return macro_indicators


# Company Past Stock Performance
def load_ticker_data(company):
    end_date = datetime.now().strftime("%Y-%m-%d")
    ticker_df = yf.download(company, start="2010-01-01", end=end_date, interval="1d")

    return ticker_df


def sma(ticker_df):
    for horizon, span in HORIZON.items():
        sma = ticker_df["Adj Close"].rolling(span).mean()
        sma_column = f"SMA_Ratio_{horizon}"
        ticker_df[sma_column] = ticker_df["Adj Close"] / sma
        
    return ticker_df

def ema_macd(ticker_df):
    for horizon, span in HORIZON .items():
        ema = ticker_df["Adj Close"].ewm(span=span, adjust=False).mean()
        ema_column = f"EMA_Ratio_False_{horizon}"
        ticker_df[ema_column] = ticker_df["Adj Close"] / ema  

    ticker_df["MACD"] = ticker_df["Adj Close"].ewm(span=26, adjust=False).mean() - ticker_df["Adj Close"].ewm(span=12, adjust=False).mean()
        
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
    ticker_df["OBV"] = (np.sign(ticker_df['Adj Close'].diff()) * ticker_df['Volume']).cumsum()
    return ticker_df


def update_ticker_price_indicators(ticker_df):
    sma(ticker_df)
    ema_macd(ticker_df)
    rsi(ticker_df)
    obv(ticker_df)

 
def ticker_macro_merge(ticker_df, macro_indicators): 
    for indicator, dataframe in macro_indicators.items():
        ticker_df = ticker_df.merge(dataframe, left_index=True, right_index=True, how='left')
        """
        if "qtr" in indicator:
            datetime_index = ticker_df.index
            ticker_df["Quarter"] = ticker_df.index.strftime("%Y-%m")
            dataframe["Quarter"] = dataframe.index.strftime("%Y-%m")
            ticker_df = ticker_df.merge(dataframe, on="Quarter", how="left").set_index(datetime_index)
        """
   
    return ticker_df


def test_train_prep(company):
    ticker_df = load_ticker_data(company)
    macro_indicators = preprocess_macro_data(fetch_macro_indicators())
    combined_df = ticker_macro_merge(ticker_df, macro_indicators)

    combined_df["Tomorrow"] = combined_df["Adj Close"].shift(-1)
    combined_df["Target"] = (combined_df["Tomorrow"] > combined_df["Adj Close"]).astype(int)  
    combined_df.replace('NA', pd.NA, inplace=True)
    combined_df = combined_df.dropna()
    
    y = combined_df["Target"]

    X = combined_df[combined_df.columns.difference(["Target", "Close Adj Close", "Adj Close", "Tomorrow", "Volume"])]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test




