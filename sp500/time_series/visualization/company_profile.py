import yfinance as yf
import pandas as pd
from datetime import datetime

COLUMNS = {
    "symbol": "Ticker",
    "industry": "Industry",
    "sector": "Sector",
    "marketCap": "Market Cap",
    "currentPrice": "Current Price",
    "fiftyDayAverage": "50 Day Average",
    "twoHundredDayAverage": "200 Day Average",
    "revenueGrowth": "Revenue Growth",
    "netIncomeToCommon": "Net Income",
    "operatingCashflow": "Operating Cash Flow",
    "freeCashflow": "Free Cash Flow",
    "ebitdaMargins": "EBITDA Margins",
    "returnOnEquity": "Return on Equity",
    "grossMargins": "Gross Margins",
    "operatingMargins": "Operating Margins",
    "trailingPE": "Trailing PE",
    "forwardPE": "Forward PE",
    "pegRatio": "Peg Ratio",
}


def profile_check(company):
    """
    Creata a dataframe to record company profile

    Parameters
    ----------
    company: a string name

    Returns
    -------
    company_profile_df: a company profile dataframe
    """
    ticker = yf.Ticker(company)
    company_profile = pd.Series(ticker.info, index=None)
    company_profile_df = pd.DataFrame(company_profile).transpose()
    company_profile_df = company_profile_df.rename(
        columns=COLUMNS, index={0: datetime.now().strftime("%Y-%m-%d")}
    )
    company_profile_df = company_profile_df[list(COLUMNS.values())]

    return company_profile_df


def company_index_exhibit(company):
    """
    Divides the company profile to four major financial sections

    Parameters
    ----------
    company: a string name

    Returns
    -------
    overview, finance_overview, cash_flow, profit_efficiency, PE:
    four dataframes with finanical info from four perspectives
    """
    company_profile_df = profile_check(company)
    overview = company_profile_df[["Ticker", "Industry", "Sector", "Market Cap"]]
    finance_overview = company_profile_df[
        ["Revenue Growth", "Net Income", "EBITDA Margins"]
    ]
    cash_flow = company_profile_df[["Operating Cash Flow", "Free Cash Flow"]]
    profit_efficiency = company_profile_df[
        ["Return on Equity", "Gross Margins", "Operating Margins"]
    ]
    PE = company_profile_df[["Trailing PE", "Forward PE", "Peg Ratio"]]

    return overview, finance_overview, cash_flow, profit_efficiency, PE
