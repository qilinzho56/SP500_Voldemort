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

    # Ensure all expected columns are present, filling missing ones with NaN
    for original, _ in COLUMNS.items():
        if original not in company_profile_df:
            company_profile_df[original] = pd.NA

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

    # Define the columns needed for each section
    overview_columns = ["Ticker", "Industry", "Sector", "Market Cap"]
    finance_overview_columns = ["Revenue Growth", "Net Income", "EBITDA Margins"]
    cash_flow_columns = ["Operating Cash Flow", "Free Cash Flow"]
    profit_efficiency_columns = ["Return on Equity", "Gross Margins", "Operating Margins"]
    PE_columns = ["Trailing PE", "Forward PE", "Peg Ratio"]

    # Helper function to check and select existing columns
    def select_existing_columns(df, columns):
        existing_columns = [col for col in columns if col in df.columns]
        return df[existing_columns] if existing_columns else pd.DataFrame(columns=columns)

    overview = select_existing_columns(company_profile_df, overview_columns)
    finance_overview = select_existing_columns(company_profile_df, finance_overview_columns)
    cash_flow = select_existing_columns(company_profile_df, cash_flow_columns)
    profit_efficiency = select_existing_columns(company_profile_df, profit_efficiency_columns)
    PE = select_existing_columns(company_profile_df, PE_columns)

    return overview, finance_overview, cash_flow, profit_efficiency, PE
