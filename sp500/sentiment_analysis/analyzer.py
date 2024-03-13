from sp500.sentiment_analysis.sa import calculate_score
from sp500.headlines.scraper import headlines
import pathlib
import pandas as pd


def implementation(df=None):
    """
    Implement the sentiment analyzer.
    """

    if df.empty:
        labeled_data_filename = (
            pathlib.Path(__file__).parent / "data/Jan_24_Jan_28_Stock_News.csv"
        )

        labeled_data = pd.read_csv(labeled_data_filename, encoding="unicode_escape")
        df1 = calculate_score(labeled_data)
    else:
        df1 = calculate_score(df)

    return df1
