from sp500.sentiment_analysis.sa import calculate_score
from sp500.headlines.scraper import headlines
from sp500.visualization.create_word_cloud import (
    create_wordcloud,
    map_stock_names_to_company_names,
)
import pathlib
import pandas as pd
from pathlib import Path


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


if __name__ == "__main__":
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "http://finviz.com/quote.ashx?t=",
    }

    news_df = headlines(headers, ["NVDA"], 2)
    _, sentiment_df = implementation(news_df)
    stock_to_company = map_stock_names_to_company_names(sentiment_df, "Company")
    visualization_dir = (
        Path(__file__).resolve().parent.parent / "visualization" / "visualization"
    )
    create_wordcloud(
        df=sentiment_df,
        company_logo_paths=None,
        stock_to_company=stock_to_company,
        visualization_dir=visualization_dir,
    )
