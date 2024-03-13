import os
from pathlib import Path
import pandas as pd
from sp500.sentiment_analysis.visualization.create_word_cloud import create_wordcloud, map_stock_names_to_company_names


PICTURE_PATE = os.path.dirname(os.path.abspath(__file__))

COMPANY_LOGO_PATHS = {
    "AAPL": os.path.join(PICTURE_PATE, "saved_pictures", "apple.png"),
    "AMZN": os.path.join(PICTURE_PATE, "saved_pictures", "amazon.png"),
    "BA": os.path.join(PICTURE_PATE, "saved_pictures", "boeing.png"),
    "GOOG": os.path.join(PICTURE_PATE, "saved_pictures", "google.png"),
    "NVDA": os.path.join(PICTURE_PATE, "saved_pictures", "nvidia.png"),
}


def run_word_cloud():
    filename = Path(__file__).parent.parent / "data/Finished_test_sa.csv"
    df = pd.read_csv(filename)
    visualization_dir = Path(__file__).resolve().parent / "saved_pictures"
    stock_to_company = map_stock_names_to_company_names(df, "Company")
    create_wordcloud(
        df=df,
        company_logo_paths=COMPANY_LOGO_PATHS,
        stock_to_company=stock_to_company,
        visualization_dir=visualization_dir,
    )


if __name__ == "__main__":
    run_word_cloud()
