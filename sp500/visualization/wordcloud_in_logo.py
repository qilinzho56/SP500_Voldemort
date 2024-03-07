import os
from pathlib import Path
import pandas as pd
from create_word_cloud import create_wordcloud

STOCK_TO_COMPANY = {
    "AAPL": "Apple",
    "NVDA": "Nvidia",
    "BA": "Boeing",
    "GOOG": "Google",
    "AMZN": "Amazon",
    "STOCK": "Stock",
}

PICTURE_PATE = os.path.dirname(os.path.abspath(__file__))

COMPANY_LOGO_PATHS = {
    "AAPL": os.path.join(PICTURE_PATE, "visualization", "apple.png"),
    "AMZN": os.path.join(PICTURE_PATE, "visualization", "amazon.png"),
    "BA": os.path.join(PICTURE_PATE, "visualization", "boeing.png"),
    "GOOG": os.path.join(PICTURE_PATE, "visualization", "google.png"),
    "NVDA": os.path.join(PICTURE_PATE, "visualization", "nvidia.png"),
}


def run_word_cloud():
    filename = Path(__file__).parent.parent / "sa/data/Finished_test_sa.csv"
    visualization_dir = Path(__file__).resolve().parent / "visualization"
    df = pd.read_csv(filename)
    create_wordcloud(
        df=df,
        company_logo_paths=COMPANY_LOGO_PATHS,
        stock_to_company=STOCK_TO_COMPANY,
        visualization_dir=visualization_dir,
    )


if __name__ == "__main__":
    run_word_cloud()
