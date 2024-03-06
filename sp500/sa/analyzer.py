from sp500.sa.sa import calculate_score
import pathlib
import pandas as pd

def implementation(df = None):
    """
    Implement the sentiment analyzer.
    """
    # Import the file

    if not df:
        labeled_data_filename = (
            pathlib.Path(__file__).parent / "data/Jan_24_Jan_28_Stock_News.csv"
        )

        labeled_data = pd.read_csv(labeled_data_filename, encoding="unicode_escape")
        calculate_score(labeled_data)
    else:
        calculate_score(df)

if __name__ == "__main__":
    implementation()
