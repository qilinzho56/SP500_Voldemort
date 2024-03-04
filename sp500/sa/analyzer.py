from sp500.sa.sa import calculate_score
import pathlib

def implementation(labeled_data_filename):
    """
    Implement the sentiment analyzer.
    """
    # Import the file
    calculate_score(labeled_data_filename)

if __name__ == "__main__":
    labeled_data_filename = (
        pathlib.Path(__file__).parent / "data/New_data.csv.csv"
    )
    implementation(labeled_data_filename)
