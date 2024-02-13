import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
import pathlib
import numpy as np
from sp500.compile.cleanup import cleanup

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")

def sentiment_analyzer():
    """ 
    """

    labeled_data_filename = (
        pathlib.Path(__file__).parent / "data/Jan_24_Jan_28_Stock_News copy.csv"
    )
    labeled_data = pd.read_csv(labeled_data_filename, encoding="unicode_escape")

    labeled_data = cleanup(labeled_data)

    labeled_data["Polarity"] = None
    labeled_data["Subjectivity"] = None
    labeled_data["Predicted PNU"] = None

    for index, row in labeled_data.iterrows():
        headline = row["Cleaned Headline"]

        if isinstance(headline, float):
            headline = str(headline)

        doc = nlp(headline)
        labeled_data.at[index, "polarity"] = doc._.blob.polarity

        if doc._.blob.polarity > 0.1:
            labeled_data.at[index, "Predicted PNU"] = "P"
        elif doc._.blob.polarity < 0:
            labeled_data.at[index, "Predicted PNU"] = "N"
        else:
            labeled_data.at[index, "Predicted PNU"] = "U"

        labeled_data.at[index, "subjectivity"] = doc._.blob.subjectivity

    print("Sentiment Analyzer - Done!")

    return labeled_data


def match_comparison():

    labeled_data = sentiment_analyzer()

    labeled_data["Label & Predicted PNU match or not"] = np.where(
        labeled_data["Predicted PNU"] == labeled_data["Label (P, N, U-neutral)"],
        "match",
        "non-match",
    )

    labeled_data.to_csv("./sp500/sa/data/Finished_test_sa.csv", index=False)

    print("Sentiment Analyzer - Done!")


if __name__ == "__main__":
    match_comparison()
