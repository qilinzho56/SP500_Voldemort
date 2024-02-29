import importlib.util
import pandas as pd
import pathlib
import numpy as np
from sp500.compile.cleanup import cleanup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

if not importlib.util.find_spec("nltk.data"):
    nltk.download("vader_lexicon")

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):

    
    
    return score_dict


def create_label(score):
    """
    assign a PNU label according to its polarity score

    Inputs:
        score: the polarity score predicted by nltk
    
    Returns:
        PNU label (str)
    """

    if score >= 0.05:
        return "P"
    elif score < -0.05:
        return "N"
    else:
        return "U"


def sentiment_analyzer():
    """ 
    Using Textblob package to conduct sentiment analysis

    Inputs:

    Returns:
        labeled_data: DataFrame
    """

    labeled_data_filename = (
        pathlib.Path(__file__).parent / "data/Jan_24_Jan_28_Stock_News copy.csv"
    )
    labeled_data = pd.read_csv(labeled_data_filename, encoding="unicode_escape")


    #ignore stop words
    labeled_data = cleanup(labeled_data)

    for index, row in labeled_data.iterrows():
        score_dict = analyzer.polarity_scores(row["Cleaned Headline"])
        labeled_data.at[index, "neg"] = score_dict["neg"]
        labeled_data.at[index, "neu"] = score_dict["neu"]
        labeled_data.at[index, "pos"] = score_dict["pos"]
        labeled_data.at[index, "compoud"] = score_dict["compound"]

    labeled_data["Predicted PNU"] = labeled_data["compoud"].apply(create_label)

    print("Sentiment Analyzer - Done!")

    return labeled_data


def match_comparison():
    """
    To check if predicted PNU matches with manually labeled PNU

    Inputs:

    Returns:
        labeled_data: DataFrame
    """

    #obtain labeled data dataframe with polarity & subjectivity score and its PNU label
    labeled_data = sentiment_analyzer()

    #determine if predicted PNU matches with manually labeled PNU
    labeled_data["Label & Predicted PNU match or not"] = np.where(
        labeled_data["Predicted PNU"] == labeled_data["Label (P, N, U-neutral)"],
        "match",
        "non-match",
    )
    
    match_count = labeled_data[labeled_data["Label & Predicted PNU match or not"] == "match"].shape[0]
    match_ratio = match_count/labeled_data.shape[0]

    if match_ratio >= 0.8:
        #write a new csv file with labeled data
        labeled_data.to_csv("./sp500/sa/data/Finished_test_sa.csv", index=False)
    else:
        print(match_ratio, "is below 0.8!!")

    print("Sentiment Analyzer - Done!")

    return labeled_data


if __name__ == "__main__":
    match_comparison()
