import importlib.util
import pandas as pd
import pathlib
from sp500.compile.cleanup import cleanup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sp500.sa.test import label_score
from sp500.sa.train_classifier import train_classifier
import nltk
from textblob import TextBlob

if not importlib.util.find_spec("nltk.data"):
    nltk.download("vader_lexicon")

analyzer = SentimentIntensityAnalyzer()


def extract_sentiment_words(df, sentiment_column, sentiment_score):
    """
    Extract the words within the specificed sentiment score from the
    givern Dataframe and adds a Score column

    Inputs:
        df (DataFrame): DataFrame containing Word columns and sentiment
            columns for filtering
        sentiment_column (str): the name of the sentiment column (such as
            "Negative" or "Positive")
        sentiment_score (int): sentiment score (such as -1 or 1)

    Return:
        sentiment_words_dict (dict): Dictionary containing words that meet
            the condition and a Score column
    """

    sentiment_words = df[(df[sentiment_column] > 0)]
    sentiment_words = sentiment_words[["Word"]]
    sentiment_words["Score"] = sentiment_score
    sentiment_words_dict = sentiment_words.set_index("Word")["Score"].to_dict()

    return sentiment_words_dict


def fine_tune_SA():
    """
    Fine-tunes the SentimentIntensityAnalyzer (SA) by updating its lexicon based on
    the Loughran and McDonald's dictionaries.

    Returns:
        None
    """

    LM_master_dict_filenmae = (
        pathlib.Path(__file__).parent
        / "data/Loughran-McDonald_MasterDictionary_1993-2021.csv"
    )
    LM_master_dict = pd.read_csv(LM_master_dict_filenmae)

    # Pre-process of DataFrame
    LM_master_dict = LM_master_dict[
        (LM_master_dict["Doc Count"] > 300) | (LM_master_dict["Word Count"] > 300)
    ]
    LM_master_dict["Word"] = LM_master_dict["Word"].str.lower()

    # Set sentiment score
    sentiment_score = {"Negative": -10, "Positive": 10, "Uncertainty": 1}

    for sentiment, score in sentiment_score.items():
        new_dict = extract_sentiment_words(LM_master_dict, sentiment, score)
        analyzer.lexicon.update(new_dict)

    print(
        f"Your SentimentIntensityAnalyzer has been fine-tuned based on Loughran and McDonald's dictionaries, "
        f"with a positive score of {sentiment_score['Positive']}, a negative score of {sentiment_score['Negative']}, "
        f"and a neutral score of {sentiment_score['Uncertainty']} for this round."
    )


def score_grouping(score):
    """
    Categorize a score into one of the predefined groups based on given cutoff values.

    Inputs:
        score (float): The numerical score to be categorized.

    Returns:
        str: The category label for the input score.
    """

    cutoff1 = -0.50
    cutoff2 = -0.05
    cutoff3 = -cutoff2
    cutoff4 = -cutoff1

    if score > cutoff4 and score <= 1.0 :
        return "Positive High"
    elif score > cutoff3 and score <= cutoff4:
        return "Positive Low"
    elif score > cutoff1 and score <= cutoff2:
        return "Negative Low"
    elif score >= -1.0 and score <= cutoff1:
        return "Negative High"
    else:
        return "Neutral"
    
def create_label(row, classifier):
    """
    Find corresponding label of a combination of segmentation compound and segmentation textblob

    Returns:
        PNU label (str)
    """
    combination = (row["Segmentation Compound"] , row["Segmentation Textblob"])
    return classifier.get(combination)


def sentiment_analyzer(labeled_data_filename):
    """
    Function to conduct sentiment analysis using NLTK package.

    Reads labeled data from a CSV file, cleans it, and performs sentiment analysis.
    Calculates sentiment scores for each headline, predicts sentiment labels.

    Inputs:
        None

    Returns:
        labeled_data (DataFrame)
    """

    labeled_data = pd.read_csv(labeled_data_filename, encoding="unicode_escape")

    # Clean up labeled data (convert to lowercase, skip stop words)
    labeled_data = cleanup(labeled_data)
    fine_tune_SA()

    # Iterate through each cleaned headline and calcualte the polarity score
    # neg, neu, pos represent the proportion of negative, neutral, and positive emotions in the sentence
    # compound represents the polarity score
    
    for index, row in labeled_data.iterrows():
        score_dict = analyzer.polarity_scores(row["Cleaned Headline"])
        labeled_data.at[index, "neg"] = score_dict["neg"]
        labeled_data.at[index, "neu"] = score_dict["neu"]
        labeled_data.at[index, "pos"] = score_dict["pos"]
        labeled_data.at[index, "compound"] = score_dict["compound"]
        blob = TextBlob(row["Cleaned Headline"])
        labeled_data.at[index, "textblob polarity"] = blob.sentiment_assessments.polarity

    # Convert the score intot different groups
    labeled_data["Segmentation Compound"] = labeled_data["compound"].apply(score_grouping)
    labeled_data["Segmentation Textblob"] = labeled_data["textblob polarity"].apply(score_grouping)

    # Train the classifier based on manually labeled dataset
    classifier = train_classifier(labeled_data)
    labeled_data["Predicted PNU"] = labeled_data.apply(create_label, axis = 1, args = (classifier, ))

    return labeled_data

def calculate_score(labeled_data_filename):
    """
    Evaluate the average sentiment score for each company and provide corresponding views.

    This function performs sentiment analysis on labeled data for different companies. 
    It converts the predicted PNU label into corresponding sentiment scores and calculates 
    the average sentiment score for each company.

    Returns:
        None
    """

    labeled_data = sentiment_analyzer(labeled_data_filename)

    # Convert the PNU label to corresponding score
    labeled_data["Predicted PNU score"] = labeled_data["Predicted PNU"].apply(
        label_score
    )
    
    average_predicted_scores = labeled_data.groupby("Company")[
        "Predicted PNU score"
    ].mean()

    # Based on PNU score calculated, obtained our view
    for ticker, score in average_predicted_scores.items():
        if score > 0.05:
            print(f"We hold a bullish view on {ticker}.")
        elif score < -0.05:
            print(f"We hold a bearish view on {ticker}")
        else:
            print(f"The stock price movement of {ticker} is uncertain.")

    print("Sentiment Analysis - Done!")

    # labeled_data_test = match_comparison(labeled_data)
    # overall_sentiment_socre(labeled_data_test)

    return labeled_data
