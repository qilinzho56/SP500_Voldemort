import numpy as np

def match_comparison(labeled_data):
    """
    This function compares the predicted PNU (Positive, Negative, Neutral) labels with
    manually labeled PNU labelsin a DataFrame obtained from a sentiment analyzer.

    Returns:
        labeled_data (DataFrame)
            DataFrame containing labeled data with additional column indicating
            whether the predicted PNU matches the manually labeled PNU for each entry.
    """

    # Obtain labeled data dataframe with polarity & subjectivity score and its PNU label

    # Determine if predicted PNU matches with manually labeled PNU
    labeled_data["Label & Predicted PNU match or not"] = np.where(
        labeled_data["Predicted PNU"] == labeled_data["Label (P, N, U-neutral)"],
        "match",
        "non-match",
    )

    match_count = labeled_data[
        labeled_data["Label & Predicted PNU match or not"] == "match"
    ].shape[0]
    match_ratio = match_count / labeled_data.shape[0]

    print(f"The match ratio is {match_ratio}.")

    # Write a new file
    labeled_data.to_csv("./sp500/sa/data/Finished_test_sa.csv", index=False)

    print("Comparison - Done!")

    return labeled_data


def label_score(label):
    """
    In our classifier, it is over optimisitc in (neutral, neutral) situation.
    We need to introduce some penalties here to reduce the inflation of label score.

    Inputs:
        label (str)

    Returns:
        score (int)
    """

    if label == "P":
        return 0.5
    elif label == "N":
        return -1
    else:
        return -0.5


def overall_sentiment_socre(labeled_data):
    """
    Calculate the overall sentiment scores by comparing labeled and predicted PNU (Positive,
    Negative, U-neutral) scores for each company in the labeled data.

    Inputs:
        labeled_data (DataFrame): DataFrame containing labeled data with columns "Label
        (P, N, U-neutral)" and "Predicted PNU" along with other necessary columns.

    Returns:
        None
    """

    labeled_data["Labeled PNU score"] = labeled_data["Label (P, N, U-neutral)"].apply(
        label_score
    )
    labeled_data["Predicted PNU score"] = labeled_data["Predicted PNU"].apply(
        label_score
    )

    average_labeled_scores = labeled_data.groupby("Company")["compound"].mean()
    average_predicted_scores = labeled_data.groupby("Company")[
        "Predicted PNU score"
    ].mean()

    match_count = (
        ((average_labeled_scores > 0.05) & (average_predicted_scores > 0.05))
        | ((average_labeled_scores < -0.05) & (average_predicted_scores < -0.05))
        | (
            (average_labeled_scores >= -0.05)
            & (average_labeled_scores <= 0.05)
            & (average_predicted_scores >= -0.05)
            & (average_predicted_scores <= 0.05)
        )
    )

    match_ratio = match_count.sum() / len(match_count)

    if match_ratio >= 0.8:
        print("Overall sentimental scores match!")
    else:
        print("Scores do not match. Fine tune the analyzer.")
