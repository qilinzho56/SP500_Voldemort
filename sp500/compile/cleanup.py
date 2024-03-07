INDEX_IGNORE = (
    "a",
    "an",
    "and",
    "&",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "i",
    "she",
)


def cleanup(labeled_data):
    """
    To filter out words that matches INDEX_IGNORE

    Inputs:
        labeled_data: DataFrame

    Returns:
        labeled_data: DataFrame
    """

    # create new column
    labeled_data["Cleaned Headline"] = None

    # iterate through each row of dataframe
    for index, row in labeled_data.iterrows():

        headline = row["Headline"].split()
        cleaned_headline = []

        for word in headline:
            if word.lower() not in INDEX_IGNORE:
                cleaned_headline.append(word)

        labeled_data.at[index, "Cleaned Headline"] = " ".join(cleaned_headline)
        labeled_data["Cleaned Headline"] = labeled_data["Cleaned Headline"].str.lower()

    return labeled_data
