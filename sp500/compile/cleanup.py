import json
import pandas as pd

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
    "with"
)

def cleanup(labeled_data):
    
    labeled_data["Cleaned Headline"] = None

    for index, row in labeled_data.iterrows():
        
        headline = row["Headline"].split()
        cleaned_headline = []

        for word in headline:
            if word.lower() not in INDEX_IGNORE:
                cleaned_headline.append(word)
        
        labeled_data.at[index, "Cleaned Headline"] = ' '.join(cleaned_headline)
    
    return labeled_data










