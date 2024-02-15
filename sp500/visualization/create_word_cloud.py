import pandas as pd
from wordcloud import WordCloud, STOPWORDS, get_single_color_func
import matplotlib.pyplot as plt
from pathlib import Path
from datatypes import GroupedColorFunc
import os
from sp500.compile.cleanup import INDEX_IGNORE
def read_headlines():
    """
    This function reads headlines from a cleaned CSV file and stores them in a dictionary with the stock name as the key.
    Each key maps to another dictionary that categorizes headlines into 'positive', 'negative', and 'neutral' based on their polarity.

    Returns:
        dict: A dictionary with company names as keys. Each key maps to another dictionary with keys 'positive', 'negative', and 'neutral',
        each of which is a list of headlines corresponding to the sentiment category.
    """

    # Use pathlib to construct the path to the CSV file relative to the current file's directory
    filename = Path(__file__).parent.parent / "sa/data/Finished_test_sa.csv"

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)
    # Classifier function to categorize polarity into 'positive', 'negative', or 'neutral'
    def classifier_sentiment(label):
        if label == 'P':
            return 'positive'
        elif label == 'N':
            return 'negative'
        else:
            return 'uncertain'

    # Apply the classifier function to each row's Polarity value to create a new 'sentiment' column
    df['sentiment'] = df['Label (P, N, U-neutral)'].apply(classifier_sentiment)
    # Initialize a dictionary to store the categorized headlines for each company
    company_headlines_sentiment = {}

    # Group the DataFrame by company and iterate through each group
    for company, group in df.groupby('Company'):
        # Filter the group for positive, negative, and neutral headlines and convert to lists
        positive_headlines = group[group['sentiment'] == 'positive']['Headline'].tolist()
        negative_headlines = group[group['sentiment'] == 'negative']['Headline'].tolist()
        uncertain_headlines = group[group['sentiment'] == 'uncertain']['Headline'].tolist()

        # Store the lists in the dictionary under the company's key
        company_headlines_sentiment[company] = {
            'positive': positive_headlines,
            'negative': negative_headlines,
            'uncertain': uncertain_headlines
        }
    # Return the dictionary containing the categorized headlines for each company
    print(company_headlines_sentiment)
    return company_headlines_sentiment



def extract_keywords_from_headlines(headlines):
    """
    Extract keywords from headlines, excluding stopwords.

    Parameters:
        headlines (list of str): A list of headlines from which to extract keywords.
        stopwords (set of str): A set of stopwords to ignore in the extraction process.

    Returns:
        list of str: A list of keywords extracted from the headlines, excluding stopwords.
    """
    # Split headlines into words, exclude stopwords, and flatten the list
    keywords = [word for headline in headlines for word in headline.split() if word.lower() not in INDEX_IGNORE ]
    return keywords

def create_sentiment_to_keywords_dict(company_headlines_sentiment):
    """
    Creates a dictionary that maps sentiment categories (positive, negative)
    to lists of keywords extracted from headlines.

    Parameters:
        company_headlines_sentiment (dict): A dictionary with company names as keys
        and dictionaries as values, where each inner dictionary has 'positive',
        'negative', and 'neutral' keys mapping to lists of headlines.

    Returns:
        dict: A dictionary with sentiment categories as keys ('positive', 'negative')
        and lists of keywords associated with that sentiment as values.
    """
    # Initialize the dictionary to hold sentiment categories and their keywords
    sentiment_to_keywords = {'positive': [], 'negative': [], 'uncertain':[]}

    # Iterate through each company and its associated sentiment categories and headlines
    for company, sentiments in company_headlines_sentiment.items():
        for sentiment, headlines in sentiments.items():
            if sentiment in sentiment_to_keywords:
                # Extract keywords from headlines
                keywords = extract_keywords_from_headlines(headlines)
                # Append keywords to the appropriate sentiment category
                sentiment_to_keywords[sentiment].extend(keywords)
    return sentiment_to_keywords

def map_sentiments_to_colors(sentiment_to_keywords, sentiment_color_mapping):
    """
    Maps sentiments to their corresponding colors and aggregates keywords.

    Parameters:
        sentiment_to_keywords (dict): A dictionary with sentiments as keys and lists of keywords as values.
        sentiment_color_mapping (dict): A dictionary mapping sentiments to color codes.

    Returns:
        dict: A dictionary with color codes as keys and lists of keywords associated with those colors as values.
    """
    color_to_keywords = {}

    for sentiment, keywords in sentiment_to_keywords.items():
        # Get the color corresponding to the current sentiment
        color = sentiment_color_mapping.get(sentiment, "#808080")  # Default to gray if sentiment is not found

        # Initialize the keyword list for this color if it doesn't exist
        if color not in color_to_keywords:
            color_to_keywords[color] = []

        # Extend the keyword list for this color with the current keywords
        color_to_keywords[color].extend(keywords)

    return color_to_keywords

def create_wordcloud():
    # Read headlines and perform sentiment analysis, returning a dictionary of {company: {sentiment: [headlines]}}
    headlines_dict = read_headlines()

    # Define the mapping from sentiments to color codes
    sentiment_color_mapping = {
        "positive": "#FF0000",  # Red for positive sentiment
        "negative": "#008000",  # Green for negative sentiment
        "neutral": "#808080",   # Gray for neutral sentiment
    }

    # Define the path to the visualization directory
    visualization_dir = os.path.join(os.path.dirname(__file__), '..', 'visualization')

    # Ensure the visualization directory exists
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)

    for company, sentiments in headlines_dict.items():
        # Create a sentiment to keywords mapping for the current company
        sentiment_to_keywords = create_sentiment_to_keywords_dict({company: sentiments})
        # Map sentiments to colors
        color_to_keywords = map_sentiments_to_colors(sentiment_to_keywords, sentiment_color_mapping)

        # Initialize GroupedColorFunc with the current company's color to keywords mapping
        color_func = GroupedColorFunc(color_to_keywords, 'grey')

        # Combine all headlines into a single long string for word cloud generation
        all_headlines = ' '.join([' '.join(keywords) for keywords in color_to_keywords.values()])

        # Avoid including the company name in the word cloud
        stopwords = set(STOPWORDS)
        stopwords.update([company])

        # Create a WordCloud instance
        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords, random_state=21).generate(all_headlines)

        # Apply the color function to assign colors based on sentiment
        wordcloud.recolor(color_func=color_func)

        # Plot the word cloud image and save it
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')

        # Construct the file path for the company's word cloud image
        filepath = os.path.join(visualization_dir, f"{company}_wordcloud.png")
        plt.savefig(filepath)
        plt.close()

if __name__ == "__main__":
    create_wordcloud()
