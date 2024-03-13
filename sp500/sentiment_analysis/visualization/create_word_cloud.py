from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sp500.sentiment_analysis.visualization.datatypes import GroupedColorFunc
import os
import re
import numpy as np
from PIL import Image
import yfinance as yf

ADDITIONAL_STOPWORDS = [
    "stock",
    "stocks",
    "market",
    "markets",
    "Stock",
    "Stocks",
    "Alphabet",
    "Microsoft",
]


def read_headlines(df):
    """
    This function processes a DataFrame of headlines and stores them in a dictionary with the stock name as the key.
    Each key maps to another dictionary that categorizes headlines into 'positive', 'negative', and 'uncertain'
    based on scores and labels.

    Parameters:
        df (pandas.DataFrame): A DataFrame containing the headlines and associated sentiment scores and labels.

    Returns:
        dict: A dictionary with company names as keys. Each key maps to another dictionary with keys 'positive', 'negative', and 'uncertain',
        each of which is a list of headlines corresponding to the sentiment category.
    """

    # Classifier function to categorize sentiment based on pos, neg scores, and Label
    def classifier_sentiment(row):
        if row["pos"] > row["neg"] and row["Segmentation Compound"] == "Positive High":
            return "positive"
        elif (
            row["pos"] < row["neg"] and row["Segmentation Compound"] == "Negative High"
        ):
            return "negative"
        else:
            return "uncertain"

    # Apply the classifier function to each row to create a new 'sentiment' column
    df["sentiment"] = df.apply(classifier_sentiment, axis=1)

    # Initialize a dictionary to store the categorized headlines for each company
    company_headlines_sentiment = {}

    # Group the DataFrame by company and iterate through each group
    for company, group in df.groupby("Company"):
        positive_headlines = group[group["sentiment"] == "positive"][
            "Headline"
        ].tolist()
        negative_headlines = group[group["sentiment"] == "negative"][
            "Headline"
        ].tolist()
        uncertain_headlines = group[group["sentiment"] == "uncertain"][
            "Headline"
        ].tolist()

        # Store the lists in the dictionary under the company's key
        company_headlines_sentiment[company] = {
            "positive": positive_headlines,
            "negative": negative_headlines,
            "uncertain": uncertain_headlines,
        }

    return company_headlines_sentiment


def add_stopwords_with_regex(texts, stopwords):
    """
    Adds a list of texts to the stopwords set, using regex to catch variations.

    Parameters:
        texts: A list of text that should include in stopwords
        stopwords: A set of words that should be added to the stopword list.

    Returns:
        A set containing the updated stopwords.
    """

    # Using regular expression to handle with company name like 'Amazon.com' or 'Apple Inc'
    for text in texts:
        # Split the text by spaces and punctuation, except when part of a domain name
        words = re.split(r"\s+(?![^()]*\))|(?<!\.\w)\b", text)
        for word in words:
            if word and len(word) > 2 and not word.lower().endswith(".com"):
                word = word.lower()
                pattern = r"\b" + re.escape(word) + r"\b"
                stopwords.add(word)
                additional_stopwords = set(
                    filter(lambda w: re.search(pattern, w, re.I), stopwords)
                )
                stopwords.update(additional_stopwords)

    return stopwords


def extract_keywords_from_headlines(headlines, stopwords):
    """
    Extract keywords from headlines, excluding stopwords.

    Parameters:
        headlines (list of str): A list of headlines from which to extract keywords.
        stopwords (set of str): A set of stopwords to ignore in the extraction process.

    Returns:
        list of str: A list of keywords extracted from the headlines, excluding stopwords.
    """
    # Split headlines into words, exclude stopwords, and flatten the list
    keywords = [
        word
        for headline in headlines
        for word in headline.split()
        if word.lower() not in stopwords
    ]
    return keywords


def map_stock_names_to_company_names(df, stock_column):
    """
    Using Yfinance built a projection from stock name to company name.
    It is helpful to exclude company name in wordcloud

    Parameters:
        df (pandas.DataFrame): A DataFrame containing stock name
        stock_column(str): A str represent the column of stock name in DataFrame

    Returns:
        A dict: A dict contains projection from stock name to company name
    """

    # Ensure the stock_column exists in the DataFrame
    if stock_column not in df.columns:
        raise ValueError(f"Column {stock_column} does not exist in DataFrame")

    # Initialize a dictionary to store company names
    company_names = {}

    # Fetch company names using yfinance
    for symbol in df[stock_column]:
        ticker = yf.Ticker(symbol)
        company_name = ticker.info.get("longName", None)
        company_names[symbol] = company_name

    return company_names


def create_sentiment_to_keywords_dict(company_headlines_sentiment, stopwords):
    """
    Creates a dictionary that maps sentiment categories (positive, negative)
    to lists of keywords extracted from headlines.

    Parameters:
        company_headlines_sentiment (dict): A dictionary with company names as keys
        and dictionaries as values, where each inner dictionary has 'positive',
        'negative', and 'neutral' keys mapping to lists of headlines.
        stopwords (set of str): A set of stopwords to ignore in the extraction process.

    Returns:
        dict: A dictionary with sentiment categories as keys ('positive', 'negative')
        and lists of keywords associated with that sentiment as values.
    """

    # Initialize the dictionary to hold sentiment categories and their keywords
    sentiment_to_keywords = {"positive": [], "negative": [], "uncertain": []}

    # Iterate through each company and its associated sentiment categories and headlines
    # Extract keywords from headlines and Append keywords to the appropriate sentiment category
    for company, sentiments in company_headlines_sentiment.items():
        for sentiment, headlines in sentiments.items():
            if sentiment in sentiment_to_keywords:
                keywords = extract_keywords_from_headlines(headlines, stopwords)
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

    # Get the color corresponding to the current sentiment
    # Default to gray if sentiment is not found
    for sentiment, keywords in sentiment_to_keywords.items():
        color = sentiment_color_mapping.get(sentiment, "#808080")

        # Initialize the keyword list for this color if it doesn't exist
        if color not in color_to_keywords:
            color_to_keywords[color] = []

        # Extend the keyword list for this color with the current keywords
        color_to_keywords[color].extend(keywords)

    return color_to_keywords


def create_default_mask():
    """
    Creates a default mask for WordCloud in case a company logo is not available.

    Returns:
        numpy.ndarray: A default mask array.
    """
    # Create a simple circle as a default mask
    mask_size = (300, 300)
    mask = np.zeros(mask_size, dtype=np.uint8)
    center_x, center_y = np.array(mask_size) // 2
    radius = min(mask_size) // 2 - 10

    for x in range(mask_size[0]):
        for y in range(mask_size[1]):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2:
                mask[x, y] = 255

    return mask


def create_wordcloud(
    df,
    company_logo_paths=None,
    stock_to_company=None,
    visualization_dir="visualization",
):
    """
    Generates word clouds for each company, optionally shaped by the company's logo, and saves them to a specified directory.

    Parameters:
        df (pandas.DataFrame): A DataFrame containing the headlines and associated sentiment scores and labels.
        company_logo_paths (dict, optional): A dictionary mapping company names to file paths of their logos.
        If provided, each word cloud will be shaped like the respective company's logo.
        The keys should match the company names used in the DataFrame or provided by the stock_to_company mapping.

        stock_to_company (dict, optional): A dictionary mapping stock symbols to company names.
        This is useful for converting stock symbols in the DataFrame to company names, particularly if the textual data
        references the stocks by their symbols. This mapping ensures that the text is correctly attributed to companies.

        visualization_dir (str): The path to the directory where the generated word cloud images will be saved.
        If the directory does not exist, it will be created. Defaults to a directory named "visualization" within the
        current working directory.
    """
    # Read headlines and perform sentiment analysis, returning a dictionary of {company: {sentiment: [headlines]}}
    headlines_dict = read_headlines(df)

    # Define the mapping from sentiments to color codes
    sentiment_color_mapping = {
        "positive": "#008000",
        "negative": "#FF0000",
        "neutral": "#808080",
    }

    # Initialize the visualization directory exists
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)

    # Initialize stopwords
    custom_stopwords = set(STOPWORDS)

    # Add company names and stock symbols to stopwords if provided
    if stock_to_company:
        texts_to_add = list(stock_to_company.values()) + list(stock_to_company.keys())
        custom_stopwords = add_stopwords_with_regex(texts_to_add, custom_stopwords)

    # Add additional custom stopwords
    custom_stopwords = add_stopwords_with_regex(ADDITIONAL_STOPWORDS, custom_stopwords)

    # Iterate through each entry in headlines_dict
    # And check if a dictionary of company logo paths exists
    for company, sentiments in headlines_dict.items():
        if company_logo_paths and company in company_logo_paths:
            mask_path = company_logo_paths[company]
            mask = np.array(Image.open(mask_path))
        else:
            mask = create_default_mask()

        inverted_mask = np.invert(mask)

        # Create a sentiment to keywords mapping for the current company
        sentiment_to_keywords = create_sentiment_to_keywords_dict(
            {company: sentiments}, custom_stopwords
        )

        # Extract keywords excluding stopwords for each sentiment
        for sentiment, headlines in sentiment_to_keywords.items():
            sentiment_to_keywords[sentiment] = extract_keywords_from_headlines(
                headlines, custom_stopwords
            )

        # Map sentiments to colors
        color_to_keywords = map_sentiments_to_colors(
            sentiment_to_keywords, sentiment_color_mapping
        )

        # Initialize GroupedColorFunc with the current company's color to keywords mapping
        color_func = GroupedColorFunc(color_to_keywords, "grey")

        # Combine all headlines into a single long string for word cloud generation
        all_headlines = " ".join(
            [" ".join(keywords) for keywords in sentiment_to_keywords.values()]
        )

        # Create a WordCloud instance
        wordcloud = WordCloud(
            background_color="white",
            stopwords=custom_stopwords,
            mask=inverted_mask,
            max_words=200,
            max_font_size=120,
            scale=2,
            contour_width=3,
            contour_color="black",
            random_state=21,
        ).generate(all_headlines)

        # Apply the color function to assign colors based on sentiment
        wordcloud.recolor(color_func=color_func)

        # Plot the word cloud image and save it
        plt.figure(figsize=(10, 10)) 
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")

        # Construct the file path for the company's word cloud image
        filepath = os.path.join(visualization_dir, f"{company}_wordcloud.png")
        plt.savefig(filepath)
        plt.close()


# Simplified version of wordcloud without controlling company name frequency, for GUI compatibility
# With color revision
def create_wordcloud_for_company(
    df,
    company_name,
    company_logo_paths=None,
    stock_to_company=None,
    visualization_dir="visualization",
):
    """
    Generates word clouds for each company, optionally shaped by the company's logo, and saves them to a specified directory.

    Parameters:
        df (pandas.DataFrame): A DataFrame containing the headlines and associated sentiment scores and labels.
        company_name (str)
        company_logo_paths (dict, optional): A dictionary mapping company names to file paths of their logos.
        If provided, each word cloud will be shaped like the respective company's logo.
        The keys should match the company names used in the DataFrame or provided by the stock_to_company mapping.

        stock_to_company (dict, optional): A dictionary mapping stock symbols to company names.
        This is useful for converting stock symbols in the DataFrame to company names, particularly if the textual data
        references the stocks by their symbols. This mapping ensures that the text is correctly attributed to companies.

        visualization_dir (str): The path to the directory where the generated word cloud images will be saved.
        If the directory does not exist, it will be created. Defaults to a directory named "visualization" within the
        current working directory.
    """
    # Read headlines and perform sentiment analysis, returning a dictionary of {company: {sentiment: [headlines]}}
    headlines_dict = read_headlines(df[df["Company"] == company_name])

    # Define the mapping from sentiments to color codes
    sentiment_color_mapping = {
        "positive": "#008000",
        "negative": "#FF0000",
        "neutral": "#808080",
    }

    # Initialize the visualization directory exists
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)

    # Initialize stopwords
    custom_stopwords = set(STOPWORDS)
    
    if stock_to_company:
        texts_to_add = list(stock_to_company.values()) + list(stock_to_company.keys())
        custom_stopwords = add_stopwords_with_regex(texts_to_add, custom_stopwords)

    # Add additional custom stopwords
    custom_stopwords = add_stopwords_with_regex(ADDITIONAL_STOPWORDS, custom_stopwords)

    # Iterate through each entry in headlines_dict
    # And check if a dictionary of company logo paths exists
    for company, sentiments in headlines_dict.items():
        if company_logo_paths and company in company_logo_paths:
            mask_path = company_logo_paths[company]
            mask = np.array(Image.open(mask_path))
        else:
            mask = create_default_mask()

        inverted_mask = np.invert(mask)

        # Create a sentiment to keywords mapping for the current company
        sentiment_to_keywords = create_sentiment_to_keywords_dict(
            {company: sentiments}, custom_stopwords
        )

        # Extract keywords excluding stopwords for each sentiment
        for sentiment, headlines in sentiment_to_keywords.items():
            sentiment_to_keywords[sentiment] = extract_keywords_from_headlines(
                headlines, custom_stopwords
            )

        # Map sentiments to colors
        color_to_keywords = map_sentiments_to_colors(
            sentiment_to_keywords, sentiment_color_mapping
        )

        # Initialize GroupedColorFunc with the current company's color to keywords mapping
        color_func = GroupedColorFunc(color_to_keywords, "grey")

        # Combine all headlines into a single long string for word cloud generation
        all_headlines = " ".join(
            [" ".join(keywords) for keywords in sentiment_to_keywords.values()]
        )

        # Create a WordCloud instance
        wordcloud = WordCloud(
            background_color="white",
            stopwords=custom_stopwords,
            mask=inverted_mask,
            max_words=200,
            max_font_size=120,
            scale=2,
            contour_width=3,
            contour_color="black",
            random_state=21,
        ).generate(all_headlines)

        # Apply the color function to assign colors based on sentiment
        wordcloud.recolor(color_func=color_func)

        filepath = os.path.join(visualization_dir, f"{company_name}_wordcloud_simplified.png")
        wordcloud.to_file(filepath)

        return filepath