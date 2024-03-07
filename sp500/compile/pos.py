import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

def add_new_column(dict, df, row_index):
    """
    According to part of speech of words, we add them
    to the dataframe

    Inputs:
        dict: dict
            A dictionary containing part of speech and corresponding words
        df: dataframe
            The dataframe we import
        row_index: int
            row index
    """

    for pos in dict:
        if pos not in df.columns:
            df[pos] = None
        df.at[row_index, pos] = ', '.join(dict[pos])


def POS_tagging(text, df, row_index):
    """
    Using spaCy to do Part of Speech (POS) Tagging
    
    Parameters
    ----------
    text: str
        News headline we scrap
    df: dataframe
        The dataframe we import
    row_index: int
    """

    part_of_speech_dict = {}
    docs = nlp(text)
    
    for word in docs:
        if word.pos_ not in part_of_speech_dict:
            part_of_speech_dict[word.pos_] = []
            part_of_speech_dict[word.pos_].append(word.text)
        else:
            part_of_speech_dict[word.pos_].append(word.text)
    
    add_new_column(part_of_speech_dict, df, row_index)


def POS_in_DF(df):
    """
    Conduct Part of Speech code for DataFrame
    """  
    for index, row in df.iterrows():
        POS_tagging(row["Headline"], df, index)
    
    return df
