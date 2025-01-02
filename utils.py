import pandas as pd
import json
from collections import Counter


def json_to_list(json_file: str) -> list:
    """
    Reads a json file and return a dict
    :param json_file: full path to json file
    :return: list of dictionaries
    """
    with open(json_file,'r', encoding='utf-8') as json_file:
        return json.load(json_file)


def json_to_df(json_file: str) -> pd.DataFrame:
    """
    Reads a json file and return a dataframe
    :param json_file: full path to json file
    :return: a dataframe
    """
    return pd.read_json(json_file)


def get_most_common_word_simple(post: str) -> str:
    """
    Returns the most common word in a post. note: this function is very simplistic
    :param post: the post text
    :return: the most common word. if there are several words with the same frequency, it will return the first one
    """
    if pd.isnull(post) or not post.strip():
        return None
    words = post.split()
    most_common = Counter(words).most_common(1)
    return most_common[0][0] if most_common else None