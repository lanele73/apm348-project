import numpy as np
import pandas as pd
import requests

def get_match_data(url):
    """ Return dataframe all matches from dataset.
    Includes match week, teams, goals, final result and bet365 odds.
    """
    df = pd.read_csv(url)
    return df[["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]]

def odds_probabilities(df):
    return df
    