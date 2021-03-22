
import numpy as np
import pandas as pd
import requests

cross_ref = pd.read_csv("https://raw.githubusercontent.com/footballcsv/england/master/2010s/2014-15/eng.1.csv")
teams_dict = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Hull City AFC": "Hull",
    "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Queens Park Rangers FC": "QPR",
    "Southampton FC": "Southampton",
    "Stoke City FC": "Stoke",
    "Sunderland AFC": "Sunderland",
    "Swansea City FC": "Swansea",
    "Tottenham Hotspur FC": "Tottenham",
    "West Bromwich Albion FC": "West Brom",
    "West Ham United FC": "West Ham"
    }


def get_match_data(url):
    """ Return dataframe all matches from dataset.
    Includes teams, goals, final result and bet365 odds.
    """

    df = pd.read_csv(url)
    df["MatchWeek"] = 0
    df = get_gameweek(df, cross_ref)
    return df[["Date", "MatchWeek", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]][:-1]


def get_gameweek(df, cross_ref):
    """ Obtain the matchweek of the games in df given the cross reference data.
    """
    for i, match in cross_ref.iterrows():
        home = teams_dict[match["Team 1"]]
        away = teams_dict[match["Team 2"]]
        row = df.loc[(df["HomeTeam"] == home) & (df["AwayTeam"] == away)]
        df.loc[i, "MatchWeek"] = match["Round"]
    return df

def get_winstreak(df):
    teams = sorted(df.HomeTeam.unique())
    ws = pd.DataFrame(np.zeros(shape=(38, len(teams))), index = range(1,39), columns = teams)
#     ws.index += 1

    for i in df.index:
        if df["MatchWeek"][i] < 38:
            result = df["FTR"][i]
            home = df["HomeTeam"][i]
            away = df["AwayTeam"][i]
            if result == "H":
                ws[home][df["MatchWeek"][i] + 1] = ws[home][df["MatchWeek"][i]] + 1
            elif result == "A":
                ws[away][df["MatchWeek"][i] + 1] = ws[away][df["MatchWeek"][i]] + 1
    return ws