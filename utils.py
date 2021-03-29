import numpy as np
import pandas as pd


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
    ws = pd.DataFrame(np.zeros(shape=(38, len(teams)), dtype=int), index = range(1,39), columns = teams)

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


def get_result(score):
    if score[0]>score[1]:
        return "H"
    elif score[0]<score[1]:
        return "A"
    return "D"


def simulate_match(params, home_index, away_index):
    """
    Simulate a match between home and away teams.
    Returns (home_goals, away_goals).
    """

    home_goals = np.random.poisson( lam = (params[0,home_index] * params[3,away_index]) )
    away_goals = np.random.poisson( lam = (params[1,home_index] * params[2,away_index]) )
    return int(home_goals), int(away_goals)


def simulate_league(matches, params, teams_index):
    simulated = matches.copy(deep=True)
    for i, row in simulated.iterrows():
        home_index = teams_index[row["HomeTeam"]]
        away_index = teams_index[row["AwayTeam"]]
        home_goals, away_goals = simulate_match(params, home_index, away_index)
        result = get_result((home_goals, away_goals))
        simulated.loc[i, "FTHG"] = home_goals
        simulated.loc[i, "FTAG"] = away_goals
        simulated.loc[i, "FTR"] = result
    return simulated


def momentum_sim_match(team_params, streak_param, home_streak, away_streak, home_index, away_index):
    home_goals = np.random.poisson( lam = (team_params[0,home_index] * team_params[3,away_index]) * streak_param ** home_streak)
    away_goals = np.random.poisson( lam = (team_params[1,home_index] * team_params[2,away_index]) * streak_param ** away_streak)
    return int(home_goals), int(away_goals)


def momentum_sim_league(matches, team_params, streak_param, teams_index, streaks):
    simulated = matches.copy(deep=True)
    for i, row in simulated.iterrows():
        home_index = teams_index[row["HomeTeam"]]
        away_index = teams_index[row["AwayTeam"]]
        home_streak = streaks[row["MatchWeek"] - 1, home_index]
        away_streak = streaks[row["MatchWeek"] - 1, away_index]
        home_goals, away_goals = momentum_sim_match(team_params, streak_param, home_streak, away_streak, home_index, away_index)
        result = get_result((home_goals, away_goals))
        simulated.loc[i, "FTHG"] = home_goals
        simulated.loc[i, "FTAG"] = away_goals
        simulated.loc[i, "FTR"] = result
        if row["MatchWeek"] < 38:
            if result == "H":
                streaks[row["MatchWeek"], home_index] = home_streak + 1
            elif result == "A":
                streaks[row["MatchWeek"], away_index] = away_streak + 1
    return simulated


def standings(df):
    """
    Given a dataframe of results, return final standings as a dataframe
    """
    columns = ["Team", "Points", "W", "D", "L", "GD", "GF", "GA"]
    table = pd.DataFrame(data=np.zeros((20,8), dtype=int), columns=columns)
    table["Team"] = list(teams_dict.values())
    for i, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        score = (int(row["FTHG"]),int(row["FTAG"]))
        result = get_result(score)
        # update results
        if result == "H":
            table.loc[table["Team"] == home, "Points"] += 3
            table.loc[table["Team"] == home, "W"] += 1
            table.loc[table["Team"] == away, "L"] += 1
        elif result == "A":
            table.loc[table["Team"] == away, "Points"] += 3
            table.loc[table["Team"] == away, "W"] += 1
            table.loc[table["Team"] == home, "L"] += 1
        else:
            table.loc[table["Team"] == home, "Points"] += 1
            table.loc[table["Team"] == away, "Points"] += 1
            table.loc[table["Team"] == home, "D"] += 1
            table.loc[table["Team"] == away, "D"] += 1
        
        # update goals
        table.loc[table["Team"] == home, "GF"] += score[0]
        table.loc[table["Team"] == home, "GA"] += score[1]
        table.loc[table["Team"] == home, "GD"] += (score[0] - score[1])

        table.loc[table["Team"] == away, "GF"] += score[1]
        table.loc[table["Team"] == away, "GA"] += score[0]
        table.loc[table["Team"] == away, "GD"] += (score[1] - score[0])
    
    table = table.sort_values(by=["Points", "W", "GD"], ascending=False)
    table = table.set_index(pd.Index([i for i in range(1,21)]))
    return table
        
    
    