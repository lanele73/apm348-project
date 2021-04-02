import numpy as np
import pandas as pd
from scipy.stats import poisson

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

teams_ind = {
    "Arsenal": 0,
    "Aston Villa": 1,
    "Burnley": 2,
    "Chelsea": 3,
    "Crystal Palace": 4,
    "Everton": 5,
    "Hull": 6,
    "Leicester": 7,
    "Liverpool": 8,
    "Man City": 9,
    "Man United": 10,
    "Newcastle": 11,
    "QPR": 12,
    "Southampton": 13,
    "Stoke": 14,
    "Sunderland": 15,
    "Swansea": 16,
    "Tottenham": 17,
    "West Brom": 18,
    "West Ham": 19
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
        df.loc[row.index, "MatchWeek"] = match["Round"]
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


def get_standings(df):
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


def simulate_with_standings(matches, params, teams_index):
    columns = ["Team", "Points", "W", "GD"]
    data = np.zeros((20,3), dtype=int)
    for i, row in matches.iterrows():
        home_index = teams_index[row["HomeTeam"]]
        away_index = teams_index[row["AwayTeam"]]
        home_goals, away_goals = simulate_match(params, home_index, away_index)
        result = get_result((home_goals, away_goals))
        if result == "H":
            data[home_index, 0] += 3
            data[home_index, 1] += 1
        elif result == "A":
            data[away_index, 0] += 3
            data[away_index, 1] += 1
        else:
            data[home_index, 0] += 1
            data[away_index, 0] += 1
        data[home_index, 2] += (home_goals - away_goals)
        data[away_index, 2] += (away_goals - home_goals)
    
    table = pd.DataFrame(data=np.zeros((20,4), dtype=int), columns=columns)
    table["Team"] = list(teams_ind.keys())
    table["Points"] = data[:,0]
    table["W"] = data[:,1]
    table["GD"] = data[:,2]
    
    table = table.sort_values(by=["Points", "W", "GD"], ascending=False)
    table = table.set_index(pd.Index(range(1,21)))
    return table


def sim_with_standings_momentum(matches, team_params, streak_param, teams_index, streaks=np.zeros((38,20), dtype=int)):
    """
    Matches is sorted by matchweek.
    """
    columns = ["Team", "Points", "W", "GD"]
    data = np.zeros((20,3), dtype=int)
    for i, row in matches.iterrows():
        home_index = teams_index[row["HomeTeam"]]
        away_index = teams_index[row["AwayTeam"]]
        home_streak = streaks[row["MatchWeek"] - 1, home_index]
        away_streak = streaks[row["MatchWeek"] - 1, away_index]

        home_goals, away_goals = momentum_sim_match(team_params, streak_param, home_streak, away_streak, home_index, away_index)
        result = get_result((home_goals, away_goals))

        if result == "H":
            data[home_index, 0] += 3
            data[home_index, 1] += 1
        elif result == "A":
            data[away_index, 0] += 3
            data[away_index, 1] += 1
        else:
            data[home_index, 0] += 1
            data[away_index, 0] += 1
        data[home_index, 2] += (home_goals - away_goals)
        data[away_index, 2] += (away_goals - home_goals)

        if row["MatchWeek"] < 38:
            if result == "H":
                streaks[row["MatchWeek"], home_index] = home_streak + 1
                streaks[row["MatchWeek"], away_index] = 0
            elif result == "A":
                streaks[row["MatchWeek"], home_index] = 0
                streaks[row["MatchWeek"], away_index] = away_streak + 1
            else:
                streaks[row["MatchWeek"], home_index] = 0
                streaks[row["MatchWeek"], away_index] = 0

    table = pd.DataFrame(data=np.zeros((20,4), dtype=int), columns=columns)
    table["Team"] = list(teams_ind.keys())
    table["Points"] = data[:,0]
    table["W"] = data[:,1]
    table["GD"] = data[:,2]
    
    table = table.sort_values(by=["Points", "W", "GD"], ascending=False)
    table = table.set_index(pd.Index(range(1,21)))
    return table


def get_winner(table):
    return table.head(1)["Team"]


def get_top4(table):
    return table.head(4)["Team"]


def get_bottom4(table):
    return table.tail(4)["Team"]


def get_odds(params, home_index, away_index, complex=True):
    if complex:
        home_goal_param = params[0, home_index] * params[3, away_index]
        away_goal_param = params[1, home_index] * params[2, away_index]
    else:
        home_goal_param = params[0,home_index] * params[1, away_index]
        away_goal_param = params[1,home_index] * params[0, away_index]
    size=16
    scores = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            scores[i,j] = poisson.pmf(i, home_goal_param) * poisson.pmf(j, away_goal_param)
    odds = np.array([1/np.sum(np.tril(scores, -1)), 1/np.sum(np.diagonal(scores)), 1/np.sum(np.triu(scores, 1))])
    return odds


def kelly_criterion(house_odds, model_odds):
    """
    Kelly criterion betting strategy.
    """
    return (1/model_odds * (house_odds) - 1) / (house_odds - 1)


def place_bet(current, house_odds, model_odds, result):
    results_dict = results_dict = {0: "H", 1: "D", 2: "A"}
    kelly = kelly_criterion(house_odds, model_odds)
    best_bet = np.argmax(kelly)

    if kelly[best_bet] > 0:
        if kelly[best_bet] > 0.025:
            kelly[best_bet] = 0.025
        bet = round(current * kelly[best_bet],2)
        current -= bet                          # Place bet
        if results_dict[best_bet] == result:    # Win bet
            current += bet * house_odds[best_bet]
            current = round(current, 2)
    return current
