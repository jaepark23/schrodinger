from nba_api.stats.endpoints import TeamPlayerDashboard
from nba_api.stats.static.teams import find_team_name_by_id
from nba_api.stats.endpoints import LeagueDashTeamStats
from nba_api.stats.endpoints import BoxScoreTraditionalV3
from nba_api.live.nba.endpoints import ScoreBoard

import pandas as pd
import numpy as np
import re
from datetime import date
import joblib
import pickle as pk
import keras
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
import requests
from bs4 import BeautifulSoup
import bs4

featuresv1 = ['W_PCT', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'AST_PCT', 'AST_TO', 'AST_RATIO', 'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE', 'PCT_FGA_2PT', 'PCT_FGA_3PT', 'PCT_PTS_2PT', 'PCT_PTS_2PT_MR', 'PCT_PTS_3PT', 'PCT_PTS_FB', 'PCT_PTS_FT', 'PCT_PTS_OFF_TOV', 'PCT_PTS_PAINT', 'PCT_AST_2PM', 'PCT_UAST_2PM', 'PCT_AST_3PM', 'PCT_UAST_3PM', 'PCT_AST_FGM', 'PCT_UAST_FGM', 'OPP_FGM', 'OPP_FGA', 'OPP_FG_PCT', 'OPP_FG3M', 'OPP_FG3A', 'OPP_FG3_PCT', 'OPP_FTM', 'OPP_FTA', 'OPP_FT_PCT', 'OPP_OREB', 'OPP_DREB', 'OPP_REB', 'OPP_AST', 'OPP_TOV', 'OPP_STL', 'OPP_BLK', 'OPP_BLKA', 'OPP_PF', 'OPP_PFD', 'OPP_PTS', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE', 'OPP_PTS_FB', 'OPP_PTS_PAINT', 'LAST_10_W_PCT', 'TEAM_POWER', 'W_PCT1', 'FGM1', 'FGA1', 'FG_PCT1', 'FG3M1', 'FG3A1', 'FG3_PCT1', 'FTM1', 'FTA1', 'FT_PCT1', 'OREB1', 'DREB1', 'REB1', 'AST1', 'TOV1', 'STL1', 'BLK1', 'BLKA1', 'PF1', 'PFD1', 'PTS1', 'PLUS_MINUS1', 'OFF_RATING1', 'DEF_RATING1', 'NET_RATING1', 'AST_PCT1', 'AST_TO1', 'AST_RATIO1', 'OREB_PCT1', 'DREB_PCT1', 'REB_PCT1', 'TM_TOV_PCT1', 'EFG_PCT1', 'TS_PCT1', 'E_PACE1', 'PACE1', 'PACE_PER401', 'POSS1', 'PIE1', 'PCT_FGA_2PT1', 'PCT_FGA_3PT1', 'PCT_PTS_2PT1', 'PCT_PTS_2PT_MR1', 'PCT_PTS_3PT1', 'PCT_PTS_FB1', 'PCT_PTS_FT1', 'PCT_PTS_OFF_TOV1', 'PCT_PTS_PAINT1', 'PCT_AST_2PM1', 'PCT_UAST_2PM1', 'PCT_AST_3PM1', 'PCT_UAST_3PM1', 'PCT_AST_FGM1', 'PCT_UAST_FGM1', 'OPP_FGM1', 'OPP_FGA1', 'OPP_FG_PCT1', 'OPP_FG3M1', 'OPP_FG3A1', 'OPP_FG3_PCT1', 'OPP_FTM1', 'OPP_FTA1', 'OPP_FT_PCT1', 'OPP_OREB1', 'OPP_DREB1', 'OPP_REB1', 'OPP_AST1', 'OPP_TOV1', 'OPP_STL1', 'OPP_BLK1', 'OPP_BLKA1', 'OPP_PF1', 'OPP_PFD1', 'OPP_PTS1', 'OPP_PTS_OFF_TOV1', 'OPP_PTS_2ND_CHANCE1', 'OPP_PTS_FB1', 'OPP_PTS_PAINT1', 'LAST_10_W_PCT1', 'TEAM_POWER1']

featuresv2 = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'AST_PCT', 'AST_TO', 'AST_RATIO', 'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE', 'PCT_FGA_2PT', 'PCT_FGA_3PT', 'PCT_PTS_2PT', 'PCT_PTS_2PT_MR', 'PCT_PTS_3PT', 'PCT_PTS_FB', 'PCT_PTS_FT', 'PCT_PTS_OFF_TOV', 'PCT_PTS_PAINT', 'PCT_AST_2PM', 'PCT_UAST_2PM', 'PCT_AST_3PM', 'PCT_UAST_3PM', 'PCT_AST_FGM', 'PCT_UAST_FGM', 'OPP_FGM', 'OPP_FGA', 'OPP_FG_PCT', 'OPP_FG3M', 'OPP_FG3A', 'OPP_FG3_PCT', 'OPP_FTM', 'OPP_FTA', 'OPP_FT_PCT', 'OPP_OREB', 'OPP_DREB', 'OPP_REB', 'OPP_AST', 'OPP_TOV', 'OPP_STL', 'OPP_BLK', 'OPP_BLKA', 'OPP_PF', 'OPP_PFD', 'OPP_PTS', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE', 'OPP_PTS_FB', 'OPP_PTS_PAINT', 'LAST_10_W_PCT', 'TEAM_POWER', 'W_PCT1', 'FGM1', 'FGA1', 'FG_PCT1', 'FG3M1', 'FG3A1', 'FG3_PCT1', 'FTM1', 'FTA1', 'FT_PCT1', 'OREB1', 'DREB1', 'REB1', 'AST1', 'TOV1', 'STL1', 'BLK1', 'BLKA1', 'PF1', 'PFD1', 'PTS1', 'PLUS_MINUS1', 'OFF_RATING1', 'DEF_RATING1', 'NET_RATING1', 'AST_PCT1', 'AST_TO1', 'AST_RATIO1', 'OREB_PCT1', 'DREB_PCT1', 'REB_PCT1', 'TM_TOV_PCT1', 'EFG_PCT1', 'TS_PCT1', 'E_PACE1', 'PACE1', 'PACE_PER401', 'POSS1', 'PIE1', 'PCT_FGA_2PT1', 'PCT_FGA_3PT1', 'PCT_PTS_2PT1', 'PCT_PTS_2PT_MR1', 'PCT_PTS_3PT1', 'PCT_PTS_FB1', 'PCT_PTS_FT1', 'PCT_PTS_OFF_TOV1', 'PCT_PTS_PAINT1', 'PCT_AST_2PM1', 'PCT_UAST_2PM1', 'PCT_AST_3PM1', 'PCT_UAST_3PM1', 'PCT_AST_FGM1', 'PCT_UAST_FGM1', 'OPP_FGM1', 'OPP_FGA1', 'OPP_FG_PCT1', 'OPP_FG3M1', 'OPP_FG3A1', 'OPP_FG3_PCT1', 'OPP_FTM1', 'OPP_FTA1', 'OPP_FT_PCT1', 'OPP_OREB1', 'OPP_DREB1', 'OPP_REB1', 'OPP_AST1', 'OPP_TOV1', 'OPP_STL1', 'OPP_BLK1', 'OPP_BLKA1', 'OPP_PF1', 'OPP_PFD1', 'OPP_PTS1', 'OPP_PTS_OFF_TOV1', 'OPP_PTS_2ND_CHANCE1', 'OPP_PTS_FB1', 'OPP_PTS_PAINT1', 'LAST_10_W_PCT1', 'TEAM_POWER1']
# removed 'W_PCT'

scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

player_columns_minus_name_v2 = [
    "MIN",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "REB",
    "AST",
    "STL",
    "BLK",
    "PLUS_MINUS",
]

city_to_abbr_dict = {
    "Atlanta": "ATL",
    "Boston": "BOS",
    "Brooklyn": "BKN",
    "Charlotte": "CHA",
    "Chicago": "CHI",
    "Cleveland": "CLE",
    "Dallas": "DAL",
    "Denver": "DEN",
    "Detroit": "DET",
    "Golden St.": "GSW",
    "Houston": "HOU",
    "Indiana": "IND",
    "L.A. Clippers": "LAC",
    "L.A. Lakers": "LAL",
    "Memphis": "MEM",
    "Miami": "MIA",
    "Milwaukee": "MIL",
    "Minnesota": "MIN",
    "New Orleans": "NOP",
    "New York": "NYK",
    "Oklahoma City": "OKC",
    "Orlando": "ORL",
    "Philadelphia": "PHI",
    "Phoenix": "PHO",
    "Portland": "POR",
    "Utah": "UTA",
    "San Antonio": "SAS",
    "Sacramento": "SAC",
    "Toronto": "TOR",
    "Washington": "WAS",
}
team_abbr_to_name = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "BKN": "Brooklyn Nets",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
    "SAS": "San Antonio Spurs",
}
team_abbr_to_id_dict = {
    "ATL": 1610612737,
    "BOS": 1610612738,
    "CLE": 1610612739,
    "NOP": 1610612740,
    "CHI": 1610612741,
    "DAL": 1610612742,
    "DEN": 1610612743,
    "GSW": 1610612744,
    "HOU": 1610612745,
    "LAC": 1610612746,
    "LAL": 1610612747,
    "MIA": 1610612748,
    "MIL": 1610612749,
    "MIN": 1610612750,
    "BKN": 1610612751,
    "NYK": 1610612752,
    "ORL": 1610612753,
    "IND": 1610612754,
    "PHI": 1610612755,
    "PHX": 1610612756,
    "POR": 1610612757,
    "SAC": 1610612758,
    "SAS": 1610612759,
    "OKC": 1610612760,
    "TOR": 1610612761,
    "UTA": 1610612762,
    "MEM": 1610612763,
    "WAS": 1610612764,
    "DET": 1610612765,
    "CHA": 1610612766,
}

def append_data_to_sheet(sheet_name: str, data: list):
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        "api_key.json", scope
    )
    gc = gspread.authorize(credentials)

    spreadsheet = gc.open("schrodinger")
    worksheet = spreadsheet.worksheet(sheet_name)
    worksheet.append_row(data)
    print("data append completed")


def find_team_name(tag: bs4.element.Tag) -> str:
    """
    Finds team name from a bs4 tag pulled previously

    Returns:
    team_name (str) : team name
    """
    pattern = r">(.*?)<\/a>" # regex pattern 
    team_name = str(tag.find_all(class_="TeamName")[0])
    team_name = re.search(pattern, team_name).group(1)
    team_name = team_name.split(">", 1)[-1]
    return team_name


def find_injuriesv2() -> dict:
    """
    CBSSports injury list (more accurate and up to date)

    Returns: 
    injuries (dict) : injured players around the league in {"team_abbr" : [injured players]} format
    """
    injuries = {}
    r = requests.get("https://www.cbssports.com/nba/injuries/")
    soup = BeautifulSoup(r.text, "html.parser")
    team_elements = soup.find_all(class_="TableBaseWrapper")
    for team_element in team_elements:
        team_name = find_team_name(team_element)
        team_abbr = city_to_abbr_dict[team_name]
        injuries[team_abbr] = []
        for report_element in team_element.find_all("tr", class_="TableBase-bodyTr"):
            player_name = report_element.find(
                "span", class_="CellPlayerName--long"
            ).text
            injuries[team_abbr].append(player_name)
    return injuries


def prep_training_data(data: pd.DataFrame, features):
    """
    Prepares data for model training by separating independent and dependent variables

    data (pd.DataFrame) : data prepared for training
    features (list) : specified features to filter

    Returns:
    X[features].values (list) : independent variables filtered by features in list format
    y (list) : dependent variables in list format
    """
    X = data.drop("HOME_TEAM_WIN", axis=1)
    y = pd.get_dummies(data, columns=["HOME_TEAM_WIN"], prefix="Result")[
        "Result_W"
    ].values # change W L to 0s and 1s
    return X[features].values, y


def scale_scores(scores: list) -> list:
    """
    Scales player importance scores on a team to add up to 100

    scores (list) : player importance scores

    Returns:
    scaled_values (list) : scaled player importance scores 
    """
    total = sum(scores)
    scaled_values = [value / total * 100 for value in scores]
    return scaled_values


def calculate_team_power(team_id: int, injuries: dict) -> float:
    """
    Calculates feature engineered "Team Power" which is a number between 1-100 that determines how full power a team is (taking into account injuries)

    team_id (int) : team id code from NBA API
    injuries (dict) : dictionary of injured players all throughout the league {"team_abbr" : [injured_players]}

    Returns:
    team_power (float) : team power score 
    """
    player_stats = TeamPlayerDashboard(
        team_id=team_id, per_mode_detailed="PerGame", season="2023-24"
    ).get_data_frames()[1]
    player_stats = player_stats.sort_values("MIN_RANK").iloc[0:8]
    player_stats["SCORE"] = player_stats[player_columns_minus_name_v2].sum(axis=1)
    scores = player_stats["SCORE"]
    player_stats["IMPORTANCE"] = scale_scores(scores)
    player_stats["IMPORTANCE"] = player_stats["IMPORTANCE"]
    team_power = 100
    team_abbr = find_team_name_by_id(team_id)["abbreviation"]
    if team_abbr in injuries:
        DNP_players_list = injuries[team_abbr]
    else:
        DNP_players_list = []
    print(team_abbr, DNP_players_list)
    for index, player in player_stats.iterrows():
        if player["PLAYER_NAME"] in DNP_players_list:
            team_power -= player["IMPORTANCE"]
    return team_power


def get_current_data() -> pd.DataFrame:
    """
    Retrieves current season data for real-time predicting purposes

    Returns:
    data (pd.DataFrame) : dataframe of all features we need in order to predict
    """
    base = LeagueDashTeamStats(
        season="2023-24",
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]
    advanced = LeagueDashTeamStats(
        season="2023-24",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]
    scoring = LeagueDashTeamStats(
        season="2023-24",
        measure_type_detailed_defense="Scoring",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]
    opponent = LeagueDashTeamStats(
        season="2023-24",
        measure_type_detailed_defense="Opponent",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]
    defense = LeagueDashTeamStats(
        season="2023-24",
        measure_type_detailed_defense="Defense",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]
    last_10 = LeagueDashTeamStats(
        last_n_games=10,
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame",
        season="2023-24",
    ).get_data_frames()[0]["W_PCT"]
    last_10.rename("LAST_10_W_PCT", inplace=True)
    return pd.concat([base, advanced, scoring, opponent, defense, last_10], axis=1)


def predict_winner(
    home_team_abbr: str,
    away_team_abbr: str,
    current_data: pd.DataFrame,
    features: list,
    pca_path : str,
    model_path : str,
    injuries: dict,
    sheet_name: str,
    game_id: str,
    scaler_path : str
):
    """
    Pipeline for real-time testing

    home_team_abbr (str) : home team abbreviation (CHI, BOS,...)
    home_team_abbr (str) : away team abbreviation (LAL, LAC,...)
    current_data (pd.DataFrame) : current season data for both teams used for training/predicting
    features (list) : list of features to use to predict
    pca_path (str) : file path for PCA 
    model_path (str) : Keras NN model file path
    injuries (dict) : Dictionary of injuries around the league {"team_abbr" : [injured_players]}
    sheet_name (str) : Personal use for Google Sheets
    game_id (str) : Game ID of the game to predict
    scaler_path (str) : file path for scaler 
    """
    home_team_id = team_abbr_to_id_dict[home_team_abbr]
    away_team_id = team_abbr_to_id_dict[away_team_abbr]

    current_data = current_data.loc[
        :, ~current_data.columns.duplicated()
    ]  # remove duplicate columns

    home_team_stats = current_data[
        current_data["TEAM_ID"] == home_team_id
    ]  # filter home team stats
    away_team_stats = current_data[
        current_data["TEAM_ID"] == away_team_id
    ]  # filter away team stats
    home_team_stats["TEAM_POWER"] = calculate_team_power(home_team_id, injuries)
    away_team_stats["TEAM_POWER"] = calculate_team_power(
        away_team_id, injuries
    )  # add team_power feature

    away_team_stats.columns = [col + "1" for col in away_team_stats.columns]
    home_team_stats_reset = home_team_stats.reset_index(drop=True)
    away_team_stats_reset = away_team_stats.reset_index(drop=True)

    test_data = pd.concat([home_team_stats_reset, away_team_stats_reset], axis=1)[
        features
    ].values

    scaler = joblib.load(scaler_path)
    test_data = scaler.transform(test_data)

    pca = pk.load(open(pca_path,'rb'))
    test_data = pca.transform(test_data)

    test_data = np.array(test_data).reshape(1, -1)

    model = keras.models.load_model(model_path)
    result = model.predict(test_data)

    home_team_win_percentage = result
    away_team_win_percentage = 1 - home_team_win_percentage

    data = [
            date.today().strftime("%m/%d/%Y"),
            home_team_abbr,
            away_team_abbr,
            home_team_abbr
            if home_team_win_percentage[0][0] > away_team_win_percentage[0][0]
            else away_team_abbr,
            float(home_team_win_percentage[0][0])
            if home_team_win_percentage[0][0] > away_team_win_percentage[0][0]
            else float(away_team_win_percentage[0][0]),
            game_id,
            None,
        ]
    
    append_data_to_sheet(sheet_name, data)

    print(home_team_abbr + ": " + str(home_team_win_percentage[0][0]))
    print(away_team_abbr + ": " + str(away_team_win_percentage[0][0]))


def predict_games():
    """
    predicts today's games and appends results to google sheets for each model type 
    """
    scoreboard = ScoreBoard()
    current_data = get_current_data()
    injuries = find_injuriesv2()

    # accuracy v2 model testing
    for game_dict in scoreboard.get_dict()["scoreboard"]["games"]:
        game_id = game_dict["gameId"]
        home_team_abbr = game_dict["homeTeam"]["teamTricode"]
        away_team_abbr = game_dict["awayTeam"]["teamTricode"]
        predict_winner(
            home_team_abbr,
            away_team_abbr,
            current_data,
            featuresv2,
            "./models/v2/accuracy_pca.pkl",
            "./models/v2/accuracy_model.keras",
            injuries,
            "accuracyv2",
            game_id,
            "./models/v2/accuracy_scaler.save"
        )
        time.sleep(3)

    # accuracy model testing
    for game_dict in scoreboard.get_dict()["scoreboard"]["games"]:
        game_id = game_dict["gameId"]
        home_team_abbr = game_dict["homeTeam"]["teamTricode"]
        away_team_abbr = game_dict["awayTeam"]["teamTricode"]
        predict_winner(
            home_team_abbr,
            away_team_abbr,
            current_data,
            featuresv1,
            "./models/v1/accuracy_pca.pkl",
            "./models/v1/accuracy_model.keras",
            injuries,
            "accuracy",
            game_id,
            "./models/v1/scaler.save"
        )
        time.sleep(3)

    # precision model testing
    for game_dict in scoreboard.get_dict()["scoreboard"]["games"]:
        game_id = game_dict["gameId"]
        home_team_abbr = game_dict["homeTeam"]["teamTricode"]
        away_team_abbr = game_dict["awayTeam"]["teamTricode"]
        predict_winner(
            home_team_abbr,
            away_team_abbr,
            current_data,
            featuresv1,
            "./models/v1/precision_pca.pkl",
            "./models/v1/precision_model.keras",
            injuries,
            "precision",
            game_id,
            "./models/v1/scaler.save"
        )
        time.sleep(3)

    # recall model testing
    for game_dict in scoreboard.get_dict()["scoreboard"]["games"]:
        game_id = game_dict["gameId"]
        home_team_abbr = game_dict["homeTeam"]["teamTricode"]
        away_team_abbr = game_dict["awayTeam"]["teamTricode"]
        predict_winner(
            home_team_abbr,
            away_team_abbr,
            current_data,
            featuresv1,
            "./models/v1/recall_pca.pkl",
            "./models/v1/recall_model.keras",
            injuries,
            "recall",
            game_id,
            "./models/v1/scaler.save"
        )
        time.sleep(3)


def find_results():
    """
    Find results of predicted games and append results to google sheets for each model 
    """
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        "api_key.json", scope
    )
    gc = gspread.authorize(credentials)
    spreadsheet = gc.open("schrodinger")
    worksheet = spreadsheet.worksheet("accuracyv2")
    truths = []

    for row_index, row_data in enumerate(worksheet.get_all_values()[1:], start=2):
        game_id = row_data[5]
        if row_data[6] == "":
            scoreboard = BoxScoreTraditionalV3(game_id)
            if (
                scoreboard.get_dict()["boxScoreTraditional"]["homeTeam"]["starters"][
                    "minutes"
                ]
                == ""
            ):
                pass
            else:
                home_team_points = scoreboard.get_dict()["boxScoreTraditional"]["homeTeam"][
                    "statistics"
                ]["points"]
                home_team_abbr = scoreboard.get_dict()["boxScoreTraditional"]["homeTeam"][
                    "teamTricode"
                ]
                away_team_points = scoreboard.get_dict()["boxScoreTraditional"]["awayTeam"][
                    "statistics"
                ]["points"]
                away_team_abbr = scoreboard.get_dict()["boxScoreTraditional"]["awayTeam"][
                    "teamTricode"
                ]
                if home_team_points > away_team_points:
                    differential = home_team_points - away_team_points
                    worksheet.update_cell(row_index, 7, home_team_abbr)
                    worksheet.update_cell(row_index, 8, differential)
                    truths.append((row_index, home_team_abbr, differential))
                    print("updated")
                else:
                    differential = away_team_points - home_team_points
                    worksheet.update_cell(row_index, 7, away_team_abbr)
                    worksheet.update_cell(row_index, 8, differential)
                    truths.append((row_index, away_team_abbr, differential))
                    print("updated")
                time.sleep(3)

    for sheet_name in ["accuracy", "precision", "recall"]:
        spreadsheet = gc.open("schrodinger")
        worksheet = spreadsheet.worksheet(sheet_name)
        for truth in truths:
            worksheet.update_cell(truth[0], 7, truth[1])
            worksheet.update_cell(truth[0], 8, truth[2])
            time.sleep(3)
