from nba_api.stats.endpoints import TeamPlayerDashboard, LeagueDashTeamStats,BoxScoreTraditionalV3
from nba_api.stats.static.teams import find_team_name_by_id
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
import os
from pymongo.mongo_client import MongoClient
import pytz
import datetime
import time
import requests
from constants import FEATURES_V2, SHEET_NAME, PLAYER_COLUMNS_MINUS_NAME_V2, CITY_TO_ABBR_DICT, TEAM_ABBR_TO_ID_DICT, TEAM_NAME_TO_ABBREVIATION, SCOPE, ENDPOINT, URI, DB_NAME

class PredictionTools:
    def get_odds(response : list) -> dict:
        """
        Retrieves today's odds for NBA games 

        response (list) : list of today's games from API call

        Returns:
        (dict) : Dictionary of odds in {"team_abbr" : odds (int)} format
        """
        odds_dict = {}
        for game in response:
            first_bookmaker = game['bookmakers'][0] # first bookmaker in list of bookmakers (idc which bookmaker)
            for market in first_bookmaker['markets']:
                if market['key'] == 'h2h':
                    outcomes = market['outcomes']
                    for outcome in outcomes:
                        team_name = outcome['name']
                        team_abbr = TEAM_NAME_TO_ABBREVIATION[team_name]
                        decimal_odds = outcome['price']
                        if decimal_odds >= 2:
                            american_odds = (decimal_odds - 1) * 100
                        else:
                            american_odds = (-100) / (decimal_odds - 1)
                        odds_dict[team_abbr] = round(american_odds)
        return odds_dict

    def append_data_to_sheet(sheet_name: str, data: list) -> None:
        """
        Appends prediction data to personal spreadsheet. 

        sheet_name (str) : spreadsheet name 
        data (list) : data to append 

        Returns:
        (None)
        """
        if sheet_name:
            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                "api_key.json", SCOPE
            )
            gc = gspread.authorize(credentials)

            spreadsheet = gc.open("schrodinger")
            worksheet = spreadsheet.worksheet(sheet_name)
            worksheet.append_row(data)
            print("data append completed")

    def find_team_name(tag: bs4.element.Tag) -> str:
        """
        Extracts team name from a BS4 tag. Used for identifying injured players' team

        tag (bs4.element.Tag) : tag element to extract team name

        Returns:
        (str) : team name
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
            team_name = PredictionTools.find_team_name(team_element)
            team_abbr = CITY_TO_ABBR_DICT[team_name]
            injuries[team_abbr] = []
            for report_element in team_element.find_all("tr", class_="TableBase-bodyTr"):
                player_name = report_element.find(
                    "span", class_="CellPlayerName--long"
                ).text
                injuries[team_abbr].append(player_name)
        return injuries

    def prep_training_data(data: pd.DataFrame, features : list) -> tuple:
        """
        Prepares data for model training by separating independent and dependent variables

        data (pd.DataFrame) : data prepared for training
        features (list) : specified features to filter

        Returns:
        (list, list) : independent variables filtered by features in list format, dependent variables in list format
        """
        X = data.drop("HOME_TEAM_WIN", axis=1)
        y = pd.get_dummies(data, columns=["HOME_TEAM_WIN"], prefix="Result")[
            "Result_W"
        ].values # change W L to 0s and 1s
        return X[features].values, y

    def scale_scores(scores : list) -> list:
        """
        Scales player importance scores by 100. 

        scores (list) : list of player scores to scale transform

        Returns:
        (list) : scaled player score values 
        """
        total = sum(scores)
        scaled_values = [value / total * 100 for value in scores]
        return scaled_values

    def calculate_team_power(team_id: int, injuries: dict, season : str) -> float:
        """
        Calculates feature engineered "Team Power" which is a number between 1-100 that determines how full power a team is, 
        taking into account players available for a game.

        team_id (int) : team id code from NBA API
        injuries (dict) : dictionary of injured players all throughout the league {"team_abbr" : [injured_players]}
        season (str) : current season to calculate off of

        Returns:
        (float) : team power score 
        """
        # collect player stats to calculate player importance. 
        player_stats = TeamPlayerDashboard(
            team_id=team_id, per_mode_detailed="PerGame", season=season
        ).get_data_frames()[1]
        player_stats = player_stats.sort_values("MIN_RANK").iloc[0:8]
        player_stats["SCORE"] = player_stats[PLAYER_COLUMNS_MINUS_NAME_V2].sum(axis=1)
        scores = player_stats["SCORE"]
        player_stats["IMPORTANCE"] = PredictionTools.scale_scores(scores)
        player_stats["IMPORTANCE"] = player_stats["IMPORTANCE"]
        team_power = 100
        team_abbr = find_team_name_by_id(team_id)["abbreviation"]
        if team_abbr in injuries:
            DNP_players_list = injuries[team_abbr]
        else:
            DNP_players_list = []
        for index, player in player_stats.iterrows():
            if player["PLAYER_NAME"] in DNP_players_list:
                team_power -= player["IMPORTANCE"]
        return team_power

    def get_current_data(current_season : str) -> pd.DataFrame:
        """
        Retrieves current season data of all teams used for real-time prediction. 

        current_season (str) : current year of prediction

        Returns:
        (pd.DataFrame) : dataframe of all features we need in order to predict
        """
        base = LeagueDashTeamStats(
            season=current_season,
            measure_type_detailed_defense="Base",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        advanced = LeagueDashTeamStats(
            season=current_season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        scoring = LeagueDashTeamStats(
            season=current_season,
            measure_type_detailed_defense="Scoring",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        opponent = LeagueDashTeamStats(
            season=current_season,
            measure_type_detailed_defense="Opponent",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        defense = LeagueDashTeamStats(
            season=current_season,
            measure_type_detailed_defense="Defense",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        last_10 = LeagueDashTeamStats(
            last_n_games=10,
            measure_type_detailed_defense="Base",
            per_mode_detailed="PerGame",
            season=current_season
        ).get_data_frames()[0]["W_PCT"]
        last_10.rename("LAST_10_W_PCT", inplace=True)
        return pd.concat([base, advanced, scoring, opponent, defense, last_10], axis=1)


    def predict_game(
        home_team_abbr: str,
        away_team_abbr: str,
        current_data: pd.DataFrame,
        features: list,
        pca_path : str,
        model_path : str,
        scaler_path : str,
        injuries: dict,
        sheet_name: str,
        game_id: str,
        season : str,
        odds : dict = {}
    ) -> None:
        """
        Pipeline for real-time prediction: predicts and appends results to google sheets

        home_team_abbr (str) : home team abbreviation (CHI, BOS,...)
        home_team_abbr (str) : away team abbreviation (LAL, LAC,...)
        current_data (pd.DataFrame) : current season data for both teams used for training/predicting
        features (list) : list of features to use to predict
        pca_path (str) : file path for PCA 
        model_path (str) : Keras NN model file path
        scaler_path (str) : file path for scaler 
        injuries (dict) : Dictionary of injuries around the league {"team_abbr" : [injured_players]}
        sheet_name (str) : Personal use for Google Sheets
        game_id (str) : Game ID of the game to predict
        
        Returns:
        (None)
        """
        home_team_id = TEAM_ABBR_TO_ID_DICT[home_team_abbr]
        away_team_id = TEAM_ABBR_TO_ID_DICT[away_team_abbr]

        current_data = current_data.loc[
            :, ~current_data.columns.duplicated()
        ]  # remove duplicate columns

        home_team_stats = current_data[
            current_data["TEAM_ID"] == home_team_id
        ]  # filter home team stats
        away_team_stats = current_data[
            current_data["TEAM_ID"] == away_team_id
        ]  # filter away team stats
        home_team_stats["TEAM_POWER"] = PredictionTools.calculate_team_power(home_team_id, injuries, season)
        away_team_stats["TEAM_POWER"] = PredictionTools.calculate_team_power(
            away_team_id, injuries, season
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

        if odds:
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
                    None,
                    odds[home_team_abbr] if home_team_win_percentage[0][0] > away_team_win_percentage[0][0]
                    else odds[away_team_abbr],
                    odds[away_team_abbr] if home_team_win_percentage[0][0] > away_team_win_percentage[0][0]
                    else odds[home_team_abbr] 
                ]
        else:
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
                    None
                ]
        PredictionTools.append_data_to_sheet(sheet_name, data)

        print(home_team_abbr + ": " + str(home_team_win_percentage[0][0]))
        print(away_team_abbr + ": " + str(away_team_win_percentage[0][0]))
        print("----------------" * 15)

    def prediction_pipeline():
        """
        Pipeline that infinitely checks and predicts upcoming games. 
        """
        client = MongoClient(URI)
        database = client[DB_NAME]
        games_collection = database.Games
        try:
            while True:
                injuries = None
                current_data = None
                odds = None
                time_now = datetime.now(pytz.timezone('America/Chicago'))

                if time_now.time() <= time(13, 0) and time_now.time() >= time(11, 0) and games_collection.count_documents({}) == 0: # if current time is between 11:00am and 1:00pm and there are no games in the database
                    games = PredictionTools.collect_games()
                    for game in games:
                        result = games_collection.insert_one(game) # insert today's games to database
                        if result.inserted_id:
                            print("Game added to DB successfully")
                        else:
                            print("Game wasn't added for some reason, check it out")

                elif games_collection.count_documents({}) > 0: # games are loaded in the database
                    predicted_games = []
                    for game in games_collection.find(): 
                        if game['predicted'] == False: # iterate through games loaded in database and check if they were predicted yet
                            utc_gametime = datetime.strptime(game['game_time'], "%Y-%m-%dT%H:%M:%SZ")
                            utc_timezone = pytz.utc
                            cst_timezone = pytz.timezone('America/Chicago')  # CST time zone
                            cst_gametime = utc_timezone.localize(utc_gametime).astimezone(cst_timezone)
                            if (cst_gametime - time_now).total_seconds() <= 3600: # if game time is within 1 hour of current time  
                                if not injuries and not current_data: # save API costs by minimizing calls
                                    injuries = PredictionTools.find_injuriesv2()
                                    current_data = PredictionTools.get_current_data()
                                    response = requests.get(ENDPOINT)
                                    odds = PredictionTools.get_odds(response.json())

                                game_id = game['game_id']
                                home_team_abbr = game['home_team_abbr']
                                away_team_abbr = game['away_team_abbr']
                                PredictionTools.predict_game(home_team_abbr, away_team_abbr, current_data, FEATURES_V2, "./models/v2/pca.pkl", "./models/v2/model.keras", "./models/v2/scaler.save", injuries, SHEET_NAME, game_id, odds)
                            else:
                                print("game not within range")
                    for id in predicted_games: # delete predicted games from the database
                        games_collection.delete_one({"_id": id})
                        print("Deleted document:", id)

                elif (time_now.time() >= time(23, 0) or time_now.time() <= time(1, 0)):
                    PredictionTools.find_results()
                else:
                    print("not time yet")

                print("sleep for 15")
                time.sleep(900) # sleep for 15 minutes

        except Exception as e:
            print("Something went wrong: ", e)

    def find_and_update_results() -> None:
        """
        Find results of predicted games and append results to google sheets for each model.

        Returns:
        (None)
        """
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            "api_key.json", SCOPE
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

        for sheet_name in ["accuracy"]:
            spreadsheet = gc.open("schrodinger")
            worksheet = spreadsheet.worksheet(sheet_name)
            for truth in truths:
                worksheet.update_cell(truth[0], 7, truth[1])
                worksheet.update_cell(truth[0], 8, truth[2])
                time.sleep(3)

    def collect_games() -> list[dict]:
        """
        Collects today's NBA games using NBA api. Used for iterating through today's games and predicting each one.

        Returns:
        (list) : dictionary of games 
        """
        scoreboard = ScoreBoard()
        games = []
        for game_dict in scoreboard.get_dict()["scoreboard"]["games"]:
            game_id = game_dict["gameId"]
            home_team_abbr = game_dict["homeTeam"]["teamTricode"]
            away_team_abbr = game_dict["awayTeam"]["teamTricode"]
            game_time = game_dict['gameTimeUTC']
            game_status = game_dict['gameStatus']
            game_data = {"game_id" : game_id, "home_team_abbr" : home_team_abbr, "away_team_abbr" : away_team_abbr, "game_time" : game_time, "game_status" : game_status, "predicted" : False}
            games.append(game_data)
        return games

    def notify(title : str, text : str) -> None:
        """
        Internal tool used for notifying predictions. 

        Returns:
        (None)
        """
        os.system("""
                osascript -e 'display notification "{}" with title "{}"'
                """.format(text, title))