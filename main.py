# from func import predict_games, find_results
# from dotenv import load_dotenv

from func import collect_games, find_injuriesv2, get_current_data, get_odds, predict_game
import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from datetime import datetime, time
import time as t
import pytz
import requests

load_dotenv()

uri = os.environ["URI"]
db_name = os.environ["DB_NAME"]
odds_api_key = os.environ['ODDS_API_KEY']
endpoint = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?regions=us&markets=h2h&apiKey={odds_api_key}"

if __name__ == "__main__":
    client = MongoClient(uri)
    database = client[db_name]
    games_collection = database.Games
    try:
        while True:
            injuries = None
            current_data = None
            odds = None
            time_now = datetime.now(pytz.timezone('America/Chicago'))

            if time_now.time() <= time(13, 0) and time_now.time() >= time(11, 0) and games_collection.count_documents({}) == 0: # if current time is between 11:00am and 1:00pm and there are no games in the database
                games = collect_games()
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
                                injuries = find_injuriesv2()
                                current_data = get_current_data()
                                response = requests.get(endpoint)
                                odds = get_odds(response.json())

                            game_id = game['game_id']
                            home_team_abbr = game['home_team_abbr']
                            away_team_abbr = game['away_team_abbr']
                            if predict_game(game_id, home_team_abbr, away_team_abbr, current_data, injuries, odds):
                                print("Game " + str(game_id) + " predicted successfully")
                                predicted_games.append(game["_id"])
                            else:
                                print("Game " + str(game_id) + " not predicted successfully")
                        else:
                            print("game not within range")
                for id in predicted_games: # delete predicted games from the database
                    games_collection.delete_one({"_id": id})
                    print("Deleted document:", id)
            else:
                print("not time yet")

            print("sleep for 15")
            t.sleep(900) # sleep for 15 minutes
    except Exception as e:
        print("Something went wrong: ", e)


# if __name__ == "__main__":
#     load_dotenv()
#     option = input("enter: ")
#     if option == "1":
#         predict_games()
#     else:
#         find_results()