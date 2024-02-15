from func import predict_games, find_results
import keras
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    option = input("enter: ")
    api_key = os.environ['ODDS_API_KEY']
    if option == "1":
        predict_games()
    else:
        find_results()