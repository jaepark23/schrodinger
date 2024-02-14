from func import predict_games, find_results
import keras
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    option = input("enter: ")
    api_key = os.environ['ODDS_API_KEY']
    if option == "1":
        NN_model = keras.models.load_model("./models/accuray_model.keras")
        precision_NN_model = keras.models.load_model("./models/precision_model.keras")
        recall_NN_model = keras.models.load_model("./models/recall_model.keras")
        predict_games(NN_model, precision_NN_model, recall_NN_model)
    else:
        find_results()