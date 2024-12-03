from prediction_tools import PredictionTools
from dotenv import load_dotenv
from constants import FEATURES_V2
import time
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

if __name__ == "__main__":
    games_today = PredictionTools.collect_games()
    for game in games_today:
        injuries = PredictionTools.find_injuriesv2()
        current_data = PredictionTools.get_current_data(current_season = "2024-25")
        game_id = game['game_id']
        home_team_abbr = game['home_team_abbr']
        away_team_abbr = game['away_team_abbr']
        PredictionTools.predict_game(home_team_abbr, away_team_abbr, current_data, FEATURES_V2, "./models/v2/pca.pkl", "./models/v2/model.keras", "./models/v2/scaler.save", injuries, "", game_id)
        time.sleep(15)