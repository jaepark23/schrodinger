import pandas as pd
from nba_api.stats.endpoints import BoxScorePlayerTrackV3
from nba_api.stats.endpoints import LeagueDashTeamStats
from nba_api.stats.endpoints import TeamPlayerDashboard
from nba_api.stats.endpoints import LeagueGameFinder
from datetime import datetime
import time
from constants import player_columns_minus_name_v2, seasons, cols, base_cols, advanced_cols, scoring_cols, opponent_cols, defense_cols

def scale_scores(scores):
    total = sum(scores)
    scaled_values = [value / total * 100 for value in scores]
    return scaled_values


def get_dnp_players(game_id) -> list:
    box_score = BoxScorePlayerTrackV3(str(game_id)).get_data_frames()[0]
    box_score["seconds"] = box_score["minutes"].apply(
        lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1])
    )
    DNP_players = box_score[box_score["seconds"] == 0]
    DNP_players_list = []
    for index, player in DNP_players.iterrows():
        full_name = player["firstName"] + " " + player["familyName"]
        DNP_players_list.append(full_name)
    return DNP_players_list


def get_team_power(date_start, date_end, season, team_id, DNP_players_list) -> float:
    player_stats = TeamPlayerDashboard(
        team_id=team_id,
        per_mode_detailed="PerGame",
        date_from_nullable=date_start,
        date_to_nullable=date_end,
        season=season,
    ).get_data_frames()[1]
    player_stats = player_stats.sort_values("MIN_RANK").iloc[0:8]
    player_stats["SCORE"] = player_stats[player_columns_minus_name_v2].sum(axis=1)
    scores = player_stats["SCORE"]
    player_stats["IMPORTANCE"] = scale_scores(scores)
    team_power = 100
    for index, player in player_stats.iterrows():
        if player["PLAYER_NAME"] in DNP_players_list:
            team_power -= player["IMPORTANCE"]
    return team_power


df = pd.DataFrame(columns=cols)
for season in seasons:
    year_start = season[0:4]
    year_end = "20" + str(season[5:7])
    games = (
        LeagueGameFinder(
            date_from_nullable="11/20/" + year_start,
            season_nullable=season,
            league_id_nullable="00",
        )
        .get_data_frames()[0]
        .sort_values(["GAME_DATE", "GAME_ID"])
    )
    counter = 0
    new_row = [None for i in range(342)]
    prev_date = ""
    for index, game in games.iterrows():
        try:
            counter += 1
            team_name = game["TEAM_NAME"]
            team_id = game["TEAM_ID"]
            team_name_short = team_name.split()[-1]
            game_date = game["GAME_DATE"]
            game_id = game["GAME_ID"]
            if prev_date != game_date:
                # new set of stats if new date
                base = LeagueDashTeamStats(
                    date_from_nullable="11/20/" + year_start,
                    date_to_nullable=datetime.strptime(game_date, "%Y-%m-%d").strftime(
                        "%m/%d/%Y"
                    ),
                    season=season,
                    measure_type_detailed_defense="Base",
                    per_mode_detailed="PerGame",
                ).get_data_frames()[0]
                advanced = LeagueDashTeamStats(
                    date_from_nullable="11/20/" + year_start,
                    date_to_nullable=datetime.strptime(game_date, "%Y-%m-%d").strftime(
                        "%m/%d/%Y"
                    ),
                    season=season,
                    measure_type_detailed_defense="Advanced",
                    per_mode_detailed="PerGame",
                ).get_data_frames()[0]
                scoring = LeagueDashTeamStats(
                    date_from_nullable="11/20/" + year_start,
                    date_to_nullable=datetime.strptime(game_date, "%Y-%m-%d").strftime(
                        "%m/%d/%Y"
                    ),
                    season=season,
                    measure_type_detailed_defense="Scoring",
                    per_mode_detailed="PerGame",
                ).get_data_frames()[0]
                opponent = LeagueDashTeamStats(
                    date_from_nullable="11/20/" + year_start,
                    date_to_nullable=datetime.strptime(game_date, "%Y-%m-%d").strftime(
                        "%m/%d/%Y"
                    ),
                    season=season,
                    measure_type_detailed_defense="Opponent",
                    per_mode_detailed="PerGame",
                ).get_data_frames()[0]
                defense = LeagueDashTeamStats(
                    date_from_nullable="11/20/" + year_start,
                    date_to_nullable=datetime.strptime(game_date, "%Y-%m-%d").strftime(
                        "%m/%d/%Y"
                    ),
                    season=season,
                    measure_type_detailed_defense="Defense",
                    per_mode_detailed="PerGame",
                ).get_data_frames()[0]
                last_10 = LeagueDashTeamStats(
                    last_n_games=10,
                    measure_type_detailed_defense="Base",
                    date_from_nullable=datetime.strptime(
                        game_date, "%Y-%m-%d"
                    ).strftime("%m/%d/%Y"),
                    per_mode_detailed="PerGame",
                    season=season,
                ).get_data_frames()[0]
            base_stats = (
                base.loc[base["TEAM_NAME"] == team_name][base_cols]
                .values.flatten()
                .tolist()
            )
            advanced_stats = (
                advanced.loc[advanced["TEAM_NAME"] == team_name][advanced_cols]
                .values.flatten()
                .tolist()
            )
            scoring_stats = (
                scoring.loc[scoring["TEAM_NAME"] == team_name][scoring_cols]
                .values.flatten()
                .tolist()
            )
            opponent_stats = (
                opponent.loc[opponent["TEAM_NAME"] == team_name][opponent_cols]
                .values.flatten()
                .tolist()
            )
            defense_stats = (
                defense.loc[defense["TEAM_NAME"] == team_name][defense_cols]
                .values.flatten()
                .tolist()
            )
            last_10_win_pct = (
                last_10.loc[last_10["TEAM_NAME"] == team_name]["W_PCT"]
                .values.flatten()
                .tolist()
            )
            if len(last_10_win_pct) > 0:
                last_10_win_pct = last_10_win_pct[0]
            else:
                last_10_win_pct = None
            print(last_10_win_pct)
            DNP_players_list = get_dnp_players(game_id)
            team_power = get_team_power(
                "11/01/" + year_start,
                datetime.strptime(game_date, "%Y-%m-%d").strftime("%m/%d/%Y"),
                season,
                team_id,
                DNP_players_list,
            )
            new_row[341] = game_date
            if "@" not in game["MATCHUP"]:
                # home game
                new_row[0:58] = base_stats
                new_row[58:89] = advanced_stats
                new_row[89:119] = scoring_stats
                new_row[119:159] = opponent_stats
                new_row[159:167] = defense_stats
                new_row[167] = last_10_win_pct
                new_row[168] = team_power
                new_row[338] = game["WL"]
                new_row[339] = team_name
            else:
                # away game
                new_row[169:227] = base_stats
                new_row[227:258] = advanced_stats
                new_row[258:288] = scoring_stats
                new_row[288:328] = opponent_stats
                new_row[328:336] = defense_stats
                new_row[336] = last_10_win_pct
                new_row[337] = team_power
                new_row[340] = team_name
            if counter == 2:
                counter = 0
                print(new_row)
                df.loc[len(df.index)] = new_row
                new_row = [None for i in range(342)]
            prev_date = game["GAME_DATE"]
            print(game_date)
            time.sleep(5)
        except Exception as e:
            if counter == 2:
                counter = 0
                print(new_row)
                # df.loc[len(df.index)] = new_row
                new_row = [None for i in range(342)]
            print("ERROR: ")
            print(e)

    df.to_csv("./data/" + str(season) + ".csv")
    df = pd.DataFrame(columns=cols)
