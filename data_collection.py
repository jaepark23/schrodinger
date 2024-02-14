import pandas as pd
from nba_api.stats.endpoints import BoxScorePlayerTrackV3
from nba_api.stats.static import teams
from nba_api.stats.endpoints import LeagueDashTeamStats
from nba_api.stats.endpoints import TeamPlayerDashboard
from calendar import monthrange
from nba_api.stats.endpoints import LeagueGameFinder
from datetime import datetime
import time

player_columns = [
    "PLAYER_NAME",
    "W_PCT_RANK",
    "FGM_RANK",
    "FGA_RANK",
    "FG_PCT_RANK",
    "REB_RANK",
    "AST_RANK",
    "TOV_RANK",
    "STL_RANK",
    "BLK_RANK",
    "PLUS_MINUS_RANK",
]
player_columns_minus_name = [
    "W_PCT_RANK",
    "FGM_RANK",
    "FGA_RANK",
    "FG_PCT_RANK",
    "REB_RANK",
    "AST_RANK",
    "TOV_RANK",
    "STL_RANK",
    "BLK_RANK",
    "PLUS_MINUS_RANK",
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
seasons = [
    "2013-14",
    "2014-15",
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
]

seasons = [
    # "2022-23",
    # "2021-22",
    # "2020-21",
    # "2019-20",
    # "2018-19",
    # "2017-18",
    # "2016-17",
    # "2015-16",
    # "2014-15",
    # "2013-14",
    # "2010-11",
    "2009-10",
    "2008-09",
    "2007-08",
    "2006-07",
    "2005-06",
    "2004-05",
    "2003-04",
    "2002-03",
    "2001-02",
]
cols = [
    "W_PCT",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "BLKA",
    "PF",
    "PFD",
    "PTS",
    "PLUS_MINUS",
    "W_PCT_RANK",
    "MIN_RANK",
    "FGM_RANK",
    "FGA_RANK",
    "FG_PCT_RANK",
    "FG3M_RANK",
    "FG3A_RANK",
    "FG3_PCT_RANK",
    "FTM_RANK",
    "FTA_RANK",
    "FT_PCT_RANK",
    "OREB_RANK",
    "DREB_RANK",
    "REB_RANK",
    "AST_RANK",
    "TOV_RANK",
    "STL_RANK",
    "BLK_RANK",
    "FG3M_RANK",
    "FG3A_RANK",
    "FG3_PCT_RANK",
    "FTM_RANK",
    "FTA_RANK",
    "FT_PCT_RANK",
    "OREB_RANK",
    "DREB_RANK",
    "REB_RANK",
    "AST_RANK",
    "TOV_RANK",
    "STL_RANK",
    "BLK_RANK",
    "BLKA_RANK",
    "PF_RANK",
    "PFD_RANK",
    "PTS_RANK",
    "PLUS_MINUS_RANK",
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "AST_PCT",
    "AST_TO",
    "AST_RATIO",
    "OREB_PCT",
    "DREB_PCT",
    "REB_PCT",
    "TM_TOV_PCT",
    "EFG_PCT",
    "TS_PCT",
    "E_PACE",
    "PACE",
    "PACE_PER40",
    "POSS",
    "PIE",
    "OFF_RATING_RANK",
    "DEF_RATING_RANK",
    "NET_RATING_RANK",
    "AST_PCT_RANK",
    "AST_TO_RANK",
    "AST_RATIO_RANK",
    "OREB_PCT_RANK",
    "DREB_PCT_RANK",
    "REB_PCT_RANK",
    "TM_TOV_PCT_RANK",
    "EFG_PCT_RANK",
    "TS_PCT_RANK",
    "PACE_RANK",
    "PIE_RANK",
    "PCT_FGA_2PT",
    "PCT_FGA_3PT",
    "PCT_PTS_2PT",
    "PCT_PTS_2PT_MR",
    "PCT_PTS_3PT",
    "PCT_PTS_FB",
    "PCT_PTS_FT",
    "PCT_PTS_OFF_TOV",
    "PCT_PTS_PAINT",
    "PCT_AST_2PM",
    "PCT_UAST_2PM",
    "PCT_AST_3PM",
    "PCT_UAST_3PM",
    "PCT_AST_FGM",
    "PCT_UAST_FGM",
    "PCT_FGA_2PT_RANK",
    "PCT_FGA_3PT_RANK",
    "PCT_PTS_2PT_RANK",
    "PCT_PTS_2PT_MR_RANK",
    "PCT_PTS_3PT_RANK",
    "PCT_PTS_FB_RANK",
    "PCT_PTS_FT_RANK",
    "PCT_PTS_OFF_TOV_RANK",
    "PCT_PTS_PAINT_RANK",
    "PCT_AST_2PM_RANK",
    "PCT_UAST_2PM_RANK",
    "PCT_AST_3PM_RANK",
    "PCT_UAST_3PM_RANK",
    "PCT_AST_FGM_RANK",
    "PCT_UAST_FGM_RANK",
    "OPP_FGM",
    "OPP_FGA",
    "OPP_FG_PCT",
    "OPP_FG3M",
    "OPP_FG3A",
    "OPP_FG3_PCT",
    "OPP_FTM",
    "OPP_FTA",
    "OPP_FT_PCT",
    "OPP_OREB",
    "OPP_DREB",
    "OPP_REB",
    "OPP_AST",
    "OPP_TOV",
    "OPP_STL",
    "OPP_BLK",
    "OPP_BLKA",
    "OPP_PF",
    "OPP_PFD",
    "OPP_PTS",
    "OPP_FGM_RANK",
    "OPP_FGA_RANK",
    "OPP_FG_PCT_RANK",
    "OPP_FG3M_RANK",
    "OPP_FG3A_RANK",
    "OPP_FG3_PCT_RANK",
    "OPP_FTM_RANK",
    "OPP_FTA_RANK",
    "OPP_FT_PCT_RANK",
    "OPP_OREB_RANK",
    "OPP_DREB_RANK",
    "OPP_REB_RANK",
    "OPP_AST_RANK",
    "OPP_TOV_RANK",
    "OPP_STL_RANK",
    "OPP_BLK_RANK",
    "OPP_BLKA_RANK",
    "OPP_PF_RANK",
    "OPP_PFD",
    "OPP_PTS_RANK",
    "OPP_PTS_OFF_TOV",
    "OPP_PTS_2ND_CHANCE",
    "OPP_PTS_FB",
    "OPP_PTS_PAINT",
    "OPP_PTS_OFF_TOV_RANK",
    "OPP_PTS_2ND_CHANCE_RANK",
    "OPP_PTS_FB_RANK",
    "OPP_PTS_PAINT_RANK",
    "LAST_10_W_PCT",
    "TEAM_POWER",
    "W_PCT1",
    "FGM1",
    "FGA1",
    "FG_PCT1",
    "FG3M1",
    "FG3A1",
    "FG3_PCT1",
    "FTM1",
    "FTA1",
    "FT_PCT1",
    "OREB1",
    "DREB1",
    "REB1",
    "AST1",
    "TOV1",
    "STL1",
    "BLK1",
    "BLKA1",
    "PF1",
    "PFD1",
    "PTS1",
    "PLUS_MINUS1",
    "W_PCT_RANK1",
    "MIN_RANK1",
    "FGM_RANK1",
    "FGA_RANK1",
    "FG_PCT_RANK1",
    "FG3M_RANK1",
    "FG3A_RANK1",
    "FG3_PCT_RANK1",
    "FTM_RANK1",
    "FTA_RANK1",
    "FT_PCT_RANK1",
    "OREB_RANK1",
    "DREB_RANK1",
    "REB_RANK1",
    "AST_RANK1",
    "TOV_RANK1",
    "STL_RANK1",
    "BLK_RANK1",
    "FG3M_RANK1",
    "FG3A_RANK1",
    "FG3_PCT_RANK1",
    "FTM_RANK1",
    "FTA_RANK1",
    "FT_PCT_RANK1",
    "OREB_RANK1",
    "DREB_RANK1",
    "REB_RANK1",
    "AST_RANK1",
    "TOV_RANK1",
    "STL_RANK1",
    "BLK_RANK1",
    "BLKA_RANK1",
    "PF_RANK1",
    "PFD_RANK1",
    "PTS_RANK1",
    "PLUS_MINUS_RANK1",
    "OFF_RATING1",
    "DEF_RATING1",
    "NET_RATING1",
    "AST_PCT1",
    "AST_TO1",
    "AST_RATIO1",
    "OREB_PCT1",
    "DREB_PCT1",
    "REB_PCT1",
    "TM_TOV_PCT1",
    "EFG_PCT1",
    "TS_PCT1",
    "E_PACE1",
    "PACE1",
    "PACE_PER401",
    "POSS1",
    "PIE1",
    "OFF_RATING_RANK1",
    "DEF_RATING_RANK1",
    "NET_RATING_RANK1",
    "AST_PCT_RANK1",
    "AST_TO_RANK1",
    "AST_RATIO_RANK1",
    "OREB_PCT_RANK1",
    "DREB_PCT_RANK1",
    "REB_PCT_RANK1",
    "TM_TOV_PCT_RANK1",
    "EFG_PCT_RANK1",
    "TS_PCT_RANK1",
    "PACE_RANK1",
    "PIE_RANK1",
    "PCT_FGA_2PT1",
    "PCT_FGA_3PT1",
    "PCT_PTS_2PT1",
    "PCT_PTS_2PT_MR1",
    "PCT_PTS_3PT1",
    "PCT_PTS_FB1",
    "PCT_PTS_FT1",
    "PCT_PTS_OFF_TOV1",
    "PCT_PTS_PAINT1",
    "PCT_AST_2PM1",
    "PCT_UAST_2PM1",
    "PCT_AST_3PM1",
    "PCT_UAST_3PM1",
    "PCT_AST_FGM1",
    "PCT_UAST_FGM1",
    "PCT_FGA_2PT_RANK1",
    "PCT_FGA_3PT_RANK1",
    "PCT_PTS_2PT_RANK1",
    "PCT_PTS_2PT_MR_RANK1",
    "PCT_PTS_3PT_RANK1",
    "PCT_PTS_FB_RANK1",
    "PCT_PTS_FT_RANK1",
    "PCT_PTS_OFF_TOV_RANK1",
    "PCT_PTS_PAINT_RANK1",
    "PCT_AST_2PM_RANK1",
    "PCT_UAST_2PM_RANK1",
    "PCT_AST_3PM_RANK1",
    "PCT_UAST_3PM_RANK1",
    "PCT_AST_FGM_RANK1",
    "PCT_UAST_FGM_RANK1",
    "OPP_FGM1",
    "OPP_FGA1",
    "OPP_FG_PCT1",
    "OPP_FG3M1",
    "OPP_FG3A1",
    "OPP_FG3_PCT1",
    "OPP_FTM1",
    "OPP_FTA1",
    "OPP_FT_PCT1",
    "OPP_OREB1",
    "OPP_DREB1",
    "OPP_REB1",
    "OPP_AST1",
    "OPP_TOV1",
    "OPP_STL1",
    "OPP_BLK1",
    "OPP_BLKA1",
    "OPP_PF1",
    "OPP_PFD1",
    "OPP_PTS1",
    "OPP_FGM_RANK1",
    "OPP_FGA_RANK1",
    "OPP_FG_PCT_RANK1",
    "OPP_FG3M_RANK1",
    "OPP_FG3A_RANK1",
    "OPP_FG3_PCT_RANK1",
    "OPP_FTM_RANK1",
    "OPP_FTA_RANK1",
    "OPP_FT_PCT_RANK1",
    "OPP_OREB_RANK1",
    "OPP_DREB_RANK1",
    "OPP_REB_RANK1",
    "OPP_AST_RANK1",
    "OPP_TOV_RANK1",
    "OPP_STL_RANK1",
    "OPP_BLK_RANK1",
    "OPP_BLKA_RANK1",
    "OPP_PF_RANK1",
    "OPP_PFD1",
    "OPP_PTS_RANK1",
    "OPP_PTS_OFF_TOV1",
    "OPP_PTS_2ND_CHANCE1",
    "OPP_PTS_FB1",
    "OPP_PTS_PAINT1",
    "OPP_PTS_OFF_TOV_RANK1",
    "OPP_PTS_2ND_CHANCE_RANK1",
    "OPP_PTS_FB_RANK1",
    "OPP_PTS_PAINT_RANK1",
    "LAST_10_W_PCT1",
    "TEAM_POWER1",
    "HOME_TEAM_WIN",
    "HOME_TEAM_NAME",
    "AWAY_TEAM_NAME",
    "DATE",
]

base_cols = [
    "W_PCT",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "BLKA",
    "PF",
    "PFD",
    "PTS",
    "PLUS_MINUS",
    "W_PCT_RANK",
    "MIN_RANK",
    "FGM_RANK",
    "FGA_RANK",
    "FG_PCT_RANK",
    "FG3M_RANK",
    "FG3A_RANK",
    "FG3_PCT_RANK",
    "FTM_RANK",
    "FTA_RANK",
    "FT_PCT_RANK",
    "OREB_RANK",
    "DREB_RANK",
    "REB_RANK",
    "AST_RANK",
    "TOV_RANK",
    "STL_RANK",
    "BLK_RANK",
    "FG3M_RANK",
    "FG3A_RANK",
    "FG3_PCT_RANK",
    "FTM_RANK",
    "FTA_RANK",
    "FT_PCT_RANK",
    "OREB_RANK",
    "DREB_RANK",
    "REB_RANK",
    "AST_RANK",
    "TOV_RANK",
    "STL_RANK",
    "BLK_RANK",
    "BLKA_RANK",
    "PF_RANK",
    "PFD_RANK",
    "PTS_RANK",
    "PLUS_MINUS_RANK",
]

advanced_cols = [
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "AST_PCT",
    "AST_TO",
    "AST_RATIO",
    "OREB_PCT",
    "DREB_PCT",
    "REB_PCT",
    "TM_TOV_PCT",
    "EFG_PCT",
    "TS_PCT",
    "E_PACE",
    "PACE",
    "PACE_PER40",
    "POSS",
    "PIE",
    "OFF_RATING_RANK",
    "DEF_RATING_RANK",
    "NET_RATING_RANK",
    "AST_PCT_RANK",
    "AST_TO_RANK",
    "AST_RATIO_RANK",
    "OREB_PCT_RANK",
    "DREB_PCT_RANK",
    "REB_PCT_RANK",
    "TM_TOV_PCT_RANK",
    "EFG_PCT_RANK",
    "TS_PCT_RANK",
    "PACE_RANK",
    "PIE_RANK",
]

scoring_cols = [
    "PCT_FGA_2PT",
    "PCT_FGA_3PT",
    "PCT_PTS_2PT",
    "PCT_PTS_2PT_MR",
    "PCT_PTS_3PT",
    "PCT_PTS_FB",
    "PCT_PTS_FT",
    "PCT_PTS_OFF_TOV",
    "PCT_PTS_PAINT",
    "PCT_AST_2PM",
    "PCT_UAST_2PM",
    "PCT_AST_3PM",
    "PCT_UAST_3PM",
    "PCT_AST_FGM",
    "PCT_UAST_FGM",
    "PCT_FGA_2PT_RANK",
    "PCT_FGA_3PT_RANK",
    "PCT_PTS_2PT_RANK",
    "PCT_PTS_2PT_MR_RANK",
    "PCT_PTS_3PT_RANK",
    "PCT_PTS_FB_RANK",
    "PCT_PTS_FT_RANK",
    "PCT_PTS_OFF_TOV_RANK",
    "PCT_PTS_PAINT_RANK",
    "PCT_AST_2PM_RANK",
    "PCT_UAST_2PM_RANK",
    "PCT_AST_3PM_RANK",
    "PCT_UAST_3PM_RANK",
    "PCT_AST_FGM_RANK",
    "PCT_UAST_FGM_RANK",
]

opponent_cols = [
    "OPP_FGM",
    "OPP_FGA",
    "OPP_FG_PCT",
    "OPP_FG3M",
    "OPP_FG3A",
    "OPP_FG3_PCT",
    "OPP_FTM",
    "OPP_FTA",
    "OPP_FT_PCT",
    "OPP_OREB",
    "OPP_DREB",
    "OPP_REB",
    "OPP_AST",
    "OPP_TOV",
    "OPP_STL",
    "OPP_BLK",
    "OPP_BLKA",
    "OPP_PF",
    "OPP_PFD",
    "OPP_PTS",
    "OPP_FGM_RANK",
    "OPP_FGA_RANK",
    "OPP_FG_PCT_RANK",
    "OPP_FG3M_RANK",
    "OPP_FG3A_RANK",
    "OPP_FG3_PCT_RANK",
    "OPP_FTM_RANK",
    "OPP_FTA_RANK",
    "OPP_FT_PCT_RANK",
    "OPP_OREB_RANK",
    "OPP_DREB_RANK",
    "OPP_REB_RANK",
    "OPP_AST_RANK",
    "OPP_TOV_RANK",
    "OPP_STL_RANK",
    "OPP_BLK_RANK",
    "OPP_BLKA_RANK",
    "OPP_PF_RANK",
    "OPP_PFD",
    "OPP_PTS_RANK",
]

defense_cols = [
    "OPP_PTS_OFF_TOV",
    "OPP_PTS_2ND_CHANCE",
    "OPP_PTS_FB",
    "OPP_PTS_PAINT",
    "OPP_PTS_OFF_TOV_RANK",
    "OPP_PTS_2ND_CHANCE_RANK",
    "OPP_PTS_FB_RANK",
    "OPP_PTS_PAINT_RANK",
]


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
