

class Team:
    def __init__(self, team_abbr):
        self.offdef_ratings = self.find_offdef_ratings(team_abbr)
        self.team_abbr = team_abbr
        self.df = pd.read_html(
            "https://www.basketball-reference.com/teams/%s/2024.html" % team_abbr
        )
        self.injuries = self.find_injuries()
        self.player_scores = self.calculate_player_scores()
        self.team_score_with, self.team_score_without = self.calculate_team_score()

    def calculate_player_scores(self):
        """
        return (dict): player --> sum of weights
        """
        player_weights = [
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
        ]  # points, assists, rebounds, win shares, value over replacement player

        player_scores = {}

        for index, player in self.df[1].iterrows():  # pts, ast, rebs
            name = player["Player"]
            player_scores[name] = []
            player_scores[name].append(player["PTS"])
            player_scores[name].append(player["AST"])
            player_scores[name].append(player["TRB"])

        for index, player in self.df[3].iterrows():
            name = player["Player"]
            player_scores[name].append(player["BPM"])
            player_scores[name].append(player["VORP"])

        scores = []
        for player in player_scores:
            for i in range(len(player_scores[player])):
                player_scores[player][i] = player_scores[player][i] * player_weights[i]
            player_scores[player] = sum(player_scores[player])
            scores.append(player_scores[player])
        scores = normalize(scores)
        for i, player in enumerate(player_scores):
            player_scores[player] = scores[i]

        return player_scores

    def calculate_team_score(self):
        """
        players_scores (dict): player: value
        injuries (list) : [[out players], [day to day players]]
        return (float): team score with, team score without
        """

        team_score_with = 1  # consider day to day players as out
        team_score_without = 1  # don't consider day to day players as out

        for lst in self.injuries:
            for player in lst:
                if (
                    player not in self.player_scores
                ):  # for whatever reason, player doesnt show up on roster/injury report
                    pass
                else:
                    score = self.player_scores[player]
                    team_score_with -= score

        for player in self.injuries[0]:
            if player not in self.player_scores:
                pass
            else:
                score = self.player_scores[player]
                team_score_without -= score
        if len(self.injuries[1]) == 0:
            return team_score_with, team_score_without
        else:
            print(
                "Day to Day Players (check for most recent update): ", self.injuries[1]
            )
            return team_score_with, team_score_without

    def find_injuries(self):
        """
        return (dict): team: [[out players], [day to day players]]
        """
        time.sleep(4)
        df = pd.read_html("https://www.basketball-reference.com/friv/injuries.fcgi")
        injuries = {}
        for index, player in df[0].iterrows():
            if player["Team"] not in injuries:
                if "Day To Day" in player["Description"].split("-")[0]:
                    injuries[player["Team"]] = [[], [player["Player"]]]
                else:
                    injuries[player["Team"]] = [[player["Player"]], []]
            else:
                if "Day To Day" in player["Description"].split("-")[0]:
                    injuries[player["Team"]][1].append(player["Player"])
                else:  # player is out
                    injuries[player["Team"]][0].append(player["Player"])
        if conversion_table[self.team_abbr] in injuries:
            return injuries[conversion_table[self.team_abbr]]
        else:
            return [[], []]

    def find_offdef_ratings(self, team_abbr):
        """
        team_abbr (str) : team abbreivation (CHI, BOS, LAL)
        return (list[float]) : [off_rtg, def_rtg]
        """
        constant = 200  # for inverse relationship for defensive ratings
        url = "https://www.basketball-reference.com/teams/%s/2024.html" % team_abbr
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        tag = soup.find("div", {"id": "all_team_misc"})

        for element in tag(text=lambda text: isinstance(text, Comment)):
            soup = BeautifulSoup(element, "html.parser")

        rtgs = soup.find_all("td", {"data-stat": ["off_rtg", "def_rtg"]})
        rtgs = np.log2(
            [float(rtgs[0].get_text()), constant - float(rtgs[1].get_text())]
        )
        time.sleep(4)
        return rtgs

    def pipeline(self):
        """
        return (list): [offdef_ratings, injuries, historical_performances, fatigue, homecourt_advantage]
        """
        data = [
            sum(self.offdef_ratings),
            self.team_abbr,
            self.team_score_with,
            self.team_score_without,
        ]

        return data
