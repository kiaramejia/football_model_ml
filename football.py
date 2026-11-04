import pandas as pd
import numpy as np
from scipy.stats import poisson


def compute_strengths(data):
    home_avg = data["FTHG"].mean()
    away_avg = data["FTAG"].mean()

    home_attack = data.groupby("HomeTeam")["FTHG"].mean() / home_avg
    home_defence = data.groupby("HomeTeam")["FTAG"].mean() / away_avg
    away_attack = data.groupby("AwayTeam")["FTAG"].mean() / away_avg
    away_defence = data.groupby("AwayTeam")["FTHG"].mean() / home_avg

    strengths = pd.DataFrame({
        "home_attack": home_attack,
        "home_defence": home_defence,
        "away_attack": away_attack,
        "away_defence": away_defence,
    })
    return strengths, home_avg, away_avg


def predict_with(home, away, s, h_avg, a_avg):
    home_xg = s.loc[home, "home_attack"] * s.loc[away, "away_defence"] * h_avg
    away_xg = s.loc[away, "away_attack"] * s.loc[home, "home_defence"] * a_avg

    home_win = 0
    draw = 0
    away_win = 0
    for h in range(10):
        for a in range(10):
            p = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            if h > a:
                home_win += p
            elif h == a:
                draw += p
            else:
                away_win += p
    return home_win, draw, away_win


def poisson_walk_forward(old, new, w):
    old_teams = set(old["HomeTeam"])
    new_teams = set(new["HomeTeam"])
    prom = list(new_teams - old_teams)
    rel = list(old_teams - new_teams)

    old_strengths, _, _ = compute_strengths(old)
    prom_strength = old_strengths.loc[rel].mean()

    rows = []
    for i in range(len(new)):
        date = new["Date"].iloc[i]
        home = new["HomeTeam"].iloc[i]
        away = new["AwayTeam"].iloc[i]

        before = new[new["Date"] < date]
        history = pd.concat([old] + [before] * w)
        s, h_avg, a_avg = compute_strengths(history)

        for team in prom:
            if (before["HomeTeam"] == team).sum() < 5 or (before["AwayTeam"] == team).sum() < 5:
                s.loc[team] = prom_strength

        hw, d, aw = predict_with(home, away, s, h_avg, a_avg)
        rows.append({"Date": date, "HomeTeam": home, "AwayTeam": away,
                     "pois_H": hw, "pois_D": d, "pois_A": aw})

    return pd.DataFrame(rows)


def add_form_features(data, n=5):
    scored = {}
    conceded = {}
    h_scored, h_conceded, a_scored, a_conceded = [], [], [], []

    for i in range(len(data)):
        home = data["HomeTeam"].iloc[i]
        away = data["AwayTeam"].iloc[i]
        hg = data["FTHG"].iloc[i]
        ag = data["FTAG"].iloc[i]

        for team in [home, away]:
            if team not in scored:
                scored[team] = []
                conceded[team] = []

        if len(scored[home]) >= n and len(scored[away]) >= n:
            h_scored.append(np.mean(scored[home][-n:]))
            h_conceded.append(np.mean(conceded[home][-n:]))
            a_scored.append(np.mean(scored[away][-n:]))
            a_conceded.append(np.mean(conceded[away][-n:]))
        else:
            h_scored.append(np.nan)
            h_conceded.append(np.nan)
            a_scored.append(np.nan)
            a_conceded.append(np.nan)

        scored[home].append(hg)
        conceded[home].append(ag)
        scored[away].append(ag)
        conceded[away].append(hg)

    data = data.copy()
    data["h_scored"] = h_scored
    data["h_conceded"] = h_conceded
    data["a_scored"] = a_scored
    data["a_conceded"] = a_conceded
    return data