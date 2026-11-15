# Football Match Prediction: Poisson Model vs Bookmaker

A personal project, can a model predict Premier League matches better than the bookmaker?

I built it step by step from a Poisson model to a machine learning model, testing everything on seasons the model had never seen.

## Data
Premier League results and Bet365 odds from 2015/16 to 2025/26, from
[football-data.co.uk](https://www.football-data.co.uk).

## Project structure
| File | What it does |
|------|--------------|
| `01_explore.ipynb` | Loading data, team attack/defence strengths, first Poisson predictions |
| `02_backtest.ipynb` | Honest train/test split, strengths updated during the season, Brier score and log loss |
| `03_ml.ipynb` | Logistic regression with recent form, then combined with the Poisson model |
| `04_many_seasons.ipynb` | Rolling evaluation across 9 test seasons |
| `05_beat_market.ipynb` | Line shopping, lower leagues and expected-value betting |
| `football.py` | Reusable functions: strengths, Poisson predictions, form features |

## Method
1. **Poisson model:** each team gets home/away attack and defence strengths.
   Expected goals = attack × opponent's defence × league average, then the Poisson distribution gives the probability of every scoreline, and so of a home win, draw or away win.
2. **Removing data leakage:** my first backtest showed a +22.4% ROI, but only because the strengths came from the same season being predicted. Training on the previous season instead turned it into −12.8%.
3. **Walk forward predictions:** before each match, strengths are recalculated from all earlier results only. Promoted teams use the average of the relegated teams until they have played 5 home and 5 away matches.
4. **Weighting recent matches:** the current season counts twice as much as the previous one. The weight was chosen on a validation season, not the test season.
5. **Machine learning:** logistic regression using each team's goals scored and conceded in its last 5 matches, then combined with the Poisson probabilities.
6. **Evaluation:** Brier score on every match (lower is better), plus a simple betting backtest. Each test season is predicted using only earlier seasons.

## Results (9 test seasons, 2017/18 to 2025/26)
| Model | Average Brier | Share of bookmaker's skill |
|-------|---------------|----------------------------|
| Bookmaker | 0.5673 | 100% |
| Combined (Poisson + form) | 0.5868 | ~75% |
| Poisson | 0.5882 | ~74% |
| Form only | 0.6132 | ~42% |
| Baseline (same probabilities every match) | 0.6465 | 0% |

*Share of skill = how much of the bookmaker's improvement over the baseline each model achieves.*

- The bookmaker was the best predictor in every season.
- Combining Poisson with recent form helped slightly on average (won 5 of 9 seasons), and helped more as training data grew.
- No model would have been profitable: betting where the model disagreed with the bookmaker lost money (around −10% to −20% ROI).

## Trying to beat the market (final experiment)

Using expected value (EV = probability × odds − 1) to choose bets, with the best
odds across bookmakers instead of Bet365 only, from 2020/21 to 2025/26:

| League | Bet365 odds | Best odds |
|---|---|---|
| Premier League | −9.4% | −4.9% |
| Championship | −6.9% | −2.8% |

- **Line shopping** improved ROI by about 4–5 points in both leagues.
- The **Championship** was the easiest market; League One and League Two were not
  easier than the Championship.
- **Pickier betting** (only bets with EV above 10% or 20%) did not help: the biggest
  "edges" were usually model errors.
- Best result: about −1.6% in the Championship, which is break-even at best,
  before real-world limits like account restrictions.

**Conclusion:** a model based on goals alone cannot beat bookmakers, even with the
best available odds in less efficient leagues.

## How to run
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Then open the notebooks in order. The data is downloaded automatically.

## Possible next steps
- More features like shots, Elo ratings, rest days between matches
- Testing other leagues
- Betting only on bigger disagreements, and checking whether results are statistically significant
