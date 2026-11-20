# gta6-trophy-forecast-eni

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/agungvg/gta6-trophy-forecast-eni/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/agungvg/gta6-trophy-forecast-eni/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Quantile forecaster for Grand Theft Auto VI PlayStation 5 trophy earn rates on Exophase in week one. For every trophy it forecasts five quantiles (5%, 25%, 50%, 75%, 95%) of the earn rate at the case target, scored by pinball loss on the logit scale.

How it works: an earn rate is a ratio, earners over tracked players, and the two sides move for different reasons, so the program models them separately. On launch night, when the history is one flat page, it inverts the documented first refresh instead of guessing: a printed rate is an integer earner count over the tracked players, so the earners are recoverable exactly and the refresh dilutes every rate by the player ratio (638 to 3,868 players, the 100% trophy staying at 100%). Once the history moves, it fits the recent trend — logit(rate) linear in log-hours since release over the last 24 hours of snapshots, because launch night is a different regime from the days that follow — blended with an earners-over-players extrapolation (log-log and Weibull-saturating fits for each side) and a persistence stabiliser, with a spread that widens with horizon, fit residual and rarity. Standard library only, deterministic, no network, runs in well under a second.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=p9r33tb4&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
