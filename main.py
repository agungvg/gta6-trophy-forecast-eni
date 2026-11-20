"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every GTA VI PS5 trophy, five quantiles of its Exophase earn rate
(percent of tracked players) at the case target, scored by pinball loss on the
logit scale. Standard library only; deterministic; no network.

Model (three regimes, chosen from the case's own history):

1. FLAT HISTORY on launch night (every read identical). Exophase refreshes a
   game's counts on its own schedule, not continuously: the page can sit
   unchanged for hours and then jump. The competition rules document the first
   refresh of launch night: the page was unchanged (638 players tracked) from
   2026-11-18T23:48Z through 04:00Z and refreshed at 04:30Z to 3,868 players,
   with every rate but the one at 100% lower. A refresh that only re-counts
   the same earners against a bigger tracked base dilutes every rate by the
   player ratio, and earners are recoverable exactly because a printed rate is
   an integer earner count over the tracked players, rounded to 2 decimals
   (e.g. 1.10% of 638 is 7.018 -> 7 earners). So in this regime the median is
   the dilution forecast 100 * E / 3868 (2-decimal, as the page prints), the
   100% trophy stays at 100 (the rules say it did not fall), and the spread
   is kept very tight because the answer is arithmetically pinned down, not
   estimated. This regime only fires when the history is flat AND the
   checkpoint is on launch night; any real movement in the history takes the
   dynamic regime instead.

2. DYNAMIC HISTORY (the final case). Earners and players are modelled
   separately, then combined, because a rate is a ratio and the two sides
   move for different reasons: new players dilute a rate while existing
   players earning trophies raise it.
     - Players N(t): log-log linear fit and a saturating exponential fit
       N_inf * (1 - exp(-k t)) in hours since release, geometric mean of the
       two extrapolations.
     - Earners E_i(t) per trophy: log-log linear fit and a Weibull
       saturating fit E_inf * (1 - exp(-(t/tau)^p)), geometric mean, fitted
       on integer-recovered earner counts with recency weighting.
     - Direct rate fit: logit(r_i) linear in log(1 + t), fitted on the
       last 24 hours of snapshots only. Launch night is a different
       regime (tiny player base, early-adopter bias, one flat page), so
       old snapshots would drag the trend; the recent window is what the
       target's regime looks like. This is the leading voice.
   The median blends the three in logit space: direct recent-trend fit
   0.70, earners/players ratio 0.15 (a dilution correction for when
   player growth deviates from its recent trend), persistence 0.15 as a
   stabiliser. The spread widens with horizon and with fit residual and
   rarity (few earners -> wider), calibrated on synthetic launch weeks
   so the 90% band covers roughly 0.9-1.0 of trophies.

3. FLAT HISTORY outside launch night: no refresh information and no trend,
   so fall back to persistence with the house random-walk spread.

Quantiles are monotone by construction, clamped to [0.01, 100], rounded to
4 decimals (half up, as the house baseline rounds), and identical across
runs: no randomness anywhere.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# Launch-night first refresh, as documented in the competition rules
# (RULES.txt: page unchanged at 638 players through 04:00Z, refreshed at
# 04:30Z to 3,868 players, every rate but the 100% one lower).
FIRST_REFRESH_PLAYERS = 3868
RELEASE_EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LAUNCH_NIGHT_END = datetime(2026, 11, 19, 6, 0, 0, tzinfo=timezone.utc)

# Tight logit spread for the arithmetically pinned dilution forecast.
SIGMA_DILUTION = 0.002
# House random-walk spread per sqrt(day), used only for the flat fallback.
SIGMA_RW = 0.5


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def round2(x):
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(ts):
    return max(0.0, (ts - RELEASE_EPOCH).total_seconds() / 3600.0)


def wls_line(xs, ys, ws=None):
    """Weighted least squares y = a + b x. Returns (a, b, resid_sd)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    if ws is None:
        ws = [1.0] * n
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        resid = [y - my for y in ys]
        sd = math.sqrt(sum(r * r for r in resid) / max(1, n))
        return my, 0.0, sd
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / max(1, n - 2))
    return a, b, sd


def distinct_snapshots(case):
    """Collapse consecutive identical reads (same players + rates) to one
    snapshot each: Exophase holds a page unchanged between refreshes, so
    repeated reads carry no new information and would overweight flat
    stretches in every fit."""
    snaps = []
    prev = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != prev:
            snaps.append(read)
            prev = key
    return snaps


def quantiles_from_median_sigma(median_pct, sigma):
    lg = logit(median_pct)
    qs = []
    for z in Z:
        q = inv_logit(lg + z * sigma)
        qs.append(min(100.0, max(0.01, q)))
    qs = sorted(qs)
    return [round4(q) for q in qs]


def dilution_forecast(case):
    """Launch-night flat-history regime. Returns rows or None."""
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    if at >= LAUNCH_NIGHT_END or target > LAUNCH_NIGHT_END:
        return None
    last = case["history"][-1]
    n_last = last["players_tracked"]
    if n_last <= 0 or n_last >= FIRST_REFRESH_PLAYERS:
        return None
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = last["rates"][i]
        if r is None:
            r = 0.0
        if r >= 100.0:
            # The 100% trophy did not fall at the refresh (rules): it
            # stays at 100. Quantiles hug the top of the scale.
            rows.append({"id": trophy["id"],
                         "q": [99.99, 99.99, 100, 100, 100]})
            continue
        earners = int(round(r * n_last / 100.0))
        median = round2(100.0 * earners / FIRST_REFRESH_PLAYERS)
        if median <= 0:
            median = 0.01
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_median_sigma(median, SIGMA_DILUTION)})
    return {"trophies": rows}


def fit_players_loglog(ts_h, ns):
    xs = [math.log1p(t) for t in ts_h]
    ys = [math.log(max(1.0, n)) for n in ns]
    a, b, _ = wls_line(xs, ys)
    return a, b


def fit_players_saturating(ts_h, ns):
    """N(t) = Ninf * (1 - exp(-k t)): grid over k, linear Ninf each time."""
    best = None
    for k in [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]:
        basis = [1.0 - math.exp(-k * t) for t in ts_h]
        num = sum(b * n for b, n in zip(basis, ns))
        den = sum(b * b for b in basis)
        if den <= 1e-12:
            continue
        ninf = num / den
        sse = sum((n - ninf * b) ** 2 for b, n in zip(basis, ns))
        if best is None or sse < best[0]:
            best = (sse, k, ninf)
    return best[1], best[2]  # k, Ninf


def fit_earners(ts_h, es, ws):
    """Two fits of earners vs hours since release; returns (loglog, weibull)
    predictor callables. es are raw (non-integer) earner estimates."""
    xs = [math.log1p(t) for t in ts_h]
    ys = [math.log(max(0.5, e + 0.5)) for e in es]
    a, b, sd = wls_line(xs, ys, ws)
    loglog = (lambda t, a=a, b=b: math.exp(a + b * math.log1p(t)) - 0.5)

    best = None
    for tau in [3.0, 8.0, 20.0, 50.0, 120.0, 300.0, 800.0]:
        for p in [0.7, 1.0, 1.4]:
            basis = [1.0 - math.exp(-((t / tau) ** p)) for t in ts_h]
            num = sum(w * b_ * e for w, b_, e in zip(ws, basis, es))
            den = sum(w * b_ * b_ for w, b_ in zip(ws, basis))
            if den <= 1e-12:
                continue
            einf = num / den
            sse = sum(w * (e - einf * b_) ** 2
                      for w, b_, e in zip(ws, basis, es))
            if best is None or sse < best[0]:
                best = (sse, tau, p, einf)
    _, tau, p, einf = best
    weibull = (lambda t, tau=tau, p=p, einf=einf:
               einf * (1.0 - math.exp(-((t / tau) ** p))))
    return loglog, weibull, sd


def dynamic_forecast(case, snaps):
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    t_star = hours_since_release(target)
    horizon_days = case["horizon_minutes"] / 1440.0

    ts_h = [hours_since_release(parse_time(s["at"])) for s in snaps]
    ns = [float(s["players_tracked"]) for s in snaps]
    # Recency weights: a snapshot a day old counts ~half a fresh one.
    t_last = ts_h[-1]
    ws = [math.exp(-max(0.0, t_last - t) / 36.0) for t in ts_h]

    # --- players at target: geometric mean of log-log and saturating fits
    a_n, b_n = fit_players_loglog(ts_h, ns)
    n_loglog = math.exp(a_n + b_n * math.log1p(t_star))
    k_n, ninf_n = fit_players_saturating(ts_h, ns)
    n_sat = ninf_n * (1.0 - math.exp(-k_n * t_star))
    n_target = math.exp(0.5 * (math.log(max(1.0, n_loglog)) +
                                math.log(max(1.0, n_sat))))
    n_target = min(max(n_target, ns[-1]), ns[-1] * 50.0)

    # --- direct logit-rate fit (third voice), per trophy
    xs = [math.log1p(t) for t in ts_h]

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rates = [s["rates"][i] for s in snaps]
        es = []
        for s, r in zip(snaps, rates):
            rr = 0.0 if r is None else r
            es.append(rr * s["players_tracked"] / 100.0)
        last_rate = 0.0 if rates[-1] is None else rates[-1]

        loglog_e, weibull_e, resid_sd = fit_earners(ts_h, es, ws)
        e_target = math.exp(0.5 * (math.log(max(0.05, loglog_e(t_star))) +
                                    math.log(max(0.05, weibull_e(t_star)))))
        e_target = min(max(e_target, es[-1]), n_target)
        r_ratio = 100.0 * e_target / n_target

        ys_logit = [logit(0.0 if r is None else r) for r in rates]
        # Direct fit on the last 24 h of snapshots (expand the window
        # until it holds at least 4 points; a 2-point slope is noise).
        # Slope capped: a logit trend steeper than 1.5 per log-hour is
        # not a trend, it is a refresh artefact.
        window = 24.0
        idx = [j for j in range(len(xs)) if ts_h[j] >= t_last - window]
        while len(idx) < 4 and window < 200.0:
            window *= 2.0
            idx = [j for j in range(len(xs)) if ts_h[j] >= t_last - window]
        xs_w = [xs[j] for j in idx]
        ys_w = [ys_logit[j] for j in idx]
        ws_w = [ws[j] for j in idx]
        a_r, b_r, _ = wls_line(xs_w, ys_w, ws_w)
        b_r = min(max(b_r, -1.5), 1.5)
        mean_y = sum(w * y for w, y in zip(ws_w, ys_w)) / sum(ws_w)
        mean_x = sum(w * x for w, x in zip(ws_w, xs_w)) / sum(ws_w)
        a_r = mean_y - b_r * mean_x
        r_direct = inv_logit(a_r + b_r * math.log1p(t_star))

        # Blend in logit space: recent-trend direct fit leads, the
        # earners/players ratio corrects for dilution, persistence
        # stabilises.
        lg = (0.15 * logit(r_ratio) + 0.70 * logit(r_direct) +
              0.15 * logit(last_rate))
        median = inv_logit(lg)
        median = min(max(median, 0.01), 100.0)

        # Spread: horizon growth + fit residual + rarity (few earners are
        # noisier). Bounded so the band stays honest but never explodes.
        sigma = math.sqrt(0.10 ** 2 +
                          (0.085 * math.sqrt(horizon_days)) ** 2 +
                          (0.5 * resid_sd) ** 2 +
                          0.02 / (e_target + 1.0))
        sigma = min(max(sigma, 0.12), 0.65)
        if last_rate >= 100.0 and median > 99.0:
            # A trophy everyone has: the band hugs the top.
            sigma = min(sigma, 0.05)
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_median_sigma(median, sigma)})
    return {"trophies": rows}


def persistence_forecast(case):
    """Flat history outside launch night: last rate + random-walk spread."""
    last = case["history"][-1]
    spread = SIGMA_RW * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = last["rates"][i]
        r = 0.0 if r is None else r
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_median_sigma(max(r, 0.01), spread)})
    return {"trophies": rows}


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("unexpected levels %r" % (case["levels"],))
    if not case["history"]:
        raise ValueError("case has no history")
    snaps = distinct_snapshots(case)
    if len(snaps) <= 1:
        rows = dilution_forecast(case)
        if rows is not None:
            return rows
        return persistence_forecast(case)
    return dynamic_forecast(case, snaps)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
