"""gta6-twitch-ccu-forecast-01: rolling-midnight log-ratio forecaster.

Forecasts a launch day's Twitch concurrent viewers, hour by hour (hours 2..24),
from the hour-1 print and the launch-day curves of past comparable launches.

Method
------
The UTC release day of a rolling-midnight console launch has a shape set by
two things: how big the game already is at 00:00 UTC (hour 1), and the unlock
wave rolling across timezones through the day (New Zealand first, US East at
04:00-05:00Z, US West last). Launches that unlock all at once, at one global
instant, have a completely different UTC-hour shape, so they are left out:
only comparables whose unlock is a rolling midnight are used, plus the
launch-day curve of the game's own category is never a launch and is left out.

For each forecast hour h, across the rolling comparables, regress the log
ratio ln(viewers_h / viewers_1) on ln(viewers_1) by local-linear (Gaussian
kernel, bandwidth 2.0 in log units) least squares around the target's own
ln(viewers_1). Bigger launches stay flatter; smaller ones explode through
the day, and the regression captures that. The five quantiles are the
regression prediction plus the empirical quantiles of that hour's residuals,
so the spread is the comparables' own spread, not an assumed normal one.

The target's log size is clamped to the comparables' range at the top end
only: outside it the shape is held at the biggest comparable's instead of
extrapolating a slope past the data (GTA VI's hour 1 will dwarf every
comparable).

Standard library only, no network, deterministic: the same input bytes give
the same output bytes. Runs in well under a second per case.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim. Outside the grader's container, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The spread of an hour's move for the persistence fallback, in natural-log
# units per square-root hour (the house baseline's value).
FALLBACK_SIGMA = 0.15
FALLBACK_Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Kernel bandwidth of the local-linear fit, in natural-log units of hour-1
# viewers. Wide enough to pool a dozen comparables, narrow enough that the
# size effect (small launches explode, big ones stay flat) stays local.
BANDWIDTH = 2.0


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def is_launch(comp: dict) -> bool:
    """False for the target game's own pre-release category history, which is
    listed among the comparables but is not a launch at all."""
    unlock = str(comp.get("unlock", ""))
    return "Not a launch" not in unlock and not comp.get("name", "").startswith(
        "Grand Theft Auto VI category"
    )


def is_rolling_midnight(comp: dict) -> bool:
    """True when the comparable unlocked at midnight local time, rolling
    across timezones — the same unlock pattern as the target. A launch that
    unlocked at one simultaneous global instant has a different UTC-hour
    shape and is left out."""
    u = str(comp.get("unlock", "")).lower()
    return ("midnight local" in u or "rolling" in u) and (
        "single simultaneous global" not in u and "single global unlock" not in u
    )


def launch_day_curve(comp: dict) -> dict:
    """Hourly viewers on the comparable's UTC release day, indexed 1..24."""
    day = str(comp.get("day_start", ""))[:10]
    out = {}
    for p in comp.get("hourly", []):
        if str(p.get("from", ""))[:10] == day:
            v = p.get("viewers")
            if v is not None and v > 0:
                out[int(str(p["from"])[11:13]) + 1] = v
    return out


def wls_intercept_slope(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, residuals)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, [y - my for y in ys])
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    return (a, b, [y - (a + b * x) for x, y in zip(xs, ys)])


def quantile(sorted_vals, tau: float) -> float:
    """Linear-interpolation quantile of an already sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    lx = math.log(p1)

    # The rolling-midnight comparables, each with its launch-day curve.
    curves = []
    for comp in case.get("comparables", []):
        if not is_launch(comp) or not is_rolling_midnight(comp):
            continue
        curve = launch_day_curve(comp)
        if 1 in curve and curve[1] > 0:
            curves.append(curve)

    # Clamp the target's log size to the comparables' range at the top end
    # only: never extrapolate the fitted slope past the biggest comparable
    # (GTA VI's hour 1 will dwarf every one of them). The bottom end is left
    # alone — a target just below the smallest comparable extrapolates a
    # short, harmless distance along the same trend.
    if curves:
        hi_x = max(math.log(c[1]) for c in curves)
        lx_fit = min(lx, hi_x)
    else:
        lx_fit = lx

    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = None
        if len(curves) >= 5:
            xs, ys, ws = [], [], []
            for curve in curves:
                if h in curve:
                    x = math.log(curve[1])
                    xs.append(x)
                    ys.append(math.log(curve[h] / curve[1]))
                    ws.append(math.exp(-0.5 * ((x - lx_fit) / BANDWIDTH) ** 2))
            fit = wls_intercept_slope(xs, ys, ws)
            if fit is not None and len(xs) >= 5:
                a, b, resid = fit
                pred = a + b * lx_fit
                sresid = sorted(resid)
                qs = [pred + quantile(sresid, tau) for tau in LEVELS]
        if qs is None:
            # Too few comparables for this hour: fall back to persistence.
            qs = [z * FALLBACK_SIGMA * math.sqrt(h - 1) for z in FALLBACK_Z]
            out_q = [p1 * math.exp(q) for q in qs]
        else:
            out_q = [p1 * math.exp(q) for q in qs]
        # Valid by construction: finite, > 0, <= 1e9, non-decreasing.
        out_q = [min(max(q, 1e-6), 1e9) for q in out_q]
        out_q = sorted(out_q)
        rows.append({"hour": h, "q": [round2(q) for q in out_q]})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
