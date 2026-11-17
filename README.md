# GTA VI launch-day Twitch viewers — hour-by-hour forecast

Entry for the make.fun bounty `gta6-twitch-ccu-forecast-01`: one Python
program (`main.py`, standard library only) that forecasts Grand Theft Auto
VI's launch-day Twitch concurrent viewers for hours 2–24, as five quantiles
per hour, from the hour-1 print and the launch-day curves of past comparable
launches.

## Method

A rolling-midnight console launch has its UTC-day shape set by two things:
how big the game already is at 00:00 UTC (hour 1), and the unlock wave
rolling across timezones through the day (New Zealand first, US East at
04:00–05:00Z, US West last). Launches that unlock all at once, at one
global instant, have a completely different UTC-hour shape, so they are
left out: only comparables whose unlock is a rolling midnight are used.
(The game's own pre-release category history, listed among the comparables,
is not a launch and is left out too.)

For each forecast hour, across the rolling comparables, the log ratio
`ln(viewers_h / viewers_1)` is regressed on `ln(viewers_1)` by local-linear
(Gaussian kernel, bandwidth 2.0 in log units) least squares around the
target's own `ln(viewers_1)`. Bigger launches stay flatter; smaller ones
explode through the day. The five quantiles are the regression prediction
plus the empirical quantiles of that hour's residuals — the spread is the
comparables' own spread, not an assumed normal one.

The target's log size is clamped at the top end of the comparables' range,
so the fitted slope is never extrapolated past the data (GTA VI's hour 1
will dwarf every comparable).

No network, no randomness, deterministic: the same input bytes give the
same output bytes. Runs in ~0.03 s per case (limit 8 s).

## Practice scores (local, mean pinball loss on the log scale)

Scored against the four practice launches' real SullyGnome hourly curves
(the kit's practice answers are sealed, so the curves were read back from
the same public source the kit used; the entry itself never touches the
network):

| case | launch | loss |
| --- | --- | --- |
| p00 | Marvel's Wolverine | 0.1822 |
| p01 | Red Dead Redemption 2 | 0.1061 |
| p02 | God of War Ragnarök | 0.1044 |
| p03 | Marvel's Spider-Man 2 | 0.2172 |
| **mean** | | **0.1525** |

v2 changes (iteration 2): DST alignment of the US-East-midnight unlock wave
and a 0.85 shrinkage of the residual-quantile spread. v1 scored 0.1547
(p00 0.1822, p01 0.1169, p02 0.1074, p03 0.2125). The 0.125 gate for a
second submission was not reached: the residual variance is dominated by a
single per-launch shape factor (85.7% of it) that no observable in the case
predicts, and the model is already unbiased (leave-one-out mean bias ≈ 0.00
at every hour).

House baseline (persistence) scores 0.387 on the same cases. The live
practice board at 17:21 +07: best 0.0975, rank 20 0.125; our v1 entry scored
0.155 live (rank #327/3,825), confirming the local scorer (0.1547 local).

## How to run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory (schema
`makefun/twitch-ccu-case/v1`), writes one `*.out` per case: exactly 23 rows
(hours 2–24), each `{"hour": h, "q": [q05, q25, q50, q75, q95]}`.

## License

MIT — see `LICENSE`.
