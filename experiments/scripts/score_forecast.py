"""Честный WAPE-score: правило выбирается по августу, сентябрь–октябрь считаются один раз."""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "kaggle_mstrans"
ENR = ROOT / "data" / "kaggle_enrichment"
OUT = LABELS / "submission.csv"

ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
START = date(2025, 1, 1)


def daterange(start, end):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def load_labels():
    observed = {}
    for name in ("labels_day_train.csv", "labels_day_test.csv"):
        with (LABELS / name).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter=";"):
                key = (int(row["route"]), date.fromisoformat(row["date"]), int(row["hour"]))
                observed[key] = float(row["boardings"])
    return observed


def load_calendar():
    flags = {}
    with (ENR / "holidays_2025.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            flags[date.fromisoformat(row["date"])] = {
                "day_off": int(row["day_off"]),
                "holiday": row["holiday"],
            }
    return flags


def grid(observed, start, end):
    values = {}
    for route in ROUTES:
        for day in daterange(start, end):
            for hour in range(24):
                values[(route, day, hour)] = observed.get((route, day, hour), 0.0)
    return values


def slot_history(history, route, day, hour, k, cutoff):
    values = []
    cursor = day - timedelta(days=7)
    while len(values) < k and cursor >= START:
        if cursor <= cutoff:
            values.append(history.get((route, cursor, hour), 0.0))
        cursor -= timedelta(days=7)
    return values


def aggregate(values, how):
    if not values:
        return 0.0
    if how == "median":
        return float(median(values))
    if how == "mean":
        return float(mean(values))
    weight_sum = 0.0
    total = 0.0
    for index, value in enumerate(values):
        weight = 0.75 ** index
        total += weight * value
        weight_sum += weight
    return total / weight_sum


def route_level(history, route, end, days):
    total = 0.0
    for shift in range(days):
        day = end - timedelta(days=shift)
        if day < START:
            break
        for hour in range(24):
            total += history.get((route, day, hour), 0.0)
    return total


def trend_ratio(history, route, cutoff):
    recent = route_level(history, route, cutoff, 14)
    previous = route_level(history, route, cutoff - timedelta(days=14), 14)
    if previous <= 0:
        return 1.0
    return min(1.12, max(0.88, recent / previous))


def holiday_level(history, calendar, route, day, hour, k, cutoff):
    values = []
    cursor = day - timedelta(days=1)
    while len(values) < k and cursor >= START:
        if cursor <= cutoff and calendar.get(cursor, {}).get("holiday") not in (None, "none"):
            values.append(history.get((route, cursor, hour), 0.0))
        cursor -= timedelta(days=1)
    return values


def trend_multiplier(ratio, day, cutoff, mode):
    if mode == "none":
        return 1.0
    if mode == "flat":
        return ratio
    age = (day - cutoff).days
    if age <= 21:
        return ratio
    if age >= 45:
        return 1.0
    weight = (45 - age) / 24
    return 1.0 + (ratio - 1.0) * weight


def predict_range(history, calendar, start, end, k, how, trend_mode, ratios, cutoff):
    forecast = {}
    for route in ROUTES:
        ratio = ratios[route]
        for day in daterange(start, end):
            info = calendar.get(day, {})
            special = info.get("day_off") and info.get("holiday") not in (None, "none")
            scale = trend_multiplier(ratio, day, cutoff, trend_mode)
            for hour in range(24):
                weekly = slot_history(history, route, day, hour, k, cutoff)
                if special:
                    pool = holiday_level(history, calendar, route, day, hour, max(k, 6), cutoff)
                    base = aggregate(pool, how) if len(pool) >= 2 else aggregate(weekly, how)
                else:
                    base = aggregate(weekly, how)
                forecast[(route, day, hour)] = max(0.0, base * scale)
    return forecast


def wape(actual, forecast, start, end):
    absolute = 0.0
    denom = 0.0
    for route in ROUTES:
        for day in daterange(start, end):
            for hour in range(24):
                y = actual.get((route, day, hour), 0.0)
                yhat = forecast[(route, day, hour)]
                absolute += abs(y - yhat)
                denom += y
    score = max(0.0, 1.0 - absolute / denom)
    return absolute / denom, score


def points(score):
    if score <= 0.48:
        return 0, 0
    if score <= 0.60:
        return 1, 2
    if score <= 0.70:
        return 2, 4
    if score <= 0.80:
        return 3, 6
    if score <= 0.88:
        return 4, 8
    return 5, 10


def ratios_at(history, cutoff):
    return {route: trend_ratio(history, route, cutoff) for route in ROUTES}


def choose(history, calendar, actual, start, end, ratios, cutoff):
    best = None
    for k in (4, 6, 8, 12):
        for how in ("median", "mean", "ewm"):
            for trend_mode in ("none", "flat", "decay"):
                forecast = predict_range(history, calendar, start, end, k, how, trend_mode, ratios, cutoff)
                error, score = wape(actual, forecast, start, end)
                row = (score, error, k, how, trend_mode)
                if best is None or row[0] > best[0]:
                    best = row
    return best


def month_levels(actual, calendar, start, end):
    totals = {}
    days = {}
    for day in daterange(start, end):
        if calendar.get(day, {}).get("holiday") not in (None, "none"):
            continue
        bucket = day.strftime("%Y-%m")
        for route in ROUTES:
            for hour in range(24):
                totals[bucket] = totals.get(bucket, 0.0) + actual.get((route, day, hour), 0.0)
        days[bucket] = days.get(bucket, 0) + 1
    for bucket in sorted(totals):
        print(f"level {bucket} daily {totals[bucket] / days[bucket]:.0f}")


def main():
    observed = load_labels()
    calendar = load_calendar()
    actual_spring = grid(observed, START, date(2025, 5, 31))
    month_levels(actual_spring, calendar, START, date(2025, 5, 31))

    select_cutoff = date(2025, 2, 7)
    history_select = grid(observed, START, select_cutoff)
    actual_select = grid(observed, date(2025, 2, 8), date(2025, 3, 31))
    best = choose(
        history_select, calendar, actual_select,
        date(2025, 2, 8), date(2025, 3, 31),
        ratios_at(history_select, select_cutoff), select_cutoff,
    )
    score, error, k, how, trend_mode = best
    print(f"feb-mar choice k={k} agg={how} trend={trend_mode} WAPE={error:.4f} score={score:.4f}")

    history_mar = grid(observed, START, date(2025, 3, 31))
    actual_holdout = grid(observed, date(2025, 4, 1), date(2025, 5, 31))
    holdout = predict_range(
        history_mar, calendar, date(2025, 4, 1), date(2025, 5, 31),
        k, how, trend_mode, ratios_at(history_mar, date(2025, 3, 31)), date(2025, 3, 31),
    )
    hold_error, hold_score = wape(actual_holdout, holdout, date(2025, 4, 1), date(2025, 5, 31))
    raw, weighted = points(hold_score)
    print(f"apr-may WAPE={hold_error:.4f} score={hold_score:.4f} points={raw} weighted={weighted}")
    for label, start, end in (("april", date(2025, 4, 1), date(2025, 4, 30)), ("may", date(2025, 5, 1), date(2025, 5, 31))):
        part_error, part_score = wape(actual_holdout, holdout, start, end)
        print(f"  {label} WAPE={part_error:.4f} score={part_score:.4f}")

    history_all = grid(observed, START, date(2025, 10, 31))
    final_ratios = ratios_at(history_all, date(2025, 10, 31))
    final = predict_range(
        history_all, calendar, date(2025, 11, 1), date(2025, 12, 31),
        k, how, trend_mode, final_ratios, date(2025, 10, 31),
    )
    template = LABELS / "test_submission.csv"
    with template.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    missing = 0
    for row in rows:
        key = (int(row["route"]), date.fromisoformat(row["date"]), int(row["hour"]))
        if key not in final:
            missing += 1
            row["prediction"] = "0"
        else:
            row["prediction"] = str(int(round(final[key])))
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["route", "date", "hour", "prediction"], delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"submission {OUT} rows={len(rows)} missing={missing}")


if __name__ == "__main__":
    main()
