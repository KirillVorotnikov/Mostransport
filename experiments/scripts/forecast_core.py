"""Cutoff-safe календарь, профили и метрики WAPE.

Маршруты, даты прогноза, горизонт и точки отсчёта backtest выводятся из
файлов меток и шаблона submission. Школьные каникулы читаются из
school_calendar.csv, часы пик определяются по будним дням до cutoff.
Параметры champion-профиля (4 недели, вес 0.75, 6 праздников) описывают
замороженный файл 0.87560 и нужны только для его воспроизведения.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "kaggle_mstrans"
ENR = ROOT / "data" / "kaggle_enrichment"
RESULTS = ROOT / "experiments" / "results"
CHAMPION_SRC = LABELS / "submission.csv"
CHAMPION_COPY = RESULTS / "champion-087560.csv"
LABEL_FILES = ("labels_day_train.csv", "labels_day_test.csv")
TEMPLATE = LABELS / "test_submission.csv"
PEAK_WIDTH = 3
DISTANCE_CAP = 21


def _scan_inputs():
    with TEMPLATE.open(encoding="utf-8", newline="") as handle:
        template = list(csv.DictReader(handle, delimiter=";"))
    routes = sorted({int(row["route"]) for row in template})
    forecast_days = sorted({date.fromisoformat(row["date"]) for row in template})
    label_days = set()
    for name in LABEL_FILES:
        with (LABELS / name).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter=";"):
                label_days.add(row["date"])
    label_days = sorted(date.fromisoformat(day) for day in label_days)
    return routes, label_days[0], label_days[-1], forecast_days[0], forecast_days[-1], len(template)


def _month_ends(first, last):
    out = []
    day = first
    while day <= last:
        if (day + timedelta(days=1)).day == 1:
            out.append(day)
        day += timedelta(days=1)
    return out


ROUTES, START, OBSERVED_END, FORECAST_START, FORECAST_END, SUBMISSION_ROWS = _scan_inputs()
END = FORECAST_END
HORIZON = (FORECAST_END - FORECAST_START).days + 1
# Заблокированное окно повторяет боевую задачу: тот же горизонт, закрывающийся на последней метке.
LOCK_ORIGIN = OBSERVED_END - timedelta(days=HORIZON)
DEV_ORIGINS = tuple(_month_ends(START + timedelta(days=HORIZON), LOCK_ORIGIN - timedelta(days=1)))
# Осенние проверочные окна короче горизонта и в подборе не участвуют.
CHECK_ORIGINS = tuple(
    LOCK_ORIGIN + timedelta(days=step)
    for step in range(14, (OBSERVED_END - LOCK_ORIGIN).days - 13, 14)
)


def _load_school_calendar():
    ranges = {"quarter": [], "module": [], "summer": []}
    with (ENR / "school_calendar.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            ranges[row["system"]].append((date.fromisoformat(row["start"]), date.fromisoformat(row["end"])))
    return tuple(ranges["quarter"]), tuple(ranges["module"]), tuple(ranges["summer"])


QUARTER_BREAKS, MODULE_BREAKS, SUMMER_BREAKS = _load_school_calendar()
ALL_BREAKS = QUARTER_BREAKS + MODULE_BREAKS + SUMMER_BREAKS


@dataclass(frozen=True)
class ProfileParams:
    """Параметры сезонного профиля: уровень по тем же дням недели, форма с пулом будней."""

    weeks: int = 4
    decay: float = 0.75
    holidays: int = 6
    shape_pool: float = 0.0
    shape_weeks: int = 4
    shape_decay: float = 1.0


CHAMPION_PARAMS = ProfileParams()


def daterange(start, end):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def in_ranges(day, ranges):
    return any(start <= day <= end for start, end in ranges)


def school_calendar(day):
    summer = int(in_ranges(day, SUMMER_BREAKS))
    quarter = int(summer or in_ranges(day, QUARTER_BREAKS))
    module = int(summer or in_ranges(day, MODULE_BREAKS))
    share = (quarter + module) / 2.0
    return quarter, module, summer, share


def detect_peaks(series, days, cutoff_pos, weekend):
    """Утренний и вечерний пик: окна по PEAK_WIDTH часов с максимумом будничного объёма."""
    workdays = [pos for pos in range(cutoff_pos + 1) if not weekend[pos]]
    profile = series[:, workdays].sum(axis=(0, 1))
    windows = np.array([profile[h: h + PEAK_WIDTH].sum() for h in range(24 - PEAK_WIDTH + 1)])
    noon = 12
    morning_start = int(np.argmax(windows[: noon - PEAK_WIDTH + 1]))
    evening_start = noon + int(np.argmax(windows[noon:]))
    return (
        frozenset(range(morning_start, morning_start + PEAK_WIDTH)),
        frozenset(range(evening_start, evening_start + PEAK_WIDTH)),
    )


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_champion():
    digest = sha256_file(CHAMPION_SRC)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not CHAMPION_COPY.exists():
        CHAMPION_COPY.write_bytes(CHAMPION_SRC.read_bytes())
    if sha256_file(CHAMPION_COPY) != digest:
        raise RuntimeError("копия champion не совпадает с исходным файлом")
    return digest


def load_bundle():
    days = list(daterange(START, END))
    index = {day: pos for pos, day in enumerate(days)}
    series = np.zeros((len(ROUTES), len(days), 24), np.float32)
    route_pos = {route: pos for pos, route in enumerate(ROUTES)}
    for name in LABEL_FILES:
        with (LABELS / name).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter=";"):
                day = date.fromisoformat(row["date"])
                pos = index.get(day)
                row_i = route_pos.get(int(row["route"]))
                if pos is None or row_i is None or day > OBSERVED_END:
                    continue
                series[row_i, pos, int(row["hour"])] = float(row["boardings"])

    holiday_name = []
    day_off = np.zeros(len(days), np.int8)
    weekend = np.zeros(len(days), np.int8)
    preholiday = np.zeros(len(days), np.int8)
    short_day = np.zeros(len(days), np.int8)
    is_holiday = np.zeros(len(days), np.int8)
    holiday_id_map = {}
    with (ENR / "holidays_2025.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = sorted({row["holiday"] for row in rows if row["holiday"] not in ("", "none")})
    holiday_id_map = {name: pos + 1 for pos, name in enumerate(names)}
    by_day = {date.fromisoformat(row["date"]): row for row in rows}
    for pos, day in enumerate(days):
        row = by_day.get(day)
        if row is None:
            holiday_name.append("none")
            weekend[pos] = int(day.weekday() >= 5)
            day_off[pos] = weekend[pos]
            continue
        name = row["holiday"] or "none"
        holiday_name.append(name)
        day_off[pos] = int(row["day_off"])
        weekend[pos] = int(row["weekend"])
        preholiday[pos] = int(row["preholiday"])
        short_day[pos] = int(row["short_day"])
        is_holiday[pos] = int(name not in ("none", ""))

    weather = {key: np.zeros((len(ROUTES), len(days), 24), np.float32) for key in ("temp", "precip", "wind", "snowfall")}
    weather_code = np.zeros((len(ROUTES), len(days), 24), np.int16)
    seen_weather = np.zeros((len(ROUTES), len(days), 24), np.int8)
    with (ENR / "weather_hourly.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            day = date.fromisoformat(row["date"])
            pos = index.get(day)
            route = route_pos.get(int(row["route"]))
            if pos is None or route is None:
                continue
            hour = int(row["hour"])
            weather["temp"][route, pos, hour] = float(row["temp"])
            weather["precip"][route, pos, hour] = float(row["precip"])
            weather["wind"][route, pos, hour] = float(row["wind"])
            weather["snowfall"][route, pos, hour] = float(row["snowfall"])
            weather_code[route, pos, hour] = int(float(row["code"]))
            seen_weather[route, pos, hour] = 1

    office = np.zeros((len(ROUTES), len(days), 4), np.float32)
    with (ENR / "route_day.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            day = date.fromisoformat(row["date"])
            pos = index.get(day)
            route = route_pos.get(int(row["route"]))
            if pos is None or route is None:
                continue
            office[route, pos] = [float(row["offices"]), float(row["acc7"]), float(row["acc30"]), float(row["tram7"])]

    daily = series.sum(axis=2)
    calendar = calendar_table(days, holiday_name, day_off, weekend, preholiday, short_day, is_holiday, holiday_id_map)
    # Пики ищутся до первой точки backtest, чтобы ни одно окно не видело своё будущее.
    morning, evening = detect_peaks(series, days, index[DEV_ORIGINS[0]], day_off | weekend | is_holiday)
    return {
        "morning_peak": morning,
        "evening_peak": evening,
        "days": days,
        "index": index,
        "series": series,
        "daily": daily,
        "holiday_name": holiday_name,
        "holiday_id_map": holiday_id_map,
        "day_off": day_off,
        "weekend": weekend,
        "preholiday": preholiday,
        "short_day": short_day,
        "is_holiday": is_holiday,
        "weather": weather,
        "weather_code": weather_code,
        "seen_weather": seen_weather,
        "office": office,
        "calendar": calendar,
    }


def calendar_table(days, holiday_name, day_off, weekend, preholiday, short_day, is_holiday, holiday_id_map):
    count = len(days)
    quarter = np.zeros(count, np.float32)
    module = np.zeros(count, np.float32)
    summer = np.zeros(count, np.float32)
    share = np.zeros(count, np.float32)
    holiday_id = np.zeros(count, np.float32)
    days_to_break = np.full(count, DISTANCE_CAP, np.float32)
    days_from_break = np.full(count, DISTANCE_CAP, np.float32)
    first_school_week = np.zeros(count, np.float32)
    days_to_holiday = np.full(count, DISTANCE_CAP, np.float32)
    days_from_holiday = np.full(count, DISTANCE_CAP, np.float32)
    for pos, day in enumerate(days):
        q, m, s, sh = school_calendar(day)
        quarter[pos], module[pos], summer[pos], share[pos] = q, m, s, sh
        holiday_id[pos] = holiday_id_map.get(holiday_name[pos], 0)
        if sh > 0:
            days_to_break[pos] = 0
            days_from_break[pos] = 0
        else:
            future = [((start - day).days) for start, _end in ALL_BREAKS if start > day]
            past = [((day - end).days) for _start, end in ALL_BREAKS if end < day]
            if future:
                days_to_break[pos] = min(DISTANCE_CAP, min(future))
            if past:
                days_from_break[pos] = min(DISTANCE_CAP, min(past))
                if min(past) <= 7:  # первая неделя после каникул
                    first_school_week[pos] = 1
        future_h = [((days[other] - day).days) for other in range(pos + 1, count) if is_holiday[other]]
        past_h = [((day - days[other]).days) for other in range(pos - 1, -1, -1) if is_holiday[other]]
        if future_h:
            days_to_holiday[pos] = min(DISTANCE_CAP, future_h[0])
        if past_h:
            days_from_holiday[pos] = min(DISTANCE_CAP, past_h[0])
    working_saturday = ((weekend == 1) & (day_off == 0)).astype(np.float32)
    return {
        "quarter": quarter,
        "module": module,
        "summer": summer,
        "share": share,
        "holiday_id": holiday_id,
        "days_to_break": days_to_break,
        "days_from_break": days_from_break,
        "first_school_week": first_school_week,
        "working_saturday": working_saturday,
        "days_to_holiday": days_to_holiday,
        "days_from_holiday": days_from_holiday,
        "day_off": day_off.astype(np.float32),
        "weekend": weekend.astype(np.float32),
        "preholiday": preholiday.astype(np.float32),
        "short_day": short_day.astype(np.float32),
        "is_holiday": is_holiday.astype(np.float32),
    }


HORIZON_EDGES = (7, 14, 28, 42)


def horizon_bucket(horizon):
    for bucket, edge in enumerate(HORIZON_EDGES):
        if horizon <= edge:
            return bucket
    return len(HORIZON_EDGES)


def horizon_ranges():
    lows = (1,) + tuple(edge + 1 for edge in HORIZON_EDGES)
    highs = HORIZON_EDGES + (HORIZON,)
    return tuple(zip(lows, highs))


def decay_weights(count, decay):
    weights = (decay ** np.arange(count)).astype(np.float64)
    return weights / weights.sum()


def weekly_profiles(bundle, cutoff_pos, params=CHAMPION_PARAMS):
    days = bundle["days"]
    series = bundle["series"]
    weekly = np.zeros((len(ROUTES), 7, 24), np.float32)
    for dow in range(7):
        pos = cutoff_pos
        while pos >= 0 and days[pos].weekday() != dow:
            pos -= 1
        chosen = list(range(pos, -1, -7))[: params.weeks]
        if not chosen:
            continue
        weekly[:, dow, :] = np.tensordot(decay_weights(len(chosen), params.decay), series[:, chosen, :], axes=(0, 1))
    return weekly


def holiday_profile(bundle, cutoff_pos, params=CHAMPION_PARAMS):
    positions = [pos for pos, flag in enumerate(bundle["is_holiday"][: cutoff_pos + 1]) if flag]
    positions = positions[-params.holidays:][::-1]
    profile = np.zeros((len(ROUTES), 24), np.float32)
    if len(positions) < 2:
        return profile
    return np.tensordot(decay_weights(len(positions), params.decay), bundle["series"][:, positions, :], axes=(0, 1))


def day_group(day):
    """Будни делят одну форму часа, суббота и воскресенье — свои."""
    weekday = day.weekday()
    return 0 if weekday < 5 else weekday - 4


def pooled_shapes(bundle, cutoff_pos, params):
    days = bundle["days"]
    series = bundle["series"]
    out = np.zeros((len(ROUTES), 3, 24), np.float64)
    for group in range(3):
        cand = [
            pos for pos in range(cutoff_pos, max(-1, cutoff_pos - 7 * params.shape_weeks), -1)
            if day_group(days[pos]) == group and not bundle["is_holiday"][pos]
        ]
        if not cand:
            continue
        weights = params.shape_decay ** np.array([(cutoff_pos - pos) // 7 for pos in cand], np.float64)
        block = series[:, cand]
        shares = block / np.maximum(block.sum(axis=2, keepdims=True), 1e-6)
        out[:, group] = np.tensordot(weights, shares, axes=(0, 1)) / weights.sum()
    return out / np.maximum(out.sum(axis=2, keepdims=True), 1e-9)


def profile_window(bundle, cutoff, first, last, params=CHAMPION_PARAMS):
    days = bundle["days"]
    index = bundle["index"]
    cutoff_pos = index[cutoff]
    weekly = weekly_profiles(bundle, cutoff_pos, params)
    holidays = holiday_profile(bundle, cutoff_pos, params)
    pooled = pooled_shapes(bundle, cutoff_pos, params) if params.shape_pool > 0 else None
    positions = [index[day] for day in daterange(first, last)]
    forecast = np.zeros((len(positions), len(ROUTES), 24), np.float32)
    for row, pos in enumerate(positions):
        forecast[row] = profile_day(bundle, pos, weekly, holidays, pooled, params)
    return np.maximum(forecast, 0.0)


class ProfileState:
    """Профиль, зафиксированный на cutoff: прогноз любого будущего дня без пересчёта."""

    def __init__(self, bundle, cutoff_pos, params):
        self.bundle = bundle
        self.params = params
        self.weekly = weekly_profiles(bundle, cutoff_pos, params)
        self.holidays = holiday_profile(bundle, cutoff_pos, params)
        self.pooled = pooled_shapes(bundle, cutoff_pos, params) if params.shape_pool > 0 else None

    def day(self, pos):
        return profile_day(self.bundle, pos, self.weekly, self.holidays, self.pooled, self.params)


def profile_day(bundle, pos, weekly, holidays, pooled, params):
    if bundle["is_holiday"][pos] and holidays.sum() > 0:
        return holidays
    day = bundle["days"][pos]
    base = weekly[:, day.weekday(), :]
    if pooled is None:
        return base
    total = base.sum(axis=1, keepdims=True)
    own = base / np.maximum(total, 1e-6)
    shape = (1 - params.shape_pool) * own + params.shape_pool * pooled[:, day_group(day)]
    return total * shape


def champion_window(bundle, cutoff, first, last):
    return profile_window(bundle, cutoff, first, last, CHAMPION_PARAMS)


def actual_window(bundle, first, last):
    index = bundle["index"]
    positions = [index[day] for day in daterange(first, last)]
    return bundle["series"][:, positions, :].transpose(1, 0, 2).copy()


def wape(actual, forecast, mask=None):
    if mask is None:
        mask = np.ones(actual.shape, dtype=bool)
    y = actual[mask]
    yhat = forecast[mask]
    denom = float(y.sum())
    if denom <= 0:
        return 1.0, 0.0
    error = float(np.abs(y - yhat).sum() / denom)
    return error, max(0.0, 1.0 - error)


def slice_report(bundle, first, actual, forecast):
    days = list(daterange(first, first + timedelta(days=actual.shape[0] - 1)))
    index = bundle["index"]
    report = {}
    error, score = wape(actual, forecast)
    report["all"] = {"wape": error, "score": score}
    work = np.zeros(actual.shape, dtype=bool)
    holiday = np.zeros(actual.shape, dtype=bool)
    school_off = np.zeros(actual.shape, dtype=bool)
    morning = np.zeros(actual.shape, dtype=bool)
    evening = np.zeros(actual.shape, dtype=bool)
    for row, day in enumerate(days):
        pos = index[day]
        holiday[row] = bundle["is_holiday"][pos] == 1
        school_off[row] = bundle["calendar"]["share"][pos] > 0
        work[row] = (bundle["weekend"][pos] == 0) and (bundle["is_holiday"][pos] == 0) and (bundle["calendar"]["share"][pos] == 0)
        morning[row, :, sorted(bundle["morning_peak"])] = True
        evening[row, :, sorted(bundle["evening_peak"])] = True
    for name, mask in (("workday", work), ("holiday", holiday), ("school_off", school_off), ("morning_peak", morning), ("evening_peak", evening)):
        error, score = wape(actual, forecast, mask)
        report[name] = {"wape": error, "score": score}
    for route_i, route in enumerate(ROUTES):
        mask = np.zeros(actual.shape, dtype=bool)
        mask[:, route_i, :] = True
        error, score = wape(actual, forecast, mask)
        report[f"route_{route}"] = {"wape": error, "score": score}
    for low, high in horizon_ranges():
        if low > actual.shape[0]:
            continue
        mask = np.zeros(actual.shape, dtype=bool)
        mask[low - 1: high] = True
        error, score = wape(actual, forecast, mask)
        report[f"horizon_{low}_{high}"] = {"wape": error, "score": score}
    return report


def window_bounds(origin):
    first = origin + timedelta(days=1)
    last = min(OBSERVED_END, origin + timedelta(days=HORIZON))
    return first, last


def write_submission(predictions, path):
    with TEMPLATE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    missing = 0
    for row in rows:
        key = (int(row["route"]), row["date"], int(row["hour"]))
        value = predictions.get(key)
        if value is None:
            missing += 1
            row["prediction"] = "0"
        else:
            row["prediction"] = str(int(round(max(0.0, float(value)))))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["route", "date", "hour", "prediction"], delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    if missing or len(rows) != SUBMISSION_ROWS:
        raise RuntimeError(f"submission {path} rows={len(rows)} missing={missing}")
    return len(rows), missing
