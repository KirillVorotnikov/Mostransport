"""Ансамбль недельного профиля, LSTM, LightGBM и CatBoost.

LSTM заменяет BERT и учит поправку к профилю того же дня недели, а не абсолютный
уровень по замороженному окну. Потери LSTM и LightGBM — L1, это тот же числитель,
что у WAPE. Погоды нет. Лаги, праздничный профиль и офисы не читают дни позже среза.

Подбор гиперпараметров идёт по сентябрю. Веса маршрутов считаются по сентябрьскому
OOF. Октябрь в выборе не участвует и остаётся проверкой. LightGBM видит только
уже прошедшие выходные сентября–октября.
"""

from __future__ import annotations

import csv
from bisect import bisect_left
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "kaggle_mstrans"
ENR = ROOT / "data" / "kaggle_enrichment"
OUT = ROOT / "experiments" / "results" / "submission-lstm-lgbm-catboost.csv"

ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
START = date(2025, 1, 1)
END = date(2025, 12, 31)
VALID_START = date(2025, 9, 1)
VALID_END = date(2025, 10, 31)
FORECAST_START = date(2025, 11, 1)
FORECAST_END = date(2025, 12, 31)
TUNE_END = date(2025, 9, 30)
HOLD_START = date(2025, 10, 1)
LAGS = 4
DEEP = 8
DECAY = 0.75
HOLIDAY_POOL = 6
WEIGHT_STEP = 0.05
LSTM_BATCH = 64
LSTM_PATIENCE = 6
LSTM_MIN_EPOCHS = 10
MORNING_PEAK = {7, 8, 9}
EVENING_PEAK = {17, 18, 19}
FEATURES = [
    "route", "hour", "dow", "month", "day_of_month", "iso_week",
    "weekend", "day_off", "preholiday", "short_day", "holiday_id", "is_holiday",
    "quarter_break", "module_break", "summer_break", "school_off_share",
    "days_to_holiday", "days_since_holiday", "days_to_break",
    "morning_peak", "evening_peak", "weekday_peak", "school_peak",
    "profile", "base", "shock",
    "lag1", "lag2", "lag3", "lag4", "n_lags",
    "lag_std", "lag_trend", "lag_ratio", "deep_mean",
    "route_level", "level_trend", "hour_share",
    "offices", "acc7", "acc30", "tram7", "office_level",
]
CATEGORICAL = ["route", "hour", "dow", "month", "holiday_id"]
EXTRA_SIZE = 2
LGBM_GRID = [
    {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20, "num_boost_round": 200, "feature_fraction": 1.0, "lambda_l2": 0.0},
    {"learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 30, "num_boost_round": 400, "feature_fraction": 0.9, "lambda_l2": 1.0},
    {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 40, "num_boost_round": 300, "feature_fraction": 0.8, "lambda_l2": 1.0},
    {"learning_rate": 0.08, "num_leaves": 15, "min_child_samples": 50, "num_boost_round": 250, "feature_fraction": 0.9, "lambda_l2": 0.1},
    {"learning_rate": 0.02, "num_leaves": 63, "min_child_samples": 20, "num_boost_round": 600, "feature_fraction": 0.7, "lambda_l2": 2.0},
    {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 80, "num_boost_round": 350, "feature_fraction": 1.0, "lambda_l2": 0.5},
    {"learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 20, "num_boost_round": 150, "feature_fraction": 0.8, "lambda_l2": 0.0},
    {"learning_rate": 0.04, "num_leaves": 31, "min_child_samples": 25, "num_boost_round": 400, "feature_fraction": 0.85, "lambda_l2": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1},
]
CAT_GRID = [
    {"iterations": 500, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0, "min_data_in_leaf": 20},
    {"iterations": 800, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 5.0, "min_data_in_leaf": 40},
    {"iterations": 400, "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 3.0, "min_data_in_leaf": 20},
    {"iterations": 600, "depth": 4, "learning_rate": 0.08, "l2_leaf_reg": 1.0, "min_data_in_leaf": 50},
    {"iterations": 500, "depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 8.0, "min_data_in_leaf": 30},
    {"iterations": 700, "depth": 6, "learning_rate": 0.1, "l2_leaf_reg": 3.0, "min_data_in_leaf": 20},
]
LSTM_GRID = [
    {"hidden": 64, "layers": 1, "dropout": 0.1, "lr": 1e-3, "lookback": 4, "max_epochs": 30},
    {"hidden": 96, "layers": 1, "dropout": 0.1, "lr": 5e-4, "lookback": 6, "max_epochs": 40},
    {"hidden": 128, "layers": 1, "dropout": 0.0, "lr": 1e-3, "lookback": 8, "max_epochs": 30},
    {"hidden": 64, "layers": 2, "dropout": 0.2, "lr": 5e-4, "lookback": 6, "max_epochs": 40},
    {"hidden": 128, "layers": 2, "dropout": 0.1, "lr": 3e-4, "lookback": 8, "max_epochs": 40},
    {"hidden": 96, "layers": 1, "dropout": 0.2, "lr": 1e-3, "lookback": 4, "max_epochs": 25},
]


def daterange(start, end):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def is_weekend(day):
    return day.weekday() >= 5


def ewm(rows, decay=DECAY):
    if not rows:
        return np.zeros(24, np.float64)
    weights = decay ** np.arange(len(rows), dtype=np.float64)
    return np.tensordot(weights, np.stack(rows), axes=(0, 0)) / weights.sum()


def load():
    days = list(daterange(START, END))
    index = {day: pos for pos, day in enumerate(days)}
    series = np.zeros((len(ROUTES), len(days), 24), np.float32)
    route_pos = {route: pos for pos, route in enumerate(ROUTES)}
    for name in ("labels_day_train.csv", "labels_day_test.csv"):
        with (LABELS / name).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter=";"):
                day = date.fromisoformat(row["date"])
                pos = index.get(day)
                route = route_pos.get(int(row["route"]))
                if pos is None or route is None or day > VALID_END:
                    continue
                series[route, pos, int(row["hour"])] = float(row["boardings"])
    flags = {}
    names = []
    with (ENR / "holidays_2025.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            day = date.fromisoformat(row["date"])
            name = row["holiday"] or "none"
            if name != "none":
                names.append(name)
            flags[day] = (
                int(row["day_off"]),
                int(row["weekend"]),
                int(row["preholiday"]),
                int(row["short_day"]),
                name,
            )
    holiday_id = {name: pos + 1 for pos, name in enumerate(sorted(set(names)))}
    school = {"quarter": [], "module": [], "summer": []}
    with (ENR / "school_calendar.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            school[row["system"]].append((date.fromisoformat(row["start"]), date.fromisoformat(row["end"])))
    office = {}
    with (ENR / "route_day.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            office[(int(row["route"]), date.fromisoformat(row["date"]))] = (
                float(row["offices"]),
                float(row["acc7"]),
                float(row["acc30"]),
                float(row["tram7"]),
            )
    holiday_dates = sorted(day for day, row in flags.items() if row[4] not in ("", "none"))
    holiday_pos = [index[day] for day in holiday_dates if day in index]
    intervals = [span for spans in school.values() for span in spans]

    def days_to_break(day):
        for start, finish in intervals:
            if start <= day <= finish:
                return 0
        ahead = [(start - day).days for start, _finish in intervals if start > day]
        return int(min(60, min(ahead))) if ahead else 60

    distances = {}
    for day in days:
        pos = bisect_left(holiday_dates, day)
        if pos < len(holiday_dates) and holiday_dates[pos] == day:
            ahead, behind = 0, 0
        else:
            ahead = (holiday_dates[pos] - day).days if pos < len(holiday_dates) else 45
            behind = (day - holiday_dates[pos - 1]).days if pos > 0 else 45
        distances[day] = (min(ahead, 45), min(behind, 45), days_to_break(day))
    return {
        "days": days,
        "index": index,
        "series": series,
        "flags": flags,
        "holiday_id": holiday_id,
        "school": school,
        "office": office,
        "holiday_pos": holiday_pos,
        "distances": distances,
    }


def school_flags(school, day):
    def inside(name):
        return any(start <= day <= finish for start, finish in school[name])

    summer = int(inside("summer"))
    quarter = int(summer or inside("quarter"))
    module = int(summer or inside("module"))
    return quarter, module, summer, (quarter + module) / 2.0


def office_asof(office, route_no, day):
    for _ in range(45):
        row = office.get((route_no, day))
        if row is not None:
            return row
        day -= timedelta(days=1)
    return (0.0, 0.0, 0.0, 0.0)


def flag_row(bundle, day):
    return bundle["flags"].get(day, (int(day.weekday() >= 5), int(day.weekday() >= 5), 0, 0, "none"))


def same_dow_rows(bundle, pos, route, observed_pos, count):
    days = bundle["days"]
    series = bundle["series"]
    dow = days[pos].weekday()
    rows = []
    cursor = pos - 7
    while cursor >= 0 and len(rows) < count:
        if days[cursor].weekday() == dow and cursor <= observed_pos:
            rows.append(series[route, cursor].astype(np.float64))
        cursor -= 7
    return rows


def holiday_rows(bundle, pos, route, observed_pos):
    series = bundle["series"]
    chosen = [item for item in bundle["holiday_pos"] if item < pos and item <= observed_pos]
    chosen = chosen[-HOLIDAY_POOL:][::-1]
    return [series[route, item].astype(np.float64) for item in chosen]


def lag_views(bundle, pos, route, observed_pos, lookback):
    """Новейшие лаги того же дня недели и профиль. Дни после observed_pos не читаются."""
    hours = same_dow_rows(bundle, pos, route, observed_pos, max(DEEP, lookback))
    base = ewm(hours[:LAGS])
    holidays = holiday_rows(bundle, pos, route, observed_pos)
    day = bundle["days"][pos]
    is_holiday = flag_row(bundle, day)[4] not in ("", "none")
    profile = ewm(holidays) if is_holiday and len(holidays) >= 2 else base
    deep = np.mean(np.stack(hours[:DEEP]), axis=0) if hours else np.zeros(24, np.float64)
    daily = [float(row.sum()) for row in hours[:LAGS]]
    return hours, base, profile, deep, daily


def build_frame(bundle, days_to_use, with_target, asof=None, observed_end=None):
    days = bundle["days"]
    index = bundle["index"]
    series = bundle["series"]
    observed_end = VALID_END if observed_end is None else observed_end
    observed_pos = index[observed_end]
    rows = []
    target = []
    keys = []
    for day in days_to_use:
        pos = index.get(day)
        if pos is None:
            continue
        feature_day = min(day, asof) if asof is not None else day
        day_off, weekend, preholiday, short_day, holiday_name = flag_row(bundle, day)
        is_holiday = int(holiday_name not in ("", "none"))
        quarter, module, summer, share = school_flags(bundle["school"], day)
        to_holiday, since_holiday, to_break = bundle["distances"][day]
        for route, route_no in enumerate(ROUTES):
            hours, base_vec, profile_vec, deep_vec, daily = lag_views(bundle, pos, route, observed_pos, DEEP)
            offices, acc7, acc30, tram7 = office_asof(bundle["office"], route_no, feature_day)
            level = float(np.mean(daily)) if daily else 0.0
            level_trend = float(daily[0] - daily[-1]) if len(daily) >= 2 else 0.0
            found = min(LAGS, len(hours))
            for hour in range(24):
                real = [float(hours[item][hour]) for item in range(found)]
                lags = real + [0.0] * (LAGS - found)
                base = float(base_vec[hour])
                profile = float(profile_vec[hour])
                morning = int(hour in MORNING_PEAK)
                evening = int(hour in EVENING_PEAK)
                weekday_peak = int((morning or evening) and not weekend and not is_holiday)
                values = {
                    "route": route_no,
                    "hour": hour,
                    "dow": day.weekday(),
                    "month": day.month,
                    "day_of_month": day.day,
                    "iso_week": day.isocalendar()[1],
                    "weekend": weekend,
                    "day_off": day_off,
                    "preholiday": preholiday,
                    "short_day": short_day,
                    "holiday_id": bundle["holiday_id"].get(holiday_name, 0),
                    "is_holiday": is_holiday,
                    "quarter_break": quarter,
                    "module_break": module,
                    "summer_break": summer,
                    "school_off_share": share,
                    "days_to_holiday": to_holiday,
                    "days_since_holiday": since_holiday,
                    "days_to_break": to_break,
                    "morning_peak": morning,
                    "evening_peak": evening,
                    "weekday_peak": weekday_peak,
                    "school_peak": weekday_peak * (1.0 - share),
                    "profile": profile,
                    "base": base,
                    "shock": (real[0] - profile) if real else 0.0,
                    "lag1": lags[0],
                    "lag2": lags[1],
                    "lag3": lags[2],
                    "lag4": lags[3],
                    "n_lags": found,
                    "lag_std": float(np.std(real)) if len(real) >= 2 else 0.0,
                    "lag_trend": (real[0] - real[-1]) if len(real) >= 2 else 0.0,
                    "lag_ratio": real[0] / max(base, 1.0) if real else 0.0,
                    "deep_mean": float(deep_vec[hour]),
                    "route_level": level,
                    "level_trend": level_trend,
                    "hour_share": base / max(level, 1.0),
                    "offices": offices,
                    "acc7": acc7,
                    "acc30": acc30,
                    "tram7": tram7,
                    "office_level": offices / max(level, 1.0),
                }
                rows.append([values[name] for name in FEATURES])
                keys.append((route_no, day.isoformat(), hour))
                if with_target:
                    target.append(float(series[route, pos, hour]))
    frame = np.asarray(rows, np.float32)
    if len(FEATURES) != len(set(FEATURES)):
        raise RuntimeError("duplicate feature names")
    if len(frame) and frame.shape[1] != len(FEATURES):
        raise RuntimeError(f"features {frame.shape[1]} != {len(FEATURES)}")
    return frame, (np.asarray(target, np.float32) if with_target else None), keys


def fit_lgbm(frame, target, spec):
    categorical = [FEATURES.index(name) for name in CATEGORICAL]
    train = lgb.Dataset(frame, label=target, feature_name=FEATURES, categorical_feature=categorical, free_raw_data=False)
    params = {
        "objective": "regression_l1",
        "learning_rate": spec["learning_rate"],
        "num_leaves": spec["num_leaves"],
        "min_child_samples": spec["min_child_samples"],
        "feature_fraction": spec.get("feature_fraction", 1.0),
        "lambda_l2": spec.get("lambda_l2", 0.0),
        "bagging_fraction": spec.get("bagging_fraction", 1.0),
        "bagging_freq": spec.get("bagging_freq", 0),
        "verbosity": -1,
        "seed": 7,
        "num_threads": 4,
    }
    return lgb.train(params, train, num_boost_round=spec["num_boost_round"])


def cat_frame(frame):
    table = pd.DataFrame(frame, columns=FEATURES)
    for name in CATEGORICAL:
        table[name] = table[name].round().astype(np.int32)
    return table


def fit_catboost(frame, target, spec):
    model = CatBoostRegressor(
        iterations=spec["iterations"],
        depth=spec["depth"],
        learning_rate=spec["learning_rate"],
        l2_leaf_reg=spec["l2_leaf_reg"],
        min_data_in_leaf=spec["min_data_in_leaf"],
        loss_function="MAE",
        boosting_type="Plain",
        one_hot_max_size=32,
        random_seed=7,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(cat_frame(frame), target, cat_features=CATEGORICAL)
    return model


def wape(actual, forecast):
    actual = np.asarray(actual, np.float64)
    forecast = np.asarray(forecast, np.float64)
    denom = float(actual.sum())
    if denom <= 0:
        return 1.0, 0.0
    error = float(np.abs(forecast - actual).sum() / denom)
    return error, max(0.0, 1.0 - error)


def weight_grid(n_models):
    units = int(round(1 / WEIGHT_STEP))
    current = [0] * n_models

    def walk(dim, left):
        if dim == n_models - 1:
            current[dim] = left
            yield tuple(value / units for value in current)
            return
        for value in range(left + 1):
            current[dim] = value
            yield from walk(dim + 1, left - value)

    yield from walk(0, units)


def learn_weights(keys, actual, columns):
    routes = np.array([key[0] for key in keys])
    grid = list(weight_grid(len(columns)))
    weights = {}
    for route in ROUTES:
        mask = routes == route
        y = actual[mask]
        if not np.any(mask) or float(y.sum()) <= 0:
            weights[route] = (1.0,) + (0.0,) * (len(columns) - 1)
            continue
        preds = [column[mask] for column in columns]
        best = None
        for candidate in grid:
            forecast = sum(weight * pred for weight, pred in zip(candidate, preds))
            error = float(np.abs(y - forecast).sum())
            if best is None or error < best[0]:
                best = (error, candidate)
        weights[route] = best[1]
    return weights


def apply_weights(keys, columns, weights):
    out = np.zeros(len(keys), np.float64)
    for index, key in enumerate(keys):
        out[index] = sum(weight * column[index] for weight, column in zip(weights[key[0]], columns))
    return np.maximum(out, 0.0)


def dead_routes(bundle, observed_end):
    series = bundle["series"]
    observed_pos = bundle["index"][observed_end]
    dead = set()
    for route, route_no in enumerate(ROUTES):
        if float(series[route, : observed_pos + 1].sum()) <= 0:
            dead.add(route_no)
    return dead


def weekend_groups(first, last):
    groups = {}
    for day in daterange(first, last):
        if is_weekend(day):
            groups.setdefault(day.isocalendar()[:2], []).append(day)
    return [groups[key] for key in sorted(groups)]


def week_groups(first, last):
    groups = {}
    for day in daterange(first, last):
        groups.setdefault(day.isocalendar()[:2], []).append(day)
    return [groups[key] for key in sorted(groups)]


def history_days(bundle, last_day):
    return [day for day in bundle["days"] if day <= last_day]


def tune_lgbm(bundle):
    """Leave-one-weekend-out только вперёд: тест — выходные сентября, учим более ранние."""
    groups = weekend_groups(VALID_START, TUNE_END)
    best_spec, best_score = None, None
    for spec in LGBM_GRID:
        actual, pred = [], []
        usable = 0
        for held_index, held_days in enumerate(groups):
            train_days = [day for group in groups[:held_index] for day in group]
            if not train_days:
                continue
            cutoff = min(held_days) - timedelta(days=1)
            train_x, train_y, _keys = build_frame(bundle, train_days, True, observed_end=cutoff)
            hold_x, hold_y, _hold_keys = build_frame(bundle, held_days, True, asof=cutoff, observed_end=cutoff)
            model = fit_lgbm(train_x, train_y, spec)
            actual.append(hold_y)
            pred.append(np.maximum(model.predict(hold_x), 0))
            usable += 1
        if not usable:
            raise RuntimeError("lgbm tune produced no folds")
        score = wape(np.concatenate(actual), np.concatenate(pred))[1]
        print(
            f"lgbm trial lr={spec['learning_rate']} leaves={spec['num_leaves']} "
            f"child={spec['min_child_samples']} rounds={spec['num_boost_round']} september={score:.4f}",
            flush=True,
        )
        if best_score is None or score > best_score:
            best_spec, best_score = spec, score
    print(f"lgbm selected {best_spec} september={best_score:.4f}", flush=True)
    return best_spec


def tune_catboost(bundle):
    """Один срез: учим до 31 августа, считаем сентябрь. Октябрь не смотрим."""
    train_days = history_days(bundle, TUNE_END.replace(month=8, day=31))
    cutoff = date(2025, 8, 31)
    september = list(daterange(VALID_START, TUNE_END))
    train_x, train_y, _keys = build_frame(bundle, train_days, True, observed_end=cutoff)
    hold_x, hold_y, _hold_keys = build_frame(bundle, september, True, asof=cutoff, observed_end=cutoff)
    hold_table = cat_frame(hold_x)
    best_spec, best_score = None, None
    for spec in CAT_GRID:
        print(
            f"catboost trial iters={spec['iterations']} depth={spec['depth']} "
            f"lr={spec['learning_rate']} l2={spec['l2_leaf_reg']} leaf={spec['min_data_in_leaf']}",
            flush=True,
        )
        model = fit_catboost(train_x, train_y, spec)
        score = wape(hold_y, np.maximum(model.predict(hold_table), 0))[1]
        print(f"catboost september={score:.4f}", flush=True)
        del model
        if best_score is None or score > best_score:
            best_spec, best_score = spec, score
    print(f"catboost selected {best_spec} september={best_score:.4f}", flush=True)
    return best_spec


def day_context(bundle, day):
    dow = np.zeros(7, np.float32)
    dow[day.weekday()] = 1.0
    day_off, weekend, preholiday, short_day, name = flag_row(bundle, day)
    quarter, module, summer, share = school_flags(bundle["school"], day)
    to_holiday, since_holiday, to_break = bundle["distances"][day]
    tail = np.array([
        weekend, day_off, preholiday, short_day, float(name not in ("", "none")),
        quarter, module, summer, share,
        to_holiday / 45.0, since_holiday / 45.0, to_break / 60.0,
        day.month / 12.0, day.isocalendar()[1] / 53.0, day.day / 31.0,
    ], np.float32)
    return np.concatenate([dow, tail])


class ResidualLSTM(nn.Module):
    def __init__(self, calendar_size, spec):
        super().__init__()
        hidden = spec["hidden"]
        layers = spec["layers"]
        dropout = spec["dropout"]
        self.route = nn.Embedding(len(ROUTES), 8)
        self.lstm = nn.LSTM(
            24,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden + 8 + calendar_size + EXTRA_SIZE)
        self.head = nn.Sequential(
            nn.Linear(hidden + 8 + calendar_size + EXTRA_SIZE, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 24),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, history, route, calendar, extra, base):
        encoded, _state = self.lstm(history)
        joined = torch.cat([encoded[:, -1], self.route(route), calendar, extra], dim=1)
        delta = self.head(self.norm(joined))
        return (base + delta).clamp(min=0)


def route_scale(series, observed_pos):
    return np.maximum(series[:, : observed_pos + 1].mean(axis=(1, 2)), 1.0).astype(np.float32)


def lstm_sequence(hours, lookback):
    newest_first = hours[:lookback]
    oldest_first = list(reversed(newest_first))
    pad = [np.zeros(24, np.float64) for _ in range(lookback - len(oldest_first))]
    return np.stack(pad + oldest_first, axis=0)


def collect_lstm_samples(bundle, history_end, spec):
    days = bundle["days"]
    index = bundle["index"]
    series = bundle["series"]
    observed_pos = index[history_end]
    scale = route_scale(series, observed_pos)
    live = series[:, : observed_pos + 1].sum(axis=(1, 2)) > 0
    lookback = spec["lookback"]
    histories, routes, calendars, extras, bases, targets, positions, weights = [], [], [], [], [], [], [], []
    for pos in range(7, observed_pos + 1):
        calendar = day_context(bundle, days[pos])
        for route in range(len(ROUTES)):
            hours, _base, profile, _deep, daily = lag_views(bundle, pos, route, observed_pos, lookback)
            if not hours:
                continue
            level = float(np.mean(daily)) if daily else 0.0
            level_trend = float(daily[0] - daily[-1]) if len(daily) >= 2 else 0.0
            divisor = max(float(scale[route]) * 24.0, 1.0)
            histories.append(lstm_sequence(hours, lookback) / scale[route])
            routes.append(route)
            calendars.append(calendar)
            extras.append([level / divisor, level_trend / divisor])
            bases.append(profile / scale[route])
            targets.append(series[route, pos] / scale[route])
            positions.append(pos)
            weights.append(1.0 if live[route] else 0.0)
    data = {
        "history": torch.tensor(np.asarray(histories, np.float32)),
        "route": torch.tensor(np.asarray(routes, np.int64)),
        "calendar": torch.tensor(np.asarray(calendars, np.float32)),
        "extra": torch.tensor(np.asarray(extras, np.float32)),
        "base": torch.tensor(np.asarray(bases, np.float32)),
        "target": torch.tensor(np.asarray(targets, np.float32)),
        "scale": torch.tensor(scale[np.asarray(routes)], dtype=torch.float32),
        "weight": torch.tensor(np.asarray(weights, np.float32)),
        "position": np.asarray(positions, np.int64),
    }
    return data, scale


def raw_l1(pred, target, scale, weight):
    error = ((pred - target).abs() * scale[:, None]).mean(dim=1)
    denom = weight.sum().clamp(min=1.0)
    return (error * weight).sum() / denom


def fit_lstm(bundle, history_end, spec, use_all_epochs=False):
    """Учит поправку к профилю. Последние две недели до среза всегда в обучении.

    Валидация ранней остановки — более ранние 14 дней, не будущая неделя.
    Финальный прогноз учится на всех днях до среза и проходит весь бюджет эпох:
    ранняя остановка на октябрьском окне оставляла первую эпоху и расходилась с OOF.
    """
    data, scale = collect_lstm_samples(bundle, history_end, spec)
    if use_all_epochs:
        train_idx = np.arange(len(data["position"]))
        val_idx = np.array([], np.int64)
    else:
        index = bundle["index"]
        val_lo = index[history_end - timedelta(days=27)]
        val_hi = index[history_end - timedelta(days=14)]
        is_val = (data["position"] >= val_lo) & (data["position"] <= val_hi)
        train_idx = np.flatnonzero(~is_val)
        val_idx = np.flatnonzero(is_val)
        if len(train_idx) == 0:
            train_idx = np.arange(len(data["position"]))
            val_idx = np.array([], np.int64)
    torch.manual_seed(7)
    model = ResidualLSTM(data["calendar"].shape[1], spec)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=spec["max_epochs"])
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = None
    best_epoch = 0
    patience = 0
    model.train()
    for epoch in range(spec["max_epochs"]):
        order = train_idx[torch.randperm(len(train_idx)).numpy()]
        for start in range(0, len(order), LSTM_BATCH):
            batch = torch.tensor(order[start: start + LSTM_BATCH], dtype=torch.long)
            optimizer.zero_grad()
            pred = model(data["history"][batch], data["route"][batch], data["calendar"][batch], data["extra"][batch], data["base"][batch])
            loss = raw_l1(pred, data["target"][batch], data["scale"][batch], data["weight"][batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        if len(val_idx) == 0:
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            continue
        model.eval()
        with torch.no_grad():
            val_batch = torch.tensor(val_idx, dtype=torch.long)
            pred = model(
                data["history"][val_batch], data["route"][val_batch], data["calendar"][val_batch],
                data["extra"][val_batch], data["base"][val_batch],
            )
            val_loss = float(raw_l1(pred, data["target"][val_batch], data["scale"][val_batch], data["weight"][val_batch]))
        model.train()
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            patience = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
            if epoch + 1 >= LSTM_MIN_EPOCHS and patience >= LSTM_PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, scale, best_epoch, best_val


def predict_lstm(model, scale, bundle, observed_end, predict_days, spec):
    days = bundle["days"]
    index = bundle["index"]
    observed_pos = index[observed_end]
    live = bundle["series"][:, : observed_pos + 1].sum(axis=(1, 2)) > 0
    lookup = {}
    lookback = spec["lookback"]
    with torch.no_grad():
        for day in predict_days:
            pos = index[day]
            calendar = torch.tensor(day_context(bundle, day), dtype=torch.float32).repeat(len(ROUTES), 1)
            histories, extras, bases = [], [], []
            for route in range(len(ROUTES)):
                hours, _base, profile, _deep, daily = lag_views(bundle, pos, route, observed_pos, lookback)
                level = float(np.mean(daily)) if daily else 0.0
                level_trend = float(daily[0] - daily[-1]) if len(daily) >= 2 else 0.0
                divisor = max(float(scale[route]) * 24.0, 1.0)
                histories.append(lstm_sequence(hours, lookback) / scale[route])
                extras.append([level / divisor, level_trend / divisor])
                bases.append(profile / max(float(scale[route]), 1.0))
            pred = model(
                torch.tensor(np.asarray(histories, np.float32)),
                torch.arange(len(ROUTES)),
                calendar,
                torch.tensor(np.asarray(extras, np.float32)),
                torch.tensor(np.asarray(bases, np.float32)),
            ).numpy() * scale[:, None]
            for route, route_no in enumerate(ROUTES):
                for hour in range(24):
                    value = 0.0 if not live[route] else max(0.0, float(pred[route, hour]))
                    lookup[(route_no, day.isoformat(), hour)] = value
    return lookup


def tune_lstm(bundle):
    september = list(daterange(VALID_START, TUNE_END))
    cutoff = date(2025, 8, 31)
    best_spec, best_score = None, None
    actual = None
    for spec in LSTM_GRID:
        print(
            f"lstm trial hidden={spec['hidden']} layers={spec['layers']} dropout={spec['dropout']} "
            f"lr={spec['lr']} lookback={spec['lookback']} epochs={spec['max_epochs']}",
            flush=True,
        )
        model, scale, epoch, val_loss = fit_lstm(bundle, cutoff, spec)
        lookup = predict_lstm(model, scale, bundle, cutoff, september, spec)
        if actual is None:
            _frame, actual, keys = build_frame(bundle, september, True, asof=cutoff, observed_end=cutoff)
        forecast = np.array([lookup[key] for key in keys], np.float64)
        score = wape(actual, forecast)[1]
        val_text = "none" if val_loss is None else f"{val_loss:.2f}"
        print(f"lstm september={score:.4f} best_epoch={epoch} val_l1={val_text}", flush=True)
        del model
        if best_score is None or score > best_score:
            best_spec, best_score = spec, score
    print(f"lstm selected {best_spec} september={best_score:.4f}", flush=True)
    return best_spec


def store_profile(frame, keys, actual, profile, actual_map):
    column = frame[:, FEATURES.index("profile")]
    for key, value, truth in zip(keys, column, actual):
        profile[key] = float(value)
        actual_map[key] = float(truth)


def oof_predictions(bundle, lgbm_spec, cat_spec, lstm_spec):
    profile, actual, lstm_map, cat_map, lgbm_map = {}, {}, {}, {}, {}
    for held_days in week_groups(VALID_START, VALID_END):
        cutoff = min(held_days) - timedelta(days=1)
        train_days = history_days(bundle, cutoff)
        train_x, train_y, _keys = build_frame(bundle, train_days, True, observed_end=cutoff)
        hold_x, hold_y, hold_keys = build_frame(bundle, held_days, True, asof=cutoff, observed_end=cutoff)
        print(f"oof week {min(held_days).isoformat()} catboost train {len(train_x)}", flush=True)
        model = fit_catboost(train_x, train_y, cat_spec)
        prediction = np.maximum(model.predict(cat_frame(hold_x)), 0)
        del model
        for key, value in zip(hold_keys, prediction):
            cat_map[key] = float(value)
        store_profile(hold_x, hold_keys, hold_y, profile, actual)
        print(f"oof week {min(held_days).isoformat()} lstm", flush=True)
        lstm_model, scale, epoch, _val = fit_lstm(bundle, cutoff, lstm_spec)
        lstm_map.update(predict_lstm(lstm_model, scale, bundle, cutoff, held_days, lstm_spec))
        print(f"oof lstm epoch {epoch}", flush=True)
        del lstm_model
    for held_index, held_days in enumerate(weekend_groups(VALID_START, VALID_END)):
        train_days = [day for group in weekend_groups(VALID_START, VALID_END)[:held_index] for day in group]
        if not train_days:
            continue
        cutoff = min(held_days) - timedelta(days=1)
        train_x, train_y, _keys = build_frame(bundle, train_days, True, observed_end=cutoff)
        hold_x, _hold_y, hold_keys = build_frame(bundle, held_days, True, asof=cutoff, observed_end=cutoff)
        print(f"oof weekend {min(held_days).isoformat()} lgbm train {len(train_x)}", flush=True)
        model = fit_lgbm(train_x, train_y, lgbm_spec)
        for key, value in zip(hold_keys, np.maximum(model.predict(hold_x), 0)):
            lgbm_map[key] = float(value)
        del model
    return profile, actual, lstm_map, lgbm_map, cat_map


def keys_of(actual, month, weekend):
    chosen = []
    for key in actual:
        day = date.fromisoformat(key[1])
        if day.month == month and is_weekend(day) == weekend:
            chosen.append(key)
    return chosen


def mapped(keys, lookup):
    return np.array([lookup[key] for key in keys], np.float64)


def report_slice(title, keys, actual, columns, names, weights):
    y = mapped(keys, actual)
    parts = []
    arrays = []
    for name, column in zip(names, columns):
        values = mapped(keys, column)
        arrays.append(values)
        parts.append(f"{name} {wape(y, values)[1]:.4f}")
    blend = apply_weights(keys, arrays, weights)
    error, score = wape(y, blend)
    print(f"{title} " + " ".join(parts) + f" blend {score:.4f} wape {error:.4f}")
    return error, score


def main():
    if len(FEATURES) != 43:
        raise RuntimeError(f"unexpected feature count {len(FEATURES)}")
    torch.set_num_threads(4)
    bundle = load()
    print(f"features {len(FEATURES)}", flush=True)
    lgbm_spec = tune_lgbm(bundle)
    cat_spec = tune_catboost(bundle)
    lstm_spec = tune_lstm(bundle)
    profile, actual, lstm_map, lgbm_map, cat_map = oof_predictions(bundle, lgbm_spec, cat_spec, lstm_spec)

    september_weekend = [key for key in keys_of(actual, 9, True) if key in lgbm_map]
    september_weekday = keys_of(actual, 9, False)
    weekend_weights = learn_weights(
        september_weekend,
        mapped(september_weekend, actual),
        [mapped(september_weekend, column) for column in (profile, lstm_map, lgbm_map, cat_map)],
    )
    weekday_weights = learn_weights(
        september_weekday,
        mapped(september_weekday, actual),
        [mapped(september_weekday, column) for column in (profile, lstm_map, cat_map)],
    )
    print("september fit, weights chosen here", flush=True)
    report_slice("september weekend", september_weekend, actual, (profile, lstm_map, lgbm_map, cat_map), ("profile", "lstm", "lgbm", "cat"), weekend_weights)
    report_slice("september weekday", september_weekday, actual, (profile, lstm_map, cat_map), ("profile", "lstm", "cat"), weekday_weights)
    october_weekend = [key for key in keys_of(actual, 10, True) if key in lgbm_map]
    october_weekday = keys_of(actual, 10, False)
    print("october holdout, weights frozen", flush=True)
    report_slice("october weekend", october_weekend, actual, (profile, lstm_map, lgbm_map, cat_map), ("profile", "lstm", "lgbm", "cat"), weekend_weights)
    report_slice("october weekday", october_weekday, actual, (profile, lstm_map, cat_map), ("profile", "lstm", "cat"), weekday_weights)
    october_keys = october_weekday + october_weekend
    october_actual = mapped(october_keys, actual)
    october_blend = np.concatenate([
        apply_weights(october_weekday, [mapped(october_weekday, column) for column in (profile, lstm_map, cat_map)], weekday_weights),
        apply_weights(october_weekend, [mapped(october_weekend, column) for column in (profile, lstm_map, lgbm_map, cat_map)], weekend_weights),
    ])
    october_profile = mapped(october_keys, profile)
    print(f"october total blend {wape(october_actual, october_blend)[1]:.4f} profile {wape(october_actual, october_profile)[1]:.4f}")
    print("weekend profile/lstm/lgbm/cat", " ".join(
        f"{route}:{weekend_weights[route][0]:.2f}/{weekend_weights[route][1]:.2f}/{weekend_weights[route][2]:.2f}/{weekend_weights[route][3]:.2f}"
        for route in ROUTES
    ))
    print("weekday profile/lstm/cat", " ".join(
        f"{route}:{weekday_weights[route][0]:.2f}/{weekday_weights[route][1]:.2f}/{weekday_weights[route][2]:.2f}"
        for route in ROUTES
    ))

    export_forecast(bundle, lgbm_spec, cat_spec, lstm_spec, weekend_weights, weekday_weights)


def export_forecast(bundle, lgbm_spec, cat_spec, lstm_spec, weekend_weights, weekday_weights):
    weekend_train = [day for day in daterange(VALID_START, VALID_END) if is_weekend(day)]
    lgbm_x, lgbm_y, _keys = build_frame(bundle, weekend_train, True, observed_end=VALID_END)
    print(f"refit lgbm weekends {len(lgbm_x)}", flush=True)
    lgbm_model = fit_lgbm(lgbm_x, lgbm_y, lgbm_spec)
    cat_days = history_days(bundle, VALID_END)
    cat_x, cat_y, _keys = build_frame(bundle, cat_days, True, observed_end=VALID_END)
    print(f"refit catboost days {len(cat_x)}", flush=True)
    cat_model = fit_catboost(cat_x, cat_y, cat_spec)
    print("refit lstm through 2025-10-31", flush=True)
    lstm_model, lstm_scale, epoch, _val = fit_lstm(bundle, VALID_END, lstm_spec, use_all_epochs=True)
    print(f"refit lstm epoch {epoch}", flush=True)

    forecast_days = list(daterange(FORECAST_START, FORECAST_END))
    future_x, _future_y, future_keys = build_frame(bundle, forecast_days, False, asof=VALID_END, observed_end=VALID_END)
    probe = future_keys.index((1, "2025-12-01", 8))
    print(
        f"dec1 route1 hour8 profile {future_x[probe, FEATURES.index('profile')]:.1f} "
        f"lag1 {future_x[probe, FEATURES.index('lag1')]:.1f}",
        flush=True,
    )
    profile_future = {key: float(value) for key, value in zip(future_keys, future_x[:, FEATURES.index("profile")])}
    cat_future = {key: float(value) for key, value in zip(future_keys, np.maximum(cat_model.predict(cat_frame(future_x)), 0))}
    weekend_days = [day for day in forecast_days if is_weekend(day)]
    weekend_x, _y, weekend_keys = build_frame(bundle, weekend_days, False, asof=VALID_END, observed_end=VALID_END)
    lgbm_future = {key: float(value) for key, value in zip(weekend_keys, np.maximum(lgbm_model.predict(weekend_x), 0))}
    lstm_future = predict_lstm(lstm_model, lstm_scale, bundle, VALID_END, forecast_days, lstm_spec)
    dead = dead_routes(bundle, VALID_END)
    blended_keys = []
    blended = []
    for key in future_keys:
        route, day_text, _hour = key
        if route in dead:
            value = 0.0
        elif is_weekend(date.fromisoformat(day_text)):
            weights = weekend_weights[route]
            value = (
                weights[0] * profile_future[key] + weights[1] * lstm_future[key]
                + weights[2] * lgbm_future[key] + weights[3] * cat_future[key]
            )
        else:
            weights = weekday_weights[route]
            value = weights[0] * profile_future[key] + weights[1] * lstm_future[key] + weights[2] * cat_future[key]
        blended_keys.append(key)
        blended.append(max(0.0, value))
    rows, missing = write_submission(blended_keys, blended)
    print(f"forecast mean {float(np.mean(blended)):.1f} sum {float(np.sum(blended)):.0f}")
    print(f"wrote {OUT} rows {rows} missing {missing}")


def write_submission(keys, forecast):
    lookup = {key: max(0.0, float(value)) for key, value in zip(keys, forecast)}
    template = LABELS / "test_submission.csv"
    with template.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    missing = 0
    for row in rows:
        key = (int(row["route"]), row["date"], int(row["hour"]))
        value = lookup.get(key)
        if value is None:
            missing += 1
            row["prediction"] = "0"
        else:
            row["prediction"] = str(int(round(value)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["route", "date", "hour", "prediction"], delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    if missing or len(rows) != 14640:
        raise RuntimeError(f"rows={len(rows)} missing={missing}")
    return len(rows), missing


if __name__ == "__main__":
    main()
