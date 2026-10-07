"""Сезонный профиль с подобранными параметрами, прямой дневной LightGBM и форма часа.

Порядок:
1. Параметры профиля (глубина недель, затухание, пул формы будней) выбираются
   по окнам разработки и принимаются, только если знаковый тест по всем окнам
   (разработка, заблокированное, осенние проверки) значим против champion.
2. Принятый профиль — база и признак для деревьев. Деревья обучаются с
   запасом MAX_ROUNDS, рабочее число итераций берётся по ошибке окон разработки.
3. Веса ансамбля выбираются по окнам разработки; сентябрь–октябрь от
   заблокированной точки в подбор не входит.
Погода будущего дня заменена климатом месяцев, уже наблюдавшихся к cutoff.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from dataclasses import asdict
from datetime import timedelta
from math import comb
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from forecast_core import (
    RESULTS,
    CHAMPION_PARAMS,
    CHECK_ORIGINS,
    DEV_ORIGINS,
    FORECAST_END,
    FORECAST_START,
    HORIZON,
    LOCK_ORIGIN,
    OBSERVED_END,
    ROUTES,
    START,
    ProfileParams,
    ProfileState,
    actual_window,
    champion_window,
    daterange,
    freeze_champion,
    horizon_bucket,
    horizon_ranges,
    load_bundle,
    profile_window,
    slice_report,
    wape,
    window_bounds,
    write_submission,
)

OUT_NAME = "submission-pooled-profile-lgbm"
MAX_ROUNDS = 800
ROUND_GRID = tuple(sorted({int(round(value)) for value in np.geomspace(25, MAX_ROUNDS, 8)}))
TRAIN_STEP_DAYS = 7
MAX_LAG_DAYS = 56
MIN_BUCKET_ROWS = 80
NLINEAR_LOOKBACK = 28
WEIGHT_STEP = 0.1
SIGN_TEST_ALPHA = 0.05
GATE_MIN_GAIN = 0.005
GATE_MIN_WINS = 3
GATE_MAX_DROP = 0.01
GATE_WORKDAY_TOL = 0.002
PROFILE_GRID = {
    "weeks": (3, 4, 5, 6),
    "decay": (0.6, 0.75, 0.9),
    "holidays": (4, 6, 8),
    "shape_pool": (0.0, 0.25, 0.5, 0.75),
    "shape_weeks": (2, 4, 8),
    "shape_decay": (0.6, 1.0),
}

DAILY_FEATURES = [
    "route", "dow", "month", "horizon", "horizon_bucket",
    "dow1", "dow2", "dow3", "dow4", "dow8",
    "level1", "mean7", "mean14", "mean28", "mean56",
    "med7", "med14", "med28", "std7", "std14", "std28",
    "trend7", "trend14", "trend28", "route_share", "network", "profile_daily",
    "weekend", "day_off", "preholiday", "short_day", "holiday_id", "is_holiday",
    "quarter", "module", "summer", "school_share",
    "days_to_break", "days_from_break", "first_school_week", "working_saturday",
    "days_to_holiday", "days_from_holiday",
    "temp", "precip", "snow",
    "offices", "acc7", "acc30", "tram7",
]
HOUR_FEATURES = DAILY_FEATURES + ["hour", "morning_peak", "evening_peak", "weekday_peak", "base"]
DAILY_CATEGORICAL = ["route", "dow", "month", "horizon_bucket", "holiday_id"]
HOUR_CATEGORICAL = DAILY_CATEGORICAL + ["hour"]
MEMBER_NAMES = ("profile", "direct_hour", "daily_shape", "daily_shape_tweedie", "daily_buckets")
ANCHOR = MEMBER_NAMES[0]
MODEL_KEYS = ("daily_l1", "daily_tweedie", "shape", "direct") + tuple(f"bucket{i}" for i in range(len(horizon_ranges())))


def safe_div(num, den):
    out = np.ones_like(num, dtype=np.float32)
    mask = den > 1
    out[mask] = num[mask] / den[mask]
    return out


# ---------------------------------------------------------------- профиль


def profile_candidates():
    seen = set()
    keys = list(PROFILE_GRID)
    for values in itertools.product(*(PROFILE_GRID[key] for key in keys)):
        params = ProfileParams(**dict(zip(keys, values)))
        if params.shape_pool == 0:
            params = ProfileParams(params.weeks, params.decay, params.holidays)
        if params not in seen:
            seen.add(params)
            yield params


def sign_test_p(wins, total):
    return sum(comb(total, k) for k in range(wins, total + 1)) / 2 ** total


def select_profile(bundle):
    """Минимакс по окнам разработки: лучший худший выигрыш против champion, затем средний."""
    dev = {}
    for origin in DEV_ORIGINS:
        first, last = window_bounds(origin)
        actual = actual_window(bundle, first, last)
        dev[origin] = (actual, wape(actual, champion_window(bundle, origin, first, last))[1])
    best = None
    for params in profile_candidates():
        gains = []
        for origin, (actual, champion_score) in dev.items():
            forecast = profile_window(bundle, origin, *window_bounds(origin), params)
            gains.append(wape(actual, forecast)[1] - champion_score)
        key = (min(gains), float(np.mean(gains)))
        if best is None or key > best[0]:
            best = (key, params)
    params = best[1]
    best = (float(np.mean([dev[o][1] for o in DEV_ORIGINS])) + best[0][1], params)
    windows = list(DEV_ORIGINS) + [LOCK_ORIGIN] + list(CHECK_ORIGINS)
    rows = []
    for origin in windows:
        first, last = window_bounds(origin)
        actual = actual_window(bundle, first, last)
        rows.append({
            "origin": origin.isoformat(),
            "role": "dev" if origin in DEV_ORIGINS else ("lock" if origin == LOCK_ORIGIN else "check"),
            "champion": wape(actual, champion_window(bundle, origin, first, last))[1],
            "profile": wape(actual, profile_window(bundle, origin, first, last, params))[1],
        })
    wins = sum(row["profile"] > row["champion"] + 1e-9 for row in rows)
    p_value = sign_test_p(wins, len(rows))
    lock = next(row for row in rows if row["role"] == "lock")
    accepted = params != CHAMPION_PARAMS and p_value <= SIGN_TEST_ALPHA and lock["profile"] >= lock["champion"]
    return {
        "params": params if accepted else CHAMPION_PARAMS,
        "searched": params,
        "dev_score": best[0],
        "accepted": accepted,
        "wins": wins,
        "windows": len(rows),
        "sign_test_p": p_value,
        "rows": rows,
    }


# ---------------------------------------------------------------- признаки


class OriginState:
    def __init__(self, bundle, origin_pos, params):
        self.bundle = bundle
        self.origin_pos = origin_pos
        daily = bundle["daily"]
        hist = daily[:, : origin_pos + 1]
        self.level1 = hist[:, -1].astype(np.float32)
        self.mean, self.median, self.std = {}, {}, {}
        for width in (7, 14, 28, 56):
            block = hist[:, max(0, hist.shape[1] - width):]
            self.mean[width] = block.mean(axis=1).astype(np.float32)
            self.median[width] = np.median(block, axis=1).astype(np.float32)
            self.std[width] = block.std(axis=1).astype(np.float32)

        def mean_ending(width, end):
            if end < 0:
                return np.zeros(len(ROUTES), np.float32)
            return daily[:, max(0, end - width + 1): end + 1].mean(axis=1).astype(np.float32)

        self.trend = {w: safe_div(mean_ending(w, origin_pos), mean_ending(w, origin_pos - w)) for w in (7, 14, 28)}
        network = float(self.mean[14].sum())
        self.network = np.float32(network)
        self.share = self.mean[14] / max(network, 1.0)
        days = bundle["days"]
        self.dow_lags = {}
        for dow in range(7):
            pos = origin_pos
            while pos >= 0 and days[pos].weekday() != dow:
                pos -= 1
            lags = [daily[:, cursor].astype(np.float32) for cursor in range(pos, -1, -7)][:8]
            while len(lags) < 8:
                lags.append(np.zeros(len(ROUTES), np.float32))
            self.dow_lags[dow] = lags
        self.profile = ProfileState(bundle, origin_pos, params)
        self._days = {}
        self.office = bundle["office"][:, origin_pos]
        self.clim = self._climatology()

    def profile_day(self, pos):
        if pos not in self._days:
            self._days[pos] = self.profile.day(pos)
        return self._days[pos]

    def _climatology(self):
        bundle = self.bundle
        shape = (len(ROUTES), 12, 24)
        temp, precip, snow, count = (np.zeros(shape) for _ in range(4))
        seen = bundle["seen_weather"]
        for pos in range(self.origin_pos + 1):
            month = bundle["days"][pos].month - 1
            mask = seen[:, pos, :]
            temp[:, month] += bundle["weather"]["temp"][:, pos] * mask
            precip[:, month] += bundle["weather"]["precip"][:, pos] * mask
            snow[:, month] += bundle["weather"]["snowfall"][:, pos] * mask
            count[:, month] += mask
        totals = [arr.sum(axis=1) for arr in (count, temp, precip, snow)]
        for month in range(12):
            missing = count[:, month] < 1
            count[:, month] = np.where(missing, np.maximum(totals[0], 1), count[:, month])
            temp[:, month] = np.where(missing, totals[1], temp[:, month])
            precip[:, month] = np.where(missing, totals[2], precip[:, month])
            snow[:, month] = np.where(missing, totals[3], snow[:, month])
        return {
            "temp": (temp / count).astype(np.float32),
            "precip": (precip / count).astype(np.float32),
            "snow": (snow / count).astype(np.float32),
        }


def feature_rows(bundle, state, target_pos):
    cal = bundle["calendar"]
    day = bundle["days"][target_pos]
    horizon = target_pos - state.origin_pos
    dow = day.weekday()
    clim_month = day.month - 1
    base = state.profile_day(target_pos)
    profile_totals = base.sum(axis=1)
    lags = state.dow_lags[dow]
    workday = cal["weekend"][target_pos] == 0 and cal["is_holiday"][target_pos] == 0
    daily_rows, hour_rows = [], []
    for route in range(len(ROUTES)):
        row = [
            ROUTES[route], dow, day.month, horizon, horizon_bucket(horizon),
            lags[0][route], lags[1][route], lags[2][route], lags[3][route], lags[7][route],
            state.level1[route], state.mean[7][route], state.mean[14][route], state.mean[28][route], state.mean[56][route],
            state.median[7][route], state.median[14][route], state.median[28][route],
            state.std[7][route], state.std[14][route], state.std[28][route],
            state.trend[7][route], state.trend[14][route], state.trend[28][route],
            state.share[route], state.network, float(profile_totals[route]),
            cal["weekend"][target_pos], cal["day_off"][target_pos], cal["preholiday"][target_pos],
            cal["short_day"][target_pos], cal["holiday_id"][target_pos], cal["is_holiday"][target_pos],
            cal["quarter"][target_pos], cal["module"][target_pos], cal["summer"][target_pos], cal["share"][target_pos],
            cal["days_to_break"][target_pos], cal["days_from_break"][target_pos],
            cal["first_school_week"][target_pos], cal["working_saturday"][target_pos],
            cal["days_to_holiday"][target_pos], cal["days_from_holiday"][target_pos],
            float(state.clim["temp"][route, clim_month].mean()),
            float(state.clim["precip"][route, clim_month].mean()),
            float(state.clim["snow"][route, clim_month].mean()),
            *state.office[route],
        ]
        daily_rows.append(row)
        for hour in range(24):
            morning = int(hour in bundle["morning_peak"])
            evening = int(hour in bundle["evening_peak"])
            hour_row = row + [hour, morning, evening, int((morning or evening) and workday), float(base[route, hour])]
            temp_col = DAILY_FEATURES.index("temp")
            hour_row[temp_col: temp_col + 3] = [
                float(state.clim["temp"][route, clim_month, hour]),
                float(state.clim["precip"][route, clim_month, hour]),
                float(state.clim["snow"][route, clim_month, hour]),
            ]
            hour_rows.append(hour_row)
    return daily_rows, hour_rows


def training_origins(label_end):
    origin = START + timedelta(days=MAX_LAG_DAYS - 1)
    while origin < label_end:
        yield origin
        origin += timedelta(days=TRAIN_STEP_DAYS)


def collect(bundle, label_end, params):
    label_pos = bundle["index"][label_end]
    daily_rows, hour_rows, daily_y, hour_y = [], [], [], []
    for origin in training_origins(label_end):
        origin_pos = bundle["index"][origin]
        state = OriginState(bundle, origin_pos, params)
        for target_pos in range(origin_pos + 1, min(label_pos, origin_pos + HORIZON) + 1):
            rows_d, rows_h = feature_rows(bundle, state, target_pos)
            daily_rows += rows_d
            hour_rows += rows_h
            daily_y += [float(v) for v in bundle["daily"][:, target_pos]]
            hour_y += [float(v) for v in bundle["series"][:, target_pos].reshape(-1)]
    daily_x = np.asarray(daily_rows, np.float32)
    hour_x = np.asarray(hour_rows, np.float32)
    return daily_x, np.asarray(daily_y, np.float32), hour_x, np.asarray(hour_y, np.float32)


def design_for_window(bundle, origin, first, last, params):
    state = OriginState(bundle, bundle["index"][origin], params)
    daily_rows, hour_rows = [], []
    for day in daterange(first, last):
        rows_d, rows_h = feature_rows(bundle, state, bundle["index"][day])
        daily_rows += rows_d
        hour_rows += rows_h
    return np.asarray(daily_rows, np.float32), np.asarray(hour_rows, np.float32)


# ---------------------------------------------------------------- модели


def fit_regressor(frame, target, feature_names, categorical, objective, weight=None):
    train = lgb.Dataset(
        frame,
        label=target,
        weight=weight,
        feature_name=feature_names,
        categorical_feature=[feature_names.index(name) for name in categorical],
        free_raw_data=False,
    )
    params = {"objective": objective, "verbosity": -1, "seed": 7, "deterministic": True, "num_threads": 4}
    return lgb.train(params, train, num_boost_round=MAX_ROUNDS)


def fit_models(bundle, label_end, params):
    daily_x, daily_y, hour_x, hour_y = collect(bundle, label_end, params)
    print(f"fit through {label_end.isoformat()} daily {len(daily_x)} hourly {len(hour_x)}", flush=True)
    models = {
        "daily_l1": fit_regressor(daily_x, daily_y, DAILY_FEATURES, DAILY_CATEGORICAL, "regression_l1"),
        "daily_tweedie": fit_regressor(daily_x, np.maximum(daily_y, 0), DAILY_FEATURES, DAILY_CATEGORICAL, "tweedie"),
    }
    horizon_col = DAILY_FEATURES.index("horizon")
    for i, (low, high) in enumerate(horizon_ranges()):
        mask = (daily_x[:, horizon_col] >= low) & (daily_x[:, horizon_col] <= high)
        models[f"bucket{i}"] = (
            fit_regressor(daily_x[mask], daily_y[mask], DAILY_FEATURES, DAILY_CATEGORICAL, "regression_l1")
            if int(mask.sum()) >= MIN_BUCKET_ROWS else None
        )
    day_total = np.repeat(daily_y, 24)
    positive = day_total > 0
    share = np.zeros(len(hour_y), np.float32)
    share[positive] = hour_y[positive] / day_total[positive]
    models["shape"] = fit_regressor(
        hour_x[positive], share[positive], HOUR_FEATURES, HOUR_CATEGORICAL, "regression_l1", weight=day_total[positive],
    )
    models["direct"] = fit_regressor(hour_x, hour_y, HOUR_FEATURES, HOUR_CATEGORICAL, "regression_l1")
    return models


def raw_predictions(models, daily_x, hour_x, rounds, cache=None):
    """Прогноз каждой модели на числе итераций rounds[key]."""
    cache = {} if cache is None else cache

    def predict(key, frame, mask=None):
        slot = (key, rounds[key])
        if slot not in cache:
            cache[slot] = models[key].predict(frame if mask is None else frame[mask], num_iteration=rounds[key])
        return cache[slot]

    out = {
        "daily_l1": predict("daily_l1", daily_x),
        "daily_tweedie": predict("daily_tweedie", daily_x),
        "shape": predict("shape", hour_x),
        "direct": predict("direct", hour_x),
    }
    horizon = daily_x[:, DAILY_FEATURES.index("horizon")]
    bucket = np.array(out["daily_l1"], copy=True)
    for i, (low, high) in enumerate(horizon_ranges()):
        mask = (horizon >= low) & (horizon <= high)
        if models[f"bucket{i}"] is not None and np.any(mask):
            bucket[mask] = predict(f"bucket{i}", daily_x, mask)
    out["buckets"] = bucket
    return out


def expand_daily(daily_pred, share_pred):
    totals = np.maximum(daily_pred, 0).reshape(-1, len(ROUTES))
    shares = np.clip(share_pred, 0, None).reshape(totals.shape[0], len(ROUTES), 24)
    denom = shares.sum(axis=2, keepdims=True)
    empty = denom[:, :, 0] <= 1e-6
    denom[denom <= 1e-6] = 1
    shares = shares / denom
    shares[empty] = 1.0 / 24
    return totals[:, :, None] * shares


def members_from_raw(raw, profile, n_days):
    shape = raw["shape"]
    return {
        "profile": profile,
        "direct_hour": np.maximum(raw["direct"], 0).reshape(n_days, len(ROUTES), 24),
        "daily_shape": expand_daily(raw["daily_l1"], shape),
        "daily_shape_tweedie": expand_daily(raw["daily_tweedie"], shape),
        "daily_buckets": expand_daily(raw["buckets"], shape),
    }


def select_rounds(dev_items):
    """Число итераций каждой модели — минимум суммарной ошибки по окнам разработки."""
    rounds = {key: MAX_ROUNDS for key in MODEL_KEYS}

    def total_error(update, member):
        absolute = 0.0
        for item in dev_items:
            trial = dict(rounds, **update)
            raw = raw_predictions(item["models"], item["daily_x"], item["hour_x"], trial, item.setdefault("cache", {}))
            forecast = members_from_raw(raw, item["profile"], item["n_days"])[member]
            absolute += float(np.abs(forecast - item["actual"]).sum())
        return absolute

    def shape_error(k):
        absolute = 0.0
        for item in dev_items:
            shares = raw_predictions(item["models"], item["daily_x"], item["hour_x"], dict(rounds, shape=k),
                                     item.setdefault("cache", {}))["shape"]
            forecast = expand_daily(item["actual"].sum(axis=2).reshape(-1), shares)
            absolute += float(np.abs(forecast - item["actual"]).sum())
        return absolute

    rounds["shape"] = min(ROUND_GRID, key=shape_error)
    rounds["direct"] = min(ROUND_GRID, key=lambda k: total_error({"direct": k}, "direct_hour"))
    rounds["daily_l1"] = min(ROUND_GRID, key=lambda k: total_error({"daily_l1": k}, "daily_shape"))
    rounds["daily_tweedie"] = min(ROUND_GRID, key=lambda k: total_error({"daily_tweedie": k}, "daily_shape_tweedie"))
    for i in range(len(horizon_ranges())):
        key = f"bucket{i}"
        rounds[key] = min(ROUND_GRID, key=lambda k: total_error({key: k}, "daily_buckets"))
    return rounds


# ---------------------------------------------------------------- ансамбль


def simplex(count, step=WEIGHT_STEP):
    units = int(round(1 / step))
    current = [0] * count

    def walk(dim, left):
        if dim == count - 1:
            current[dim] = left
            yield tuple(value / units for value in current)
            return
        for value in range(left + 1):
            current[dim] = value
            yield from walk(dim + 1, left - value)

    yield from walk(0, units)


def blend(members, weights, names=MEMBER_NAMES):
    out = np.zeros_like(members[ANCHOR])
    for name, weight in zip(names, weights):
        if weight:
            out += weight * members[name]
    return np.maximum(out, 0)


def best_weights(scored, names=MEMBER_NAMES):
    best = None
    for weights in simplex(len(names)):
        absolute = denom = 0.0
        for item in scored:
            absolute += np.abs(blend(item["members"], weights, names) - item["actual"]).sum()
            denom += item["actual"].sum()
        score = max(0.0, 1.0 - float(absolute / max(denom, 1)))
        if best is None or score > best[0]:
            best = (score, weights)
    return best


def leave_one_out(scored, names=MEMBER_NAMES):
    rows = []
    for hold in range(len(scored)):
        _score, weights = best_weights([item for pos, item in enumerate(scored) if pos != hold], names)
        actual = scored[hold]["actual"]
        rows.append({
            "origin": scored[hold]["origin"],
            "weights": [float(weight) for weight in weights],
            "anchor": wape(actual, scored[hold]["members"][ANCHOR])[1],
            "blend": wape(actual, blend(scored[hold]["members"], weights, names))[1],
        })
    return rows


def evaluate_gate(bundle, dev_rows, lock_row, weights, names=MEMBER_NAMES):
    anchor_scores, blend_scores, details = [], [], []
    wins, worst_drop, workday_ok = 0, 0.0, True
    for item in dev_rows:
        mixed_forecast = blend(item["members"], weights, names)
        anchor = wape(item["actual"], item["members"][ANCHOR])[1]
        mixed = wape(item["actual"], mixed_forecast)[1]
        anchor_scores.append(anchor)
        blend_scores.append(mixed)
        wins += int(mixed > anchor + 1e-6)
        worst_drop = max(worst_drop, anchor - mixed)
        work_anchor = slice_report(bundle, item["first"], item["actual"], item["members"][ANCHOR])["workday"]["score"]
        work_mixed = slice_report(bundle, item["first"], item["actual"], mixed_forecast)["workday"]["score"]
        workday_ok = workday_ok and work_mixed + 1e-9 >= work_anchor - GATE_WORKDAY_TOL
        details.append({"origin": item["origin"], "anchor": anchor, "blend": mixed,
                        "workday_anchor": work_anchor, "workday_blend": work_mixed})
    lock_anchor = wape(lock_row["actual"], lock_row["members"][ANCHOR])[1]
    lock_blend = wape(lock_row["actual"], blend(lock_row["members"], weights, names))[1]
    mean_gain = float(np.mean(blend_scores) - np.mean(anchor_scores))
    passed = (
        mean_gain >= GATE_MIN_GAIN and wins >= GATE_MIN_WINS and worst_drop <= GATE_MAX_DROP
        and lock_blend > lock_anchor and workday_ok
    )
    return {"passed": passed, "mean_gain": mean_gain, "wins": wins, "worst_drop": worst_drop,
            "lock_anchor": lock_anchor, "lock_blend": lock_blend, "workday_ok": workday_ok, "origins": details}


def nlinear_member(bundle, origin, first, last, params):
    """NLinear на дневных суммах; форма часа берётся из принятого профиля."""
    origin_pos = bundle["index"][origin]
    daily = bundle["daily"]
    base = profile_window(bundle, origin, first, last, params)
    base_daily = base.sum(axis=2)
    totals = np.array(base_daily, copy=True)
    for route in range(len(ROUTES)):
        xs, ys = [], []
        for end in range(NLINEAR_LOOKBACK - 1, origin_pos - HORIZON + 1):
            hist = daily[route, end - NLINEAR_LOOKBACK + 1: end + 1]
            xs.append(hist - hist[-1])
            ys.append(daily[route, end + 1: end + 1 + HORIZON] - hist[-1])
        if len(xs) < NLINEAR_LOOKBACK:
            continue
        coef, *_ = np.linalg.lstsq(np.asarray(xs, np.float64), np.asarray(ys, np.float64), rcond=1e-2)
        hist = daily[route, origin_pos - NLINEAR_LOOKBACK + 1: origin_pos + 1].astype(np.float64)
        totals[:, route] = np.maximum((hist - hist[-1]) @ coef + hist[-1], 0)[: base.shape[0]]
    shape = base / np.maximum(base_daily[:, :, None], 1e-6)
    shape[base_daily <= 1e-6] = 1.0 / 24
    return totals[:, :, None] * shape


# ---------------------------------------------------------------- запуск


def predictions_to_lookup(first, forecast):
    lookup = {}
    for row, day in enumerate(daterange(first, first + timedelta(days=forecast.shape[0] - 1))):
        for route_i, route in enumerate(ROUTES):
            for hour in range(24):
                lookup[(route, day.isoformat(), hour)] = float(forecast[row, route_i, hour])
    return lookup


def window_item(bundle, origin, first, last, params):
    models = fit_models(bundle, origin, params)
    daily_x, hour_x = design_for_window(bundle, origin, first, last, params)
    return {
        "origin": origin.isoformat(),
        "first": first,
        "models": models,
        "daily_x": daily_x,
        "hour_x": hour_x,
        "n_days": (last - first).days + 1,
        "actual": actual_window(bundle, first, last),
        "profile": profile_window(bundle, origin, first, last, params),
        "champion": champion_window(bundle, origin, first, last),
    }


def main():
    digest = freeze_champion()
    bundle = load_bundle()
    champion = champion_window(bundle, OBSERVED_END, FORECAST_START, FORECAST_END)
    lookup = predictions_to_lookup(FORECAST_START, champion)
    with (RESULTS / "champion-087560.csv").open(encoding="utf-8", newline="") as handle:
        gaps = [abs(float(row["prediction"]) - lookup[(int(row["route"]), row["date"], int(row["hour"]))])
                for row in csv.DictReader(handle, delimiter=";")]
    print(f"champion hash {digest[:12]} reproduce MAE {float(np.mean(gaps)):.3f}", flush=True)
    print(f"peaks morning {sorted(bundle['morning_peak'])} evening {sorted(bundle['evening_peak'])}", flush=True)

    selection = select_profile(bundle)
    params = selection["params"]
    print(f"profile searched {selection['searched']} dev {selection['dev_score']:.4f} "
          f"wins {selection['wins']}/{selection['windows']} p={selection['sign_test_p']:.4f} "
          f"accepted {selection['accepted']}", flush=True)
    for row in selection["rows"]:
        print(f"  {row['role']:5s} {row['origin']} champion {row['champion']:.4f} profile {row['profile']:.4f}", flush=True)

    dev_items = []
    for origin in DEV_ORIGINS:
        first, last = window_bounds(origin)
        print(f"backtest {origin.isoformat()} -> {first.isoformat()}..{last.isoformat()}", flush=True)
        dev_items.append(window_item(bundle, origin, first, last, params))
    rounds = select_rounds(dev_items)
    print("rounds", rounds, flush=True)

    oof_dir = RESULTS / f"oof-{OUT_NAME}"
    oof_dir.mkdir(exist_ok=True)
    scored = []
    for item in dev_items:
        raw = raw_predictions(item["models"], item["daily_x"], item["hour_x"], rounds, item.get("cache"))
        members = members_from_raw(raw, item["profile"], item["n_days"])
        scored.append({"origin": item["origin"], "first": item["first"], "actual": item["actual"], "members": members})
        np.savez_compressed(oof_dir / f"{item['origin']}.npz", actual=item["actual"], champion=item["champion"], **members)
        line = " ".join(f"{name} {wape(item['actual'], members[name])[1]:.4f}" for name in MEMBER_NAMES)
        print(f"  {item['origin']} champion {wape(item['actual'], item['champion'])[1]:.4f} {line}", flush=True)
        scored[-1]["slices"] = {name: slice_report(bundle, item["first"], item["actual"], members[name]) for name in MEMBER_NAMES}
        del item["models"], item["cache"]

    insample_score, weights = best_weights(scored)
    loo = leave_one_out(scored)
    print(f"dev blend {insample_score:.4f} weights {weights}", flush=True)

    lock_first, lock_last = window_bounds(LOCK_ORIGIN)
    print(f"locked {LOCK_ORIGIN.isoformat()}", flush=True)
    lock_item = window_item(bundle, LOCK_ORIGIN, lock_first, lock_last, params)
    lock_members = members_from_raw(
        raw_predictions(lock_item["models"], lock_item["daily_x"], lock_item["hour_x"], rounds),
        lock_item["profile"], lock_item["n_days"],
    )
    del lock_item["models"]
    lock_row = {"actual": lock_item["actual"], "members": lock_members}
    np.savez_compressed(oof_dir / f"{LOCK_ORIGIN.isoformat()}.npz", actual=lock_item["actual"],
                        champion=lock_item["champion"], **lock_members)
    gate = evaluate_gate(bundle, scored, lock_row, weights)
    print(f"gate passed {gate['passed']} gain {gate['mean_gain']:.4f} "
          f"lock {gate['lock_anchor']:.4f} -> {gate['lock_blend']:.4f}", flush=True)

    anchor_only = tuple(1.0 if name == ANCHOR else 0.0 for name in MEMBER_NAMES)
    final_names, tree_gate, tree_weights, nlinear_weights = MEMBER_NAMES, gate, weights, None
    accepted = weights if gate["passed"] else anchor_only
    if gate["passed"]:
        ext = MEMBER_NAMES + ("nlinear",)
        for item, origin in zip(scored, DEV_ORIGINS):
            item["members"]["nlinear"] = nlinear_member(bundle, origin, *window_bounds(origin), params)
        lock_members["nlinear"] = nlinear_member(bundle, LOCK_ORIGIN, lock_first, lock_last, params)
        _score, nlinear_weights = best_weights(scored, ext)
        ext_gate = evaluate_gate(bundle, scored, lock_row, nlinear_weights, ext)
        print(f"nlinear gate passed {ext_gate['passed']}", flush=True)
        if ext_gate["passed"]:
            final_names, accepted, gate = ext, nlinear_weights, ext_gate

    print(f"production fit through {OBSERVED_END.isoformat()}", flush=True)
    production = window_item(bundle, OBSERVED_END, FORECAST_START, FORECAST_END, params)
    forecast = members_from_raw(
        raw_predictions(production["models"], production["daily_x"], production["hour_x"], rounds),
        production["profile"], production["n_days"],
    )
    if "nlinear" in final_names:
        forecast["nlinear"] = nlinear_member(bundle, OBSERVED_END, FORECAST_START, FORECAST_END, params)
    final = blend(forecast, accepted, final_names)
    path = RESULTS / f"{OUT_NAME}.csv"
    rows, missing = write_submission(predictions_to_lookup(FORECAST_START, final), path)
    lock_accepted = blend(lock_members, accepted, final_names)
    report = {
        "champion_sha256": digest,
        "submission": str(path),
        "rows": rows,
        "missing": missing,
        "horizon": HORIZON,
        "dev_origins": [origin.isoformat() for origin in DEV_ORIGINS],
        "lock_origin": LOCK_ORIGIN.isoformat(),
        "check_origins": [origin.isoformat() for origin in CHECK_ORIGINS],
        "peaks": {"morning": sorted(bundle["morning_peak"]), "evening": sorted(bundle["evening_peak"])},
        "profile_selection": {
            **{key: value for key, value in selection.items() if key not in ("params", "searched")},
            "params": asdict(params),
            "searched": asdict(selection["searched"]),
            "champion_params": asdict(CHAMPION_PARAMS),
        },
        "rounds": rounds,
        "weights": {name: float(weight) for name, weight in zip(final_names, accepted)},
        "searched_weights": {name: float(weight) for name, weight in zip(MEMBER_NAMES, tree_weights)},
        "nlinear_weights": None if nlinear_weights is None else [float(w) for w in nlinear_weights],
        "insample_blend_score": insample_score,
        "leave_one_out": loo,
        "gate": gate,
        "tree_gate": tree_gate,
        "fold_slices": {item["origin"]: item["slices"] for item in scored},
        "locked_slices": {
            "champion": slice_report(bundle, lock_first, lock_row["actual"], lock_item["champion"]),
            "accepted": slice_report(bundle, lock_first, lock_row["actual"], lock_accepted),
            **{name: slice_report(bundle, lock_first, lock_row["actual"], lock_members[name]) for name in final_names},
        },
        "forecast_vs_champion": {
            "mean_abs_diff": float(np.abs(final - champion).mean()),
            "total_ratio": float(final.sum() / max(champion.sum(), 1)),
        },
    }
    report_path = RESULTS / f"{OUT_NAME}-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"wrote {path} profile_accepted={selection['accepted']} gate={gate['passed']}", flush=True)


if __name__ == "__main__":
    main()
