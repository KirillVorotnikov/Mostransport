"""Leakage-safe LSTM + TCN forecast for the MTTEХ tram challenge.

The script is intentionally self-contained so that it can be uploaded together
with the Kaggle notebook.  It reads the raw ``train.csv`` and ``test.csv`` in
chunks, aggregates successful validations to an hourly route grid, joins the
``mstrans-enrichment`` dataset, performs a small reproducible random search,
and writes a full ``submission.csv``.

The hidden competition period is November--December 2025.  The enrichment
dataset contains realized weather and accident aggregates for that period, so
all features for dates after a cutoff are replaced with climatology or a
frozen last-known value.  This is important: using the archive values directly
would leak information from the hidden target period.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import re
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
ROUTE_TO_INDEX = {route: index for index, route in enumerate(ROUTES)}
START = date(2025, 1, 1)
TRAIN_END = date(2025, 8, 31)
TEST_END = date(2025, 10, 31)
FORECAST_END = date(2025, 12, 31)
N_DAYS = (FORECAST_END - START).days + 1
HORIZON = 61 * 24

HOLIDAY_NAMES = [
    "newyear",
    "christmas",
    "defender",
    "womens",
    "labour",
    "victory",
    "russia",
    "unity",
    "transfer",
]
HOLIDAY_TO_INDEX = {name: index for index, name in enumerate(HOLIDAY_NAMES)}

SNOW_CODES = {71, 73, 75, 77, 85, 86}
RAIN_CODES = {
    51,
    53,
    55,
    56,
    57,
    61,
    63,
    65,
    66,
    67,
    80,
    81,
    82,
    95,
    96,
    99,
}

FEATURE_VARIANTS = {
    "calendar_static": {"weather": False, "accidents": False},
    "calendar_weather": {"weather": True, "accidents": False},
    "calendar_weather_accidents": {"weather": True, "accidents": True},
}


def set_seed(seed: int) -> None:
    """Make trial and final training runs repeatable."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_date(value: Any) -> date:
    text = str(value)[:10]
    return date.fromisoformat(text)


def date_index(day: date) -> int:
    return (day - START).days


def day_at(index: int) -> date:
    return START + timedelta(days=int(index))


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def daterange(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (date,)):
        return value.isoformat()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(as_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def find_named_file(root: Path, name: str) -> Path:
    direct_candidates = [
        root / name,
        root / "datasets" / "gvfadeev" / "mstrans-enrichment" / name,
        root / "gvfadeev" / "mstrans-enrichment" / name,
        root / "mstrans-enrichment" / name,
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate
    matches = sorted(root.rglob(name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {name!r} below {root}")


def resolve_mstrans_root(input_root: Path) -> Path:
    candidates = [
        input_root / "datasets" / "kirillvorotnikov2007" / "mstrans",
        input_root / "kirillvorotnikov2007" / "mstrans",
        input_root / "mstrans",
    ]
    for candidate in candidates:
        if (candidate / "train.csv").exists() and (candidate / "test.csv").exists():
            return candidate
    for train_path in sorted(input_root.rglob("train.csv")):
        if (train_path.parent / "test.csv").exists():
            return train_path.parent
    raise FileNotFoundError(
        f"Could not find train.csv and test.csv below {input_root}"
    )


def parse_timestamp_hour(values: pd.Series) -> pd.Series:
    text = (
        values.astype("string")
        .fillna("")
        .str.strip()
        .str.replace("T", " ", regex=False)
    )
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    iso = text.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    dmy = text.str.match(r"^\d{2}\.\d{2}\.\d{4}", na=False)
    if bool(iso.any()):
        parsed.loc[iso] = pd.to_datetime(
            text.loc[iso].str.slice(0, 13),
            format="%Y-%m-%d %H",
            errors="coerce",
        )
    if bool(dmy.any()):
        parsed.loc[dmy] = pd.to_datetime(
            text.loc[dmy].str.slice(0, 13),
            format="%d.%m.%Y %H",
            errors="coerce",
        )
    unresolved = parsed.isna() & text.ne("")
    if bool(unresolved.any()):
        try:
            parsed.loc[unresolved] = pd.to_datetime(
                text.loc[unresolved],
                errors="coerce",
                format="mixed",
            ).dt.floor("h")
        except (TypeError, ValueError):
            parsed.loc[unresolved] = pd.to_datetime(
                text.loc[unresolved],
                errors="coerce",
            ).dt.floor("h")
    return parsed


def aggregate_raw_file(path: Path, history: np.ndarray) -> tuple[int, int]:
    """Aggregate one raw validation file without loading it into memory."""

    rows_seen = 0
    boardings_seen = 0
    usecols = ["tran_date_time", "validation_result", "ngpt_route"]
    reader = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        usecols=usecols,
        dtype="string",
        chunksize=1_000_000,
        on_bad_lines="skip",
    )
    start_stamp = pd.Timestamp(START)
    for chunk_no, chunk in enumerate(reader, start=1):
        rows_seen += len(chunk)
        valid = (
            chunk["validation_result"]
            .fillna("")
            .str.strip()
            .isin(["1", "1.0"])
        )
        part = chunk.loc[valid, ["tran_date_time", "ngpt_route"]]
        if part.empty:
            continue

        route_values = pd.to_numeric(
            part["ngpt_route"].str.extract(r"(\d+)", expand=False),
            errors="coerce",
        )
        times = parse_timestamp_hour(part["tran_date_time"])
        day_values = (times - start_stamp).dt.total_seconds().div(86400.0)
        day_values = day_values.round().astype("Int64")
        hour_values = times.dt.hour.astype("Int64")
        route_index = route_values.map(ROUTE_TO_INDEX).astype("Int64")
        good = (
            times.notna()
            & day_values.notna()
            & hour_values.notna()
            & route_index.notna()
            & day_values.between(0, N_DAYS - 1)
        )
        if not bool(good.any()):
            continue

        grouped_frame = pd.DataFrame(
            {
                "day": day_values.loc[good].astype(np.int16).to_numpy(),
                "route": route_index.loc[good].astype(np.int8).to_numpy(),
                "hour": hour_values.loc[good].astype(np.int8).to_numpy(),
            }
        )
        grouped = grouped_frame.groupby(["day", "route", "hour"]).size()
        for (day, route, hour), count in grouped.items():
            history[int(day), int(route), int(hour)] += float(count)
            boardings_seen += int(count)
        if chunk_no == 1 or chunk_no % 5 == 0:
            print(
                f"{path.name}: chunks={chunk_no}, rows={rows_seen:,}, "
                f"successful={boardings_seen:,}",
                flush=True,
            )
    print(
        f"{path.name}: rows={rows_seen:,}, successful={boardings_seen:,}",
        flush=True,
    )
    return rows_seen, boardings_seen


def load_history(input_root: Path) -> np.ndarray:
    data_root = resolve_mstrans_root(input_root)
    history = np.zeros((N_DAYS, len(ROUTES), 24), dtype=np.float32)
    aggregate_raw_file(data_root / "train.csv", history)
    aggregate_raw_file(data_root / "test.csv", history)
    return history


@dataclass
class EnrichmentTables:
    calendar: np.ndarray
    holiday_ids: np.ndarray
    weather: np.ndarray
    route_day: np.ndarray


def load_enrichment(input_root: Path) -> EnrichmentTables:
    """Load enrichment tables but defer cutoff-safe projection to the provider."""

    holidays_path = find_named_file(input_root, "holidays_2025.csv")
    weather_path = find_named_file(input_root, "weather_hourly.csv")
    route_day_path = find_named_file(input_root, "route_day.csv")

    calendar = np.zeros((N_DAYS, 4 + len(HOLIDAY_NAMES)), dtype=np.float32)
    holiday_ids = np.zeros(N_DAYS, dtype=np.int16)
    holidays = pd.read_csv(holidays_path, encoding="utf-8-sig")
    for row in holidays.itertuples(index=False):
        try:
            index = date_index(parse_date(getattr(row, "date")))
        except (ValueError, TypeError):
            continue
        if not 0 <= index < N_DAYS:
            continue
        name = str(getattr(row, "holiday", "none"))
        if name in {"nan", "None", ""}:
            name = "none"
        values = [
            finite_float(getattr(row, "day_off", 0)),
            finite_float(getattr(row, "weekend", 0)),
            finite_float(getattr(row, "short_day", 0)),
            finite_float(getattr(row, "preholiday", 0)),
        ]
        calendar[index, :4] = values
        if name in HOLIDAY_TO_INDEX:
            holiday_ids[index] = HOLIDAY_TO_INDEX[name] + 1
            calendar[index, 4 + HOLIDAY_TO_INDEX[name]] = 1.0

    weather = np.zeros(
        (N_DAYS, len(ROUTES), 24, 7),
        dtype=np.float32,
    )
    weather_frame = pd.read_csv(weather_path, encoding="utf-8-sig")
    for row in weather_frame.itertuples(index=False):
        try:
            route = int(getattr(row, "route"))
            index = date_index(parse_date(getattr(row, "date")))
            hour = int(getattr(row, "hour"))
        except (ValueError, TypeError):
            continue
        if route not in ROUTE_TO_INDEX or not 0 <= index < N_DAYS or not 0 <= hour < 24:
            continue
        route_index = ROUTE_TO_INDEX[route]
        temp = finite_float(getattr(row, "temp", 0))
        precip = finite_float(getattr(row, "precip", 0))
        wind = finite_float(getattr(row, "wind", 0))
        snowfall = finite_float(getattr(row, "snowfall", 0))
        code = int(finite_float(getattr(row, "code", 0)))
        snow = float(snowfall > 0 or code in SNOW_CODES)
        rain = float(not snow and (precip >= 0.2 or code in RAIN_CODES))
        fog = float(code in {45, 48})
        weather[index, route_index, hour] = [
            temp,
            precip,
            wind,
            snowfall,
            snow,
            rain,
            fog,
        ]

    route_day = np.zeros((N_DAYS, len(ROUTES), 5), dtype=np.float32)
    route_day_frame = pd.read_csv(route_day_path, encoding="utf-8-sig")
    for row in route_day_frame.itertuples(index=False):
        try:
            route = int(getattr(row, "route"))
            index = date_index(parse_date(getattr(row, "date")))
        except (ValueError, TypeError):
            continue
        if route not in ROUTE_TO_INDEX or not 0 <= index < N_DAYS:
            continue
        route_index = ROUTE_TO_INDEX[route]
        fields = ["offices", "offices_hit", "acc7", "acc30", "tram7"]
        route_day[index, route_index] = [
            finite_float(getattr(row, field, 0)) for field in fields
        ]
    return EnrichmentTables(calendar, holiday_ids, weather, route_day)


class SafeFeatureProvider:
    """Project archive enrichment to features available at a cutoff."""

    def __init__(self, tables: EnrichmentTables):
        self.tables = tables
        self.weather_cache: dict[int, np.ndarray] = {}
        self.route_day_cache: dict[tuple[int, bool], np.ndarray] = {}

    def safe_weather(self, cutoff: int) -> np.ndarray:
        if cutoff in self.weather_cache:
            return self.weather_cache[cutoff]

        raw = self.tables.weather
        safe = np.zeros_like(raw)
        allowed_days = list(range(max(0, cutoff + 1)))
        if not allowed_days:
            safe[:] = 0.0
            self.weather_cache[cutoff] = safe
            return safe

        all_days = np.asarray(allowed_days, dtype=np.int32)
        for route_index in range(len(ROUTES)):
            for hour in range(24):
                fallback = np.nanmean(raw[all_days, route_index, hour], axis=0)
                fallback = np.nan_to_num(fallback, nan=0.0)
                for day_index in range(N_DAYS):
                    if day_index <= cutoff:
                        safe[day_index, route_index, hour] = raw[
                            day_index, route_index, hour
                        ]
                        continue
                    target = day_at(day_index)
                    same_month_dow = [
                        candidate
                        for candidate in allowed_days
                        if day_at(candidate).month == target.month
                        and day_at(candidate).weekday() == target.weekday()
                    ]
                    candidates = same_month_dow or [
                        candidate
                        for candidate in allowed_days
                        if day_at(candidate).weekday() == target.weekday()
                    ]
                    source_days = np.asarray(candidates or allowed_days, dtype=np.int32)
                    value = np.nanmean(
                        raw[source_days, route_index, hour],
                        axis=0,
                    )
                    safe[day_index, route_index, hour] = np.nan_to_num(
                        value,
                        nan=0.0,
                    )
                if not np.isfinite(fallback).all():
                    safe[:, route_index, hour] = np.nan_to_num(
                        safe[:, route_index, hour],
                        nan=0.0,
                    )
        self.weather_cache[cutoff] = safe
        return safe

    def safe_route_day(self, cutoff: int, include_accidents: bool) -> np.ndarray:
        key = (cutoff, include_accidents)
        if key in self.route_day_cache:
            return self.route_day_cache[key]
        raw = self.tables.route_day
        safe = np.zeros_like(raw)
        cutoff = max(0, min(cutoff, N_DAYS - 1))
        static_offices = raw[cutoff, :, 0].copy()
        safe[:, :, 0] = static_offices[None, :]
        safe[: cutoff + 1] = raw[: cutoff + 1]
        if include_accidents:
            safe[cutoff + 1 :, :, 1:] = raw[cutoff, :, 1:][None, :, :]
        self.route_day_cache[key] = safe
        return safe


class FeatureAssembler:
    """Create cutoff-safe encoder and direct-horizon decoder tensors."""

    def __init__(
        self,
        history: np.ndarray,
        tables: EnrichmentTables,
        provider: SafeFeatureProvider,
        variant: str,
        baseline_kind: str,
    ):
        if variant not in FEATURE_VARIANTS:
            raise ValueError(f"Unknown feature variant: {variant}")
        self.history = history
        self.tables = tables
        self.provider = provider
        self.variant = variant
        self.variant_config = FEATURE_VARIANTS[variant]
        self.baseline_kind = baseline_kind
        self.feature_names = self._feature_names()

    def _feature_names(self) -> list[str]:
        names = [
            "log_boardings",
            "log_baseline",
            "log_lag_24h",
            "log_lag_168h",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "year_sin",
            "year_cos",
            "day_off",
            "weekend",
            "short_day",
            "preholiday",
        ]
        names.extend(f"holiday_{name}" for name in HOLIDAY_NAMES)
        if self.variant_config["weather"]:
            names.extend(
                [
                    "temp",
                    "precip",
                    "wind",
                    "snowfall",
                    "snow",
                    "rain",
                    "fog",
                ]
            )
        names.append("offices")
        if self.variant_config["accidents"]:
            names.extend(["offices_hit", "acc7", "acc30", "tram7"])
        return names

    def _weighted_average(self, values: Sequence[float]) -> float:
        if not values:
            return 0.0
        if self.baseline_kind == "mean":
            return float(np.mean(values))
        weights = np.asarray([0.75**index for index in range(len(values))])
        return float(np.dot(weights, np.asarray(values)) / weights.sum())

    def baseline(
        self,
        route_index: int,
        target_day: int,
        hour: int,
        cutoff: int,
    ) -> float:
        holiday_id = int(self.tables.holiday_ids[target_day])
        values: list[float] = []
        if holiday_id:
            cursor = target_day - 1
            while cursor >= 0 and len(values) < 6:
                if cursor <= cutoff and int(self.tables.holiday_ids[cursor]) == holiday_id:
                    values.append(float(self.history[cursor, route_index, hour]))
                cursor -= 1
            if len(values) >= 2:
                return self._weighted_average(values)

        values = []
        cursor = target_day - 7
        while cursor >= 0 and len(values) < 8:
            if cursor <= cutoff:
                values.append(float(self.history[cursor, route_index, hour]))
            cursor -= 7
        return self._weighted_average(values)

    def lagged(
        self,
        route_index: int,
        target_day: int,
        hour: int,
        step_days: int,
        cutoff: int,
    ) -> float:
        cursor = target_day - step_days
        while cursor >= 0:
            if cursor <= cutoff:
                return float(self.history[cursor, route_index, hour])
            cursor -= step_days
        return 0.0

    def vector(
        self,
        route_index: int,
        target_day: int,
        hour: int,
        cutoff: int,
        weather: np.ndarray,
        route_day: np.ndarray,
        include_target: bool,
    ) -> np.ndarray:
        boardings = (
            float(self.history[target_day, route_index, hour])
            if include_target and target_day <= cutoff
            else 0.0
        )
        base = self.baseline(route_index, target_day, hour, cutoff)
        lag_24 = self.lagged(route_index, target_day, hour, 1, cutoff)
        lag_168 = self.lagged(route_index, target_day, hour, 7, cutoff)
        current = day_at(target_day)
        day_of_year = current.timetuple().tm_yday - 1
        day_of_year_period = 365.0
        values: list[float] = [
            math.log1p(max(boardings, 0.0)),
            math.log1p(max(base, 0.0)),
            math.log1p(max(lag_24, 0.0)),
            math.log1p(max(lag_168, 0.0)),
            math.sin(2.0 * math.pi * hour / 24.0),
            math.cos(2.0 * math.pi * hour / 24.0),
            math.sin(2.0 * math.pi * current.weekday() / 7.0),
            math.cos(2.0 * math.pi * current.weekday() / 7.0),
            math.sin(2.0 * math.pi * day_of_year / day_of_year_period),
            math.cos(2.0 * math.pi * day_of_year / day_of_year_period),
        ]
        values.extend(self.tables.calendar[target_day].tolist())
        if self.variant_config["weather"]:
            values.extend(weather[target_day, route_index, hour].tolist())
        values.append(float(route_day[target_day, route_index, 0]))
        if self.variant_config["accidents"]:
            values.extend(route_day[target_day, route_index, 1:].tolist())
        if len(values) != len(self.feature_names):
            raise RuntimeError(
                f"Feature dimension mismatch: {len(values)} != {len(self.feature_names)}"
            )
        return np.asarray(values, dtype=np.float32)

    def build_base(
        self,
        cutoff_day: date,
        target_end: date,
        context: int,
    ) -> dict[str, Any]:
        cutoff = date_index(cutoff_day)
        target_end_index = min(date_index(target_end), N_DAYS - 1)
        weather = self.provider.safe_weather(cutoff)
        route_day = self.provider.safe_route_day(
            cutoff,
            self.variant_config["accidents"],
        )
        sequences = np.empty((len(ROUTES), context, len(self.feature_names)), dtype=np.float32)
        futures = np.empty(
            (len(ROUTES), HORIZON, len(self.feature_names)),
            dtype=np.float32,
        )
        targets = np.zeros((len(ROUTES), HORIZON), dtype=np.float32)
        baselines = np.zeros((len(ROUTES), HORIZON), dtype=np.float32)
        masks = np.zeros((len(ROUTES), HORIZON), dtype=bool)

        for route_index in range(len(ROUTES)):
            last_absolute_hour = cutoff * 24 + 23
            first_absolute_hour = last_absolute_hour - context + 1
            for position, absolute_hour in enumerate(
                range(first_absolute_hour, last_absolute_hour + 1)
            ):
                absolute_hour = max(0, absolute_hour)
                source_day, source_hour = divmod(absolute_hour, 24)
                sequences[route_index, position] = self.vector(
                    route_index,
                    source_day,
                    source_hour,
                    cutoff,
                    weather,
                    route_day,
                    include_target=True,
                )
            for horizon_index in range(HORIZON):
                target_day = cutoff + 1 + horizon_index // 24
                hour = horizon_index % 24
                if target_day >= N_DAYS:
                    futures[route_index, horizon_index] = 0.0
                    continue
                futures[route_index, horizon_index] = self.vector(
                    route_index,
                    target_day,
                    hour,
                    cutoff,
                    weather,
                    route_day,
                    include_target=False,
                )
                baselines[route_index, horizon_index] = self.baseline(
                    route_index,
                    target_day,
                    hour,
                    cutoff,
                )
                if target_day <= target_end_index:
                    targets[route_index, horizon_index] = self.history[
                        target_day,
                        route_index,
                        hour,
                    ]
                    masks[route_index, horizon_index] = True
        return {
            "sequences": sequences,
            "futures": futures,
            "targets": targets,
            "baselines": baselines,
            "masks": masks,
            "route_ids": np.arange(len(ROUTES), dtype=np.int64),
            "cutoff": cutoff_day,
        }

    def build_dataset(
        self,
        origins: Sequence[date],
        target_end: date,
        context: int,
    ) -> "RawForecast":
        bases = [
            self.build_base(origin, target_end, context)
            for origin in origins
        ]
        return RawForecast.from_bases(bases)


@dataclass
class RawForecast:
    sequences: np.ndarray
    futures: np.ndarray
    targets: np.ndarray
    baselines: np.ndarray
    masks: np.ndarray
    route_ids: np.ndarray
    cutoffs: list[date]

    @classmethod
    def from_bases(cls, bases: Sequence[dict[str, Any]]) -> "RawForecast":
        if not bases:
            raise ValueError("At least one forecast origin is required")
        return cls(
            sequences=np.concatenate([base["sequences"] for base in bases], axis=0),
            futures=np.concatenate([base["futures"] for base in bases], axis=0),
            targets=np.concatenate([base["targets"] for base in bases], axis=0),
            baselines=np.concatenate([base["baselines"] for base in bases], axis=0),
            masks=np.concatenate([base["masks"] for base in bases], axis=0),
            route_ids=np.concatenate([base["route_ids"] for base in bases], axis=0),
            cutoffs=[base["cutoff"] for base in bases],
        )


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, sequences: np.ndarray) -> "FeatureScaler":
        mean = sequences.reshape(-1, sequences.shape[-1]).mean(axis=0)
        std = sequences.reshape(-1, sequences.shape[-1]).std(axis=0)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(mean.astype(np.float32), std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def payload(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def scale_forecast(raw: RawForecast, scaler: FeatureScaler) -> RawForecast:
    future_shape = raw.futures.shape
    scaled_future = scaler.transform(raw.futures.reshape(-1, future_shape[-1]))
    return RawForecast(
        sequences=scaler.transform(raw.sequences.reshape(-1, raw.sequences.shape[-1])).reshape(
            raw.sequences.shape
        ),
        futures=scaled_future.reshape(future_shape),
        targets=raw.targets,
        baselines=raw.baselines,
        masks=raw.masks,
        route_ids=raw.route_ids,
        cutoffs=raw.cutoffs,
    )


class SequenceDataset(Dataset):
    """One item is a full route sequence; the encoder runs once for every horizon."""

    def __init__(self, raw: RawForecast):
        if not bool(raw.masks.any()):
            raise ValueError("Dataset has no valid target pairs")
        self.sequences = torch.from_numpy(raw.sequences)
        self.futures = torch.from_numpy(raw.futures)
        self.targets = torch.from_numpy(raw.targets)
        self.baselines = torch.from_numpy(raw.baselines)
        self.masks = torch.from_numpy(raw.masks.astype(np.bool_))
        self.route_ids = torch.from_numpy(raw.route_ids)

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.sequences[index],
            self.futures[index],
            self.targets[index],
            self.baselines[index],
            self.masks[index],
            self.route_ids[index],
        )


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = 3
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=self.kernel_size,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=self.kernel_size,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def causal(self, layer: nn.Conv1d, values: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size - 1) * self.dilation
        return layer(F.pad(values, (padding, 0)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.causal(self.conv1, values)
        values = F.gelu(self.norm1(values))
        values = self.dropout(values)
        values = self.causal(self.conv2, values)
        values = F.gelu(self.norm2(values))
        values = self.dropout(values)
        return values + residual


class TCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int,
        dilations: Sequence[int],
        dropout: float,
    ):
        super().__init__()
        self.input = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                CausalResidualBlock(channels, dilation, dropout)
                for dilation in dilations
            ]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.input(values.transpose(1, 2))
        for block in self.blocks:
            values = block(values)
        return values.transpose(1, 2)


class ParallelLSTMTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        future_dim: int,
        hidden: int,
        lstm_layers: int,
        tcn_channels: int,
        dilations: Sequence[int],
        fusion_dim: int,
        dropout: float,
        num_routes: int = len(ROUTES),
        horizon: int = HORIZON,
    ):
        super().__init__()
        self.horizon_embedding = nn.Embedding(horizon, fusion_dim)
        self.route_embedding = nn.Embedding(num_routes, max(8, fusion_dim // 4))
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.tcn = TCNEncoder(input_dim, tcn_channels, dilations, dropout)
        self.lstm_projection = nn.Linear(hidden, fusion_dim)
        self.tcn_projection = nn.Linear(tcn_channels, fusion_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention_query = nn.Linear(fusion_dim, fusion_dim)
        self.route_dim = max(8, fusion_dim // 4)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim + future_dim + fusion_dim + self.route_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 1),
        )

    def encode(self, sequence: torch.Tensor) -> torch.Tensor:
        lstm_values, _ = self.lstm(sequence)
        tcn_values = self.tcn(sequence)
        return self.fusion(
            torch.cat(
                [
                    self.lstm_projection(lstm_values),
                    self.tcn_projection(tcn_values),
                ],
                dim=-1,
            )
        )

    def decode(
        self,
        fused: torch.Tensor,
        future_features: torch.Tensor,
        horizon: torch.Tensor,
        route_ids: torch.Tensor,
    ) -> torch.Tensor:
        if future_features.dim() == 2:
            horizon_values = self.horizon_embedding(horizon)
            query = self.attention_query(horizon_values).unsqueeze(1)
            scores = (fused * query).sum(dim=-1) / math.sqrt(fused.shape[-1])
            attention = torch.softmax(scores, dim=1)
            context = (fused * attention.unsqueeze(-1)).sum(dim=1)
            route_values = self.route_embedding(route_ids)
            values = torch.cat(
                [context, future_features, horizon_values, route_values],
                dim=-1,
            )
            return self.head(values).squeeze(-1)

        horizon_values = self.horizon_embedding(horizon)
        query = self.attention_query(horizon_values)
        scores = torch.einsum("btf,hf->bht", fused, query) / math.sqrt(fused.shape[-1])
        attention = torch.softmax(scores, dim=-1)
        context = torch.einsum("bht,btf->bhf", attention, fused)
        route_values = self.route_embedding(route_ids).unsqueeze(1).expand(
            -1,
            context.shape[1],
            -1,
        )
        expanded_horizon = horizon_values.unsqueeze(0).expand(fused.shape[0], -1, -1)
        values = torch.cat(
            [context, future_features, expanded_horizon, route_values],
            dim=-1,
        )
        return self.head(values).squeeze(-1)

    def forward(
        self,
        sequence: torch.Tensor,
        future_features: torch.Tensor,
        horizon: torch.Tensor,
        route_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode(self.encode(sequence), future_features, horizon, route_ids)


def prediction_from_delta(
    delta: torch.Tensor,
    baseline: torch.Tensor,
) -> torch.Tensor:
    log_level = torch.log1p(torch.clamp(baseline, min=0.0)) + delta
    log_level = torch.clamp(log_level, min=0.0, max=12.0)
    return torch.expm1(log_level)


def make_loader(
    raw: RawForecast,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = SequenceDataset(raw)
    return DataLoader(
        dataset,
        batch_size=min(10, max(1, batch_size), len(dataset)),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def masked_point_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_kind: str,
) -> torch.Tensor:
    if loss_kind == "l1":
        point = (predicted - target).abs()
    else:
        point = F.smooth_l1_loss(predicted, target, beta=1.0, reduction="none")
    weights = mask.to(point.dtype)
    return (point * weights).sum() / weights.sum().clamp_min(1.0)


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.abs(actual).sum())
    if denominator <= 1e-8:
        return 0.0 if float(np.abs(predicted).sum()) <= 1e-8 else 1.0
    return float(np.abs(actual - predicted).sum() / denominator)


def score_from_wape(error: float) -> float:
    return max(0.0, 1.0 - error)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    loss_kind: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    horizons = torch.arange(HORIZON, device=device)
    for sequence, future, target, baseline, mask, route_ids in loader:
        sequence = sequence.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        baseline = baseline.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        route_ids = route_ids.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            delta = model(sequence, future, horizons, route_ids)
            predicted = prediction_from_delta(delta, baseline)
            loss = masked_point_loss(predicted, target, mask, loss_kind)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        items = int(mask.sum().item())
        total_loss += float(loss.detach().cpu()) * items
        total_items += items
    return total_loss / max(1, total_items)


@torch.no_grad()
def predict_prepared(
    model: nn.Module,
    raw: RawForecast,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = make_loader(raw, batch_size=batch_size, shuffle=False)
    horizons = torch.arange(HORIZON, device=device)
    predicted_rows: list[np.ndarray] = []
    for sequence, future, _target, baseline, _mask, route_ids in loader:
        sequence = sequence.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)
        baseline = baseline.to(device, non_blocking=True)
        route_ids = route_ids.to(device, non_blocking=True)
        with autocast_context(device):
            delta = model(sequence, future, horizons, route_ids)
            predicted = prediction_from_delta(delta, baseline)
        predicted_rows.append(predicted.float().cpu().numpy())
    predicted_full = np.concatenate(predicted_rows, axis=0)
    return predicted_full[raw.masks], raw.targets[raw.masks]


def fit_model(
    config: dict[str, Any],
    train_raw: RawForecast,
    val_raw: RawForecast | None,
    device: torch.device,
    max_epochs: int,
    patience: int,
    seed: int,
) -> tuple[ParallelLSTMTCN, dict[str, Any]]:
    set_seed(seed)
    input_dim = int(train_raw.sequences.shape[-1])
    model = ParallelLSTMTCN(
        input_dim=input_dim,
        future_dim=input_dim,
        hidden=int(config["lstm_hidden"]),
        lstm_layers=int(config["lstm_layers"]),
        tcn_channels=int(config["tcn_channels"]),
        dilations=tuple(int(value) for value in config["dilations"]),
        fusion_dim=int(config["fusion_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, max_epochs),
    )
    amp_scaler = (
        torch.cuda.amp.GradScaler(enabled=True)
        if device.type == "cuda"
        else None
    )
    train_loader = make_loader(
        train_raw,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            amp_scaler,
            device,
            str(config["loss_kind"]),
        )
        scheduler.step()
        row = {"epoch": float(epoch), "train_loss": train_loss}
        if val_raw is not None:
            val_pred, val_actual = predict_prepared(
                model,
                val_raw,
                batch_size=int(config["batch_size"]),
                device=device,
            )
            val_error = wape(val_actual, val_pred)
            row["val_wape"] = val_error
            if val_error < best_val - 1e-5:
                best_val = val_error
                best_epoch = epoch
                stale = 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            else:
                stale += 1
            if stale >= patience:
                history.append(row)
                break
        else:
            best_epoch = epoch
        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = None
    if val_raw is not None:
        val_pred, val_actual = predict_prepared(
            model,
            val_raw,
            batch_size=int(config["batch_size"]),
            device=device,
        )
        final_val = wape(val_actual, val_pred)
    return model, {
        "best_epoch": best_epoch,
        "best_val_wape": final_val,
        "history": history,
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
    }


class PreparedCache:
    """Cache raw numpy tensors for feature variants and context lengths."""

    def __init__(
        self,
        history: np.ndarray,
        tables: EnrichmentTables,
    ):
        self.provider = SafeFeatureProvider(tables)
        self.history = history
        self.tables = tables
        self.cache: dict[tuple[Any, ...], RawForecast] = {}

    def get(
        self,
        variant: str,
        baseline_kind: str,
        context: int,
        origins: Sequence[date],
        target_end: date,
    ) -> RawForecast:
        key = (
            variant,
            baseline_kind,
            context,
            tuple(origin.isoformat() for origin in origins),
            target_end.isoformat(),
        )
        if key not in self.cache:
            assembler = FeatureAssembler(
                self.history,
                self.tables,
                self.provider,
                variant,
                baseline_kind,
            )
            self.cache[key] = assembler.build_dataset(origins, target_end, context)
        return self.cache[key]


def scale_pair(
    train_raw: RawForecast,
    val_raw: RawForecast,
) -> tuple[RawForecast, RawForecast, FeatureScaler]:
    scaler = FeatureScaler.fit(train_raw.sequences)
    return scale_forecast(train_raw, scaler), scale_forecast(val_raw, scaler), scaler


def weekly_origins(start: date, end: date) -> list[date]:
    values = []
    current = start
    while current <= end:
        values.append(current)
        current += timedelta(days=7)
    if not values or values[-1] != end:
        values.append(end)
    return values


def default_config() -> dict[str, Any]:
    return {
        "context": 336,
        "lstm_hidden": 64,
        "lstm_layers": 2,
        "tcn_channels": 64,
        "dilations": [1, 2, 4, 8, 16, 32],
        "fusion_dim": 64,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 512,
        "loss_kind": "smooth_l1",
        "feature_variant": "calendar_weather",
        "baseline_kind": "ewm",
    }


def random_config(rng: random.Random, trial: int) -> dict[str, Any]:
    if trial == 0:
        return default_config()
    dilation_options = [
        [1, 2, 4, 8, 16],
        [1, 2, 4, 8, 16, 32],
    ]
    return {
        "context": rng.choice([168, 336, 504]),
        "lstm_hidden": rng.choice([32, 64, 96, 128]),
        "lstm_layers": rng.choice([1, 2]),
        "tcn_channels": rng.choice([32, 64, 96]),
        "dilations": rng.choice(dilation_options),
        "fusion_dim": rng.choice([32, 64, 96, 128]),
        "dropout": rng.choice([0.1, 0.2, 0.3]),
        "learning_rate": 10 ** rng.uniform(-4.0, -2.5),
        "weight_decay": 10 ** rng.uniform(-6.0, -3.0),
        "batch_size": rng.choice([128, 256, 512]),
        "loss_kind": rng.choice(["l1", "smooth_l1"]),
        "feature_variant": rng.choice(list(FEATURE_VARIANTS)),
        "baseline_kind": rng.choice(["ewm", "mean"]),
    }


def tune_blend_weight(
    actual: np.ndarray,
    neural: np.ndarray,
    baseline: np.ndarray,
) -> tuple[float, float]:
    best_weight = 0.0
    best_error = float("inf")
    for weight in np.linspace(0.0, 1.0, 21):
        prediction = weight * neural + (1.0 - weight) * baseline
        error = wape(actual, prediction)
        if error < best_error:
            best_error = error
            best_weight = float(weight)
    return best_weight, best_error


def search_hyperparameters(
    cache: PreparedCache,
    fit_origins: Sequence[date],
    validation_origin: date,
    target_end: date,
    trials: int,
    epochs: int,
    patience: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    validation_cache: dict[tuple[str, str, int], RawForecast] = {}
    for trial in range(trials):
        config = random_config(rng, trial)
        baseline_kind = str(config["baseline_kind"])
        train_raw = cache.get(
            str(config["feature_variant"]),
            baseline_kind,
            int(config["context"]),
            fit_origins,
            target_end,
        )
        validation_key = (
            str(config["feature_variant"]),
            baseline_kind,
            int(config["context"]),
        )
        if validation_key not in validation_cache:
            validation_cache[validation_key] = cache.get(
                str(config["feature_variant"]),
                baseline_kind,
                int(config["context"]),
                [validation_origin],
                target_end,
            )
        val_raw = validation_cache[validation_key]
        train_scaled, val_scaled, scaler = scale_pair(train_raw, val_raw)
        model, fit_metrics = fit_model(
            config,
            train_scaled,
            val_scaled,
            device,
            max_epochs=epochs,
            patience=patience,
            seed=seed + trial,
        )
        val_pred, val_actual = predict_prepared(
            model,
            val_scaled,
            batch_size=int(config["batch_size"]),
            device=device,
        )
        baseline_values = val_scaled.baselines[val_scaled.masks]
        neural_error = wape(val_actual, val_pred)
        blend_weight, blend_error = tune_blend_weight(
            val_actual,
            val_pred,
            baseline_values,
        )
        result = {
            "trial": trial,
            "config": config,
            "neural_wape": neural_error,
            "blend_wape": blend_error,
            "blend_weight": blend_weight,
            "best_epoch": fit_metrics["best_epoch"],
        }
        results.append(result)
        selection_error = min(neural_error, blend_error)
        print(
            f"trial={trial + 1}/{trials} variant={config['feature_variant']} "
            f"context={config['context']} neural_wape={neural_error:.5f} "
            f"blend_wape={blend_error:.5f} weight={blend_weight:.2f}",
            flush=True,
        )
        if best is None or selection_error < float(best["selection_wape"]):
            best = {
                "selection_wape": selection_error,
                "config": config,
                "best_epoch": fit_metrics["best_epoch"],
                "blend_weight": blend_weight,
                "scaler": scaler.payload(),
                "state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
                "validation_prediction": val_pred,
                "validation_actual": val_actual,
                "validation_baseline": baseline_values,
            }
    if best is None:
        raise RuntimeError("Hyperparameter search did not produce a model")
    best["trials"] = results
    return best


def evaluate_baseline(raw: RawForecast) -> dict[str, float]:
    actual = raw.targets[raw.masks]
    prediction = raw.baselines[raw.masks]
    error = wape(actual, prediction)
    return {"wape": error, "score": score_from_wape(error)}


def save_submission(
    path: Path,
    predictions: np.ndarray,
    raw_forecast: RawForecast,
    history: np.ndarray,
    blend_weight: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    effective = (
        blend_weight * predictions
        + (1.0 - blend_weight) * raw_forecast.baselines
    )
    route_predictions = effective
    written = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["route", "date", "hour", "prediction"])
        cutoff = date_index(raw_forecast.cutoffs[0])
        for route_index, route in enumerate(ROUTES):
            has_history = float(history[:, route_index].sum()) > 0.0
            for horizon_index in range(HORIZON):
                target_day_index = cutoff + 1 + horizon_index // 24
                hour = horizon_index % 24
                if target_day_index >= N_DAYS:
                    continue
                prediction = (
                    float(route_predictions[route_index, horizon_index])
                    if has_history
                    else 0.0
                )
                writer.writerow(
                    [
                        route,
                        day_at(target_day_index).isoformat(),
                        hour,
                        f"{max(0.0, prediction):.6f}",
                    ]
                )
                written += 1
    expected = len(ROUTES) * HORIZON
    if written != expected:
        raise RuntimeError(f"Submission has {written} rows; expected {expected}")


def train_final_and_predict(
    config: dict[str, Any],
    cache: PreparedCache,
    history: np.ndarray,
    final_origins: Sequence[date],
    device: torch.device,
    epochs: int,
    seed: int,
    repeats: int,
    blend_weight: float,
    output_dir: Path,
) -> dict[str, Any]:
    variant = str(config["feature_variant"])
    baseline_kind = str(config["baseline_kind"])
    context = int(config["context"])
    final_raw = cache.get(
        variant,
        baseline_kind,
        context,
        final_origins,
        TEST_END,
    )
    forecast_assembler = FeatureAssembler(
        history,
        cache.tables,
        cache.provider,
        variant,
        baseline_kind,
    )
    forecast_raw_unscaled = forecast_assembler.build_dataset(
        [TEST_END],
        FORECAST_END,
        context,
    )
    scaler = FeatureScaler.fit(final_raw.sequences)
    final_scaled = scale_forecast(final_raw, scaler)
    forecast_scaled = scale_forecast(forecast_raw_unscaled, scaler)
    repeat_predictions: list[np.ndarray] = []
    repeat_metrics: list[dict[str, Any]] = []
    state_dicts: list[dict[str, torch.Tensor]] = []
    for repeat in range(repeats):
        model, fit_metrics = fit_model(
            config,
            final_scaled,
            None,
            device,
            max_epochs=max(1, epochs),
            patience=max(1, min(epochs, 5)),
            seed=seed + repeat,
        )
        prediction, _ = predict_prepared(
            model,
            forecast_scaled,
            batch_size=int(config["batch_size"]),
            device=device,
        )
        repeat_predictions.append(prediction.reshape(len(ROUTES), HORIZON))
        repeat_metrics.append(
            {
                "seed": seed + repeat,
                "best_epoch": fit_metrics["best_epoch"],
                "history": fit_metrics["history"],
            }
        )
        state_dicts.append(
            {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    forecast_prediction = np.mean(np.stack(repeat_predictions), axis=0)
    forecast_baseline = forecast_raw_unscaled.baselines
    forecast_output = (
        blend_weight * forecast_prediction
        + (1.0 - blend_weight) * forecast_baseline
    )
    submission_path = output_dir / "submission.csv"
    save_submission(
        submission_path,
        forecast_prediction,
        forecast_raw_unscaled,
        history,
        blend_weight,
    )
    checkpoint = {
        "config": config,
        "feature_names": forecast_assembler.feature_names,
        "scaler": scaler.payload(),
        "state_dicts": state_dicts,
    }
    torch.save(checkpoint, output_dir / "best_model.pt")
    return {
        "submission_path": str(submission_path),
        "forecast_prediction": forecast_prediction,
        "forecast_baseline": forecast_baseline,
        "forecast_output": forecast_output,
        "fit_metrics": {"repeats": repeat_metrics},
        "feature_names": forecast_assembler.feature_names,
        "scaler": scaler,
    }


def run_submission(
    args: argparse.Namespace,
    history: np.ndarray,
    cache: PreparedCache,
    device: torch.device,
    output_dir: Path,
) -> None:
    """Train the planned architecture and write submission.csv.

    The cancelled HPO run never reached the point where best_config.json is
    written, so this uses the fixed default configuration: the first and
    intended architecture, not a random later trial.
    """

    config = default_config()
    epochs = max(1, int(args.hpo_epochs))
    config["best_epoch"] = epochs
    write_json(
        output_dir / "best_config.json",
        {
            "source": "default_config",
            "reason": "The cancelled Kaggle HPO run did not persist a completed trial.",
            "config": config,
        },
    )
    print(f"submission_config={json.dumps(config)}", flush=True)

    calib_train_raw = cache.get(
        config["feature_variant"],
        config["baseline_kind"],
        int(config["context"]),
        weekly_origins(date(2025, 4, 30), date(2025, 6, 30)),
        date(2025, 7, 31),
    )
    calib_val_raw = cache.get(
        config["feature_variant"],
        config["baseline_kind"],
        int(config["context"]),
        [date(2025, 7, 31)],
        TRAIN_END,
    )
    calib_train, calib_val, _ = scale_pair(calib_train_raw, calib_val_raw)
    calib_model, calib_fit = fit_model(
        config,
        calib_train,
        calib_val,
        device,
        max_epochs=epochs,
        patience=args.patience,
        seed=args.seed,
    )
    calib_pred, calib_actual = predict_prepared(
        calib_model,
        calib_val,
        batch_size=10,
        device=device,
    )
    calib_baseline = calib_val.baselines[calib_val.masks]
    blend_weight, blend_error = tune_blend_weight(
        calib_actual,
        calib_pred,
        calib_baseline,
    )
    neural_error = wape(calib_actual, calib_pred)
    print(
        f"august_calibration neural_wape={neural_error:.5f} "
        f"blend_wape={blend_error:.5f} weight={blend_weight:.2f}",
        flush=True,
    )
    del calib_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    outer_train_raw = cache.get(
        config["feature_variant"],
        config["baseline_kind"],
        int(config["context"]),
        weekly_origins(date(2025, 4, 30), date(2025, 7, 31)),
        TRAIN_END,
    )
    outer_val_raw = cache.get(
        config["feature_variant"],
        config["baseline_kind"],
        int(config["context"]),
        [TRAIN_END],
        TEST_END,
    )
    outer_train, outer_val, _ = scale_pair(outer_train_raw, outer_val_raw)
    outer_model, _ = fit_model(
        config,
        outer_train,
        None,
        device,
        max_epochs=epochs,
        patience=args.patience,
        seed=args.seed + 1000,
    )
    outer_pred, outer_actual = predict_prepared(
        outer_model,
        outer_val,
        batch_size=10,
        device=device,
    )
    outer_baseline = outer_val.baselines[outer_val.masks]
    outer_neural = wape(outer_actual, outer_pred)
    outer_blend = wape(
        outer_actual,
        blend_weight * outer_pred + (1.0 - blend_weight) * outer_baseline,
    )
    outer_metrics = {
        "baseline": evaluate_baseline(outer_val_raw),
        "neural": {"wape": outer_neural, "score": score_from_wape(outer_neural)},
        "blend": {
            "weight": blend_weight,
            "wape": outer_blend,
            "score": score_from_wape(outer_blend),
        },
    }
    print(f"outer_metrics={json.dumps(outer_metrics)}", flush=True)
    del outer_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    final_result = train_final_and_predict(
        config=config,
        cache=cache,
        history=history,
        final_origins=weekly_origins(date(2025, 4, 30), date(2025, 9, 30)),
        device=device,
        epochs=epochs,
        seed=args.seed + 2000,
        repeats=1,
        blend_weight=blend_weight,
        output_dir=output_dir,
    )
    submission_path = Path(final_result["submission_path"])
    working_copy = Path("/kaggle/working/submission.csv")
    if working_copy.parent.exists() and submission_path.resolve() != working_copy.resolve():
        working_copy.write_bytes(submission_path.read_bytes())
    metrics = {
        "device": str(device),
        "config_source": "default_config",
        "august_calibration": {
            "neural_wape": neural_error,
            "blend_wape": blend_error,
            "blend_weight": blend_weight,
            "history": calib_fit["history"],
        },
        "outer_sep_oct": outer_metrics,
        "final_fit": final_result["fit_metrics"],
        "submission_rows": 14_640,
    }
    write_json(output_dir / "validation_metrics.json", metrics)
    write_json(
        output_dir / "feature_schema.json",
        {"features": final_result["feature_names"]},
    )
    write_json(output_dir / "scaler.json", final_result["scaler"].payload())
    print(f"submission={submission_path}", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def run_pipeline(args: argparse.Namespace) -> None:
    if args.trials < 1 or args.hpo_epochs < 1 or args.repeats < 1:
        raise ValueError("trials, hpo_epochs and repeats must be positive")
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    history = load_history(input_root)
    tables = load_enrichment(input_root)
    print(
        f"history_shape={history.shape} observed_boardings={history.sum():.0f}",
        flush=True,
    )
    cache = PreparedCache(history, tables)
    if args.mode == "submission":
        run_submission(args, history, cache, device, output_dir)
        return

    hpo_fit_origins = weekly_origins(date(2025, 4, 30), date(2025, 5, 31))
    hpo_validation_origin = date(2025, 7, 31)
    hpo_best = search_hyperparameters(
        cache=cache,
        fit_origins=hpo_fit_origins,
        validation_origin=hpo_validation_origin,
        target_end=TRAIN_END,
        trials=args.trials,
        epochs=args.hpo_epochs,
        patience=args.patience,
        device=device,
        seed=args.seed,
    )
    best_config = dict(hpo_best["config"])
    best_config["best_epoch"] = int(hpo_best["best_epoch"])
    write_json(
        output_dir / "best_config.json",
        {
            "config": best_config,
            "selection_wape": hpo_best["selection_wape"],
            "blend_weight": hpo_best["blend_weight"],
            "trials": hpo_best["trials"],
        },
    )
    outer_origins = weekly_origins(date(2025, 4, 30), date(2025, 7, 31))
    outer_train_raw = cache.get(
        str(best_config["feature_variant"]),
        str(best_config["baseline_kind"]),
        int(best_config["context"]),
        outer_origins,
        TRAIN_END,
    )
    outer_val_raw = cache.get(
        str(best_config["feature_variant"]),
        str(best_config["baseline_kind"]),
        int(best_config["context"]),
        [TRAIN_END],
        TEST_END,
    )
    outer_train, outer_val, _ = scale_pair(
        outer_train_raw,
        outer_val_raw,
    )
    outer_predictions: list[np.ndarray] = []
    outer_repeat_wapes: list[float] = []
    for repeat in range(args.repeats):
        outer_model, _ = fit_model(
            best_config,
            outer_train,
            None,
            device,
            max_epochs=max(1, int(best_config["best_epoch"])),
            patience=args.patience,
            seed=args.seed + 1000 + repeat,
        )
        outer_repeat_prediction, outer_actual = predict_prepared(
            outer_model,
            outer_val,
            batch_size=int(best_config["batch_size"]),
            device=device,
        )
        outer_predictions.append(outer_repeat_prediction)
        outer_repeat_wapes.append(wape(outer_actual, outer_repeat_prediction))
        del outer_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    outer_prediction = np.mean(np.stack(outer_predictions), axis=0)
    outer_baseline = outer_val.baselines[outer_val.masks]
    outer_neural_error = wape(outer_actual, outer_prediction)
    outer_blend_error = wape(
        outer_actual,
        hpo_best["blend_weight"] * outer_prediction
        + (1.0 - hpo_best["blend_weight"]) * outer_baseline,
    )
    outer_metrics = {
        "baseline": evaluate_baseline(outer_val_raw),
        "neural": {
            "wape": outer_neural_error,
            "score": score_from_wape(outer_neural_error),
            "repeat_wapes": outer_repeat_wapes,
        },
        "blend": {
            "weight": hpo_best["blend_weight"],
            "wape": outer_blend_error,
            "score": score_from_wape(outer_blend_error),
        },
    }
    print(f"outer_metrics={json.dumps(outer_metrics)}", flush=True)

    final_origins = weekly_origins(date(2025, 4, 30), date(2025, 9, 30))
    final_result = train_final_and_predict(
        config=best_config,
        cache=cache,
        history=history,
        final_origins=final_origins,
        device=device,
        epochs=max(1, int(best_config["best_epoch"])),
        seed=args.seed + 2000,
        repeats=args.repeats,
        blend_weight=float(hpo_best["blend_weight"]),
        output_dir=output_dir,
    )
    metrics = {
        "device": str(device),
        "hpo": {
            "selection_wape": hpo_best["selection_wape"],
            "blend_weight": hpo_best["blend_weight"],
            "best_epoch": hpo_best["best_epoch"],
        },
        "outer_sep_oct": outer_metrics,
        "final_fit": {
            "repeats": final_result["fit_metrics"]["repeats"],
        },
        "submission_rows": 14_640,
    }
    write_json(output_dir / "validation_metrics.json", metrics)
    write_json(
        output_dir / "feature_schema.json",
        {"features": final_result["feature_names"]},
    )
    write_json(
        output_dir / "scaler.json",
        final_result["scaler"].payload(),
    )
    print(f"submission={final_result['submission_path']}", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/lstm_tcn_artifacts",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=int(os.environ.get("LSTM_TCN_TRIALS", "20")),
    )
    parser.add_argument(
        "--hpo-epochs",
        type=int,
        default=int(os.environ.get("LSTM_TCN_HPO_EPOCHS", "20")),
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--mode",
        choices=("search", "submission"),
        default="search",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
