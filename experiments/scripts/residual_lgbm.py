"""Поправка LightGBM к замороженному LSTM+TCN.

Веса нейросети берутся из чекпоинта Kaggle и не обновляются.
Бустинг учит остаток: фактические посадки минус прогноз замороженной модели.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

TRAIN_CUTOFFS = [date(2025, 5, 31), date(2025, 6, 30)]
EARLY_CUTOFF = date(2025, 7, 31)
HOLD_CUTOFF = date(2025, 8, 31)
FORECAST_CUTOFF = date(2025, 10, 31)
AUGUST_END = date(2025, 8, 31)
FEATURE_COLUMNS = [
    "neural",
    "baseline",
    "route",
    "hour",
    "dow",
    "month",
    "weekend",
    "day_off",
    "short_day",
    "preholiday",
    "holiday_id",
    "horizon_day",
    "offices",
]
CATEGORICAL = ["route", "hour", "dow", "month", "holiday_id"]


def first_file(name: str) -> Path:
    local = {
        "best_model.pt": ROOT / "data" / "kaggle_mstrans" / "lstm_tcn_v5" / "lstm_tcn_artifacts" / "best_model.pt",
        "labels_day_train.csv": ROOT / "data" / "kaggle_mstrans" / "labels_day_train.csv",
        "labels_day_test.csv": ROOT / "data" / "kaggle_mstrans" / "labels_day_test.csv",
        "holidays_2025.csv": ROOT / "data" / "kaggle_enrichment" / "holidays_2025.csv",
        "train_lstm_tcn.py": ROOT / "experiments" / "notebooks" / "notebook35f71e374c" / "train_lstm_tcn.py",
    }
    candidate = local.get(name)
    if candidate is not None and candidate.exists():
        return candidate
    if KAGGLE_INPUT.exists():
        matches = sorted(KAGGLE_INPUT.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(name)


def output_dir() -> Path:
    if KAGGLE_WORKING.exists():
        destination = KAGGLE_WORKING / "lstm_tcn_gb"
    else:
        destination = ROOT / "data" / "kaggle_mstrans" / "lstm_tcn_v5" / "residual_lgbm"
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def load_lstm_module():
    path = first_file("train_lstm_tcn.py")
    spec = importlib.util.spec_from_file_location("train_lstm_tcn", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_lstm_tcn"] = module
    spec.loader.exec_module(module)
    return module


def load_history(module) -> np.ndarray:
    history = np.zeros((module.N_DAYS, len(module.ROUTES), 24), dtype=np.float32)
    route_index = {route: index for index, route in enumerate(module.ROUTES)}
    for name in ("labels_day_train.csv", "labels_day_test.csv"):
        frame = pd.read_csv(first_file(name), sep=";", encoding="utf-8")
        for row in frame.itertuples(index=False):
            route = route_index.get(int(row.route))
            if route is None:
                continue
            day = module.date_index(module.parse_date(row.date))
            hour = int(row.hour)
            if 0 <= day < module.N_DAYS and 0 <= hour < 24:
                history[day, route, hour] = float(row.boardings)
    return history


def frozen_model(module, checkpoint: dict):
    config = checkpoint["config"]
    scaler_payload = checkpoint["scaler"]
    scaler = module.FeatureScaler(
        np.asarray(scaler_payload["mean"], dtype=np.float32),
        np.asarray(scaler_payload["std"], dtype=np.float32),
    )
    model = module.ParallelLSTMTCN(
        input_dim=len(scaler.mean),
        future_dim=len(scaler.mean),
        hidden=int(config["lstm_hidden"]),
        lstm_layers=int(config["lstm_layers"]),
        tcn_channels=int(config["tcn_channels"]),
        dilations=tuple(int(value) for value in config["dilations"]),
        fusion_dim=int(config["fusion_dim"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dicts"][0])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, scaler


def fingerprint(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        array = state[name].detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def predict_origin(module, assembler, model, scaler, cutoff: date, target_end: date, context: int):
    raw = assembler.build_dataset([cutoff], target_end, context)
    scaled = module.scale_forecast(raw, scaler)
    predicted, _ = module.predict_prepared(model, scaled, batch_size=10, device=torch.device("cpu"))
    full = np.zeros_like(raw.targets)
    full[raw.masks] = predicted
    return raw, full


def frame_from_prediction(module, tables, raw, predicted, date_from: date, date_to: date) -> pd.DataFrame:
    cutoff = raw.cutoffs[0]
    rows = []
    for route_index, route in enumerate(module.ROUTES):
        for horizon_index in range(module.HORIZON):
            if not bool(raw.masks[route_index, horizon_index]):
                continue
            day_index = module.date_index(cutoff) + 1 + horizon_index // 24
            day = module.day_at(day_index)
            if day < date_from or day > date_to:
                continue
            hour = horizon_index % 24
            neural = float(predicted[route_index, horizon_index])
            baseline = float(raw.baselines[route_index, horizon_index])
            actual = float(raw.targets[route_index, horizon_index])
            calendar = tables.calendar[day_index]
            rows.append(
                {
                    "route": route,
                    "date": day.isoformat(),
                    "hour": hour,
                    "dow": day.weekday(),
                    "month": day.month,
                    "weekend": int(calendar[1]),
                    "day_off": int(calendar[0]),
                    "short_day": int(calendar[2]),
                    "preholiday": int(calendar[3]),
                    "holiday_id": int(tables.holiday_ids[day_index]),
                    "horizon_day": horizon_index // 24,
                    "offices": float(raw.futures[route_index, horizon_index, -1])
                    if raw.futures.shape[-1]
                    else 0.0,
                    "neural": neural,
                    "baseline": baseline,
                    "actual": actual,
                    "residual": actual - neural,
                }
            )
    return pd.DataFrame(rows)


def fit_booster(train: pd.DataFrame, early: pd.DataFrame | None, rounds: int = 500) -> lgb.Booster:
    categorical = [column for column in CATEGORICAL if column in FEATURE_COLUMNS]
    train_set = lgb.Dataset(
        train[FEATURE_COLUMNS],
        label=train["residual"],
        categorical_feature=categorical,
        free_raw_data=False,
    )
    callbacks = [lgb.log_evaluation(0)]
    valid = [train_set]
    if early is not None and not early.empty:
        valid.append(
            lgb.Dataset(
                early[FEATURE_COLUMNS],
                label=early["residual"],
                categorical_feature=categorical,
                reference=train_set,
                free_raw_data=False,
            )
        )
        callbacks.insert(0, lgb.early_stopping(40, verbose=False))
    return lgb.train(
        {
            "objective": "regression_l1",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": 2025,
        },
        train_set,
        num_boost_round=rounds,
        valid_sets=valid,
        callbacks=callbacks,
    )


def corrected(frame: pd.DataFrame, booster: lgb.Booster) -> np.ndarray:
    residual = booster.predict(frame[FEATURE_COLUMNS], num_iteration=booster.best_iteration)
    return np.maximum(frame["neural"].to_numpy() + residual, 0.0)


def wape_frame(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    actual = frame["actual"].to_numpy()
    denominator = float(np.abs(actual).sum())
    if denominator <= 1e-8:
        return 0.0
    return float(np.abs(actual - prediction).sum() / denominator)


def main():
    OUT = output_dir()
    module = load_lstm_module()
    checkpoint_path = first_file("best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, scaler = frozen_model(module, checkpoint)
    weight_id = fingerprint(model.state_dict())
    print(f"frozen_sha256={weight_id}", flush=True)

    history = load_history(module)
    tables = module.load_enrichment(first_file("holidays_2025.csv").parent)
    assembler = module.FeatureAssembler(
        history,
        tables,
        module.SafeFeatureProvider(tables),
        checkpoint["config"]["feature_variant"],
        checkpoint["config"]["baseline_kind"],
    )
    context = int(checkpoint["config"]["context"])

    def collect(cutoff: date, target_end: date, date_from: date, date_to: date) -> tuple[pd.DataFrame, object, np.ndarray]:
        print(f"infer cutoff={cutoff.isoformat()} targets={date_from.isoformat()}..{date_to.isoformat()}", flush=True)
        raw, predicted = predict_origin(module, assembler, model, scaler, cutoff, target_end, context)
        frame = frame_from_prediction(module, tables, raw, predicted, date_from, date_to)
        print(f"rows={len(frame)}", flush=True)
        return frame, raw, predicted

    train_parts = [
        collect(cutoff, AUGUST_END, cutoff + timedelta(days=1), AUGUST_END)[0]
        for cutoff in TRAIN_CUTOFFS
    ]
    train = pd.concat(train_parts, ignore_index=True)
    early, _, _ = collect(EARLY_CUTOFF, AUGUST_END, date(2025, 8, 1), AUGUST_END)
    hold, _, _ = collect(HOLD_CUTOFF, module.TEST_END, date(2025, 9, 1), module.TEST_END)
    booster = fit_booster(train, early)
    hold_prediction = corrected(hold, booster)
    metrics = {
        "frozen_checkpoint": str(checkpoint_path),
        "frozen_sha256": weight_id,
        "trainable_parameters": 0,
        "best_iteration": int(booster.best_iteration or booster.current_iteration()),
        "september_october": {
            "neural_wape": wape_frame(hold, hold["neural"].to_numpy()),
            "baseline_wape": wape_frame(hold, hold["baseline"].to_numpy()),
            "corrected_wape": wape_frame(hold, hold_prediction),
        },
    }
    metrics["september_october"]["neural_score"] = max(0.0, 1.0 - metrics["september_october"]["neural_wape"])
    metrics["september_october"]["baseline_score"] = max(0.0, 1.0 - metrics["september_october"]["baseline_wape"])
    metrics["september_october"]["corrected_score"] = max(0.0, 1.0 - metrics["september_october"]["corrected_wape"])
    print(json.dumps(metrics["september_october"], ensure_ascii=False), flush=True)

    final_train = pd.concat([train, early, hold], ignore_index=True)
    final_rounds = int(booster.best_iteration or booster.current_iteration())
    final_booster = fit_booster(final_train, None, rounds=final_rounds)
    forecast_frame, forecast_raw, _ = collect(
        FORECAST_CUTOFF,
        module.FORECAST_END,
        date(2025, 11, 1),
        module.FORECAST_END,
    )
    forecast_values = corrected(forecast_frame, final_booster)
    forecast_grid = np.zeros((len(module.ROUTES), module.HORIZON), dtype=np.float32)
    route_index = {route: index for index, route in enumerate(module.ROUTES)}
    for row, value in zip(forecast_frame.itertuples(index=False), forecast_values):
        day = date.fromisoformat(row.date)
        horizon = (day - FORECAST_CUTOFF).days - 1
        horizon = horizon * 24 + int(row.hour)
        forecast_grid[route_index[int(row.route)], horizon] = value
    submission = OUT / "submission.csv"
    module.save_submission(submission, forecast_grid, forecast_raw, history, blend_weight=1.0)
    if KAGGLE_WORKING.exists():
        (KAGGLE_WORKING / "submission.csv").write_bytes(submission.read_bytes())
    (OUT / "residual_lgbm.txt").write_text(final_booster.model_to_string(), encoding="utf-8")
    metrics["submission"] = str(submission)
    metrics["submission_rows"] = sum(1 for _ in submission.open(encoding="utf-8")) - 1
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
