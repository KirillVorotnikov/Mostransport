# Вставить целиком в одну ячейку Kaggle.

import importlib.util
import os
import subprocess
import sys
import warnings

if importlib.util.find_spec("catboost") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost", "pandas", "numpy", "matplotlib"])

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor


# =========================
# 1. Настройки
# =========================

ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
RANDOM_STATE = 42

# Поддерживает и Kaggle Dataset, и локальный запуск рядом с labels/.
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input/datasets/sauzzeth/dataset")
if not os.path.exists(os.path.join(DATA_DIR, "labels_day_train.csv")):
    DATA_DIR = "/kaggle/input/dataset"
if not os.path.exists(os.path.join(DATA_DIR, "labels_day_train.csv")):
    DATA_DIR = os.path.dirname(__file__)
    if os.path.exists(os.path.join(DATA_DIR, "labels", "labels_day_train.csv")):
        DATA_DIR = os.path.join(DATA_DIR, "labels")

TRAIN_PATH = os.path.join(DATA_DIR, "labels_day_train.csv")
TEST_PATH = os.path.join(DATA_DIR, "labels_day_test.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/kaggle/working")
if not os.path.exists("/kaggle"):
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.dirname(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 2. Подготовка данных
# =========================


def read_labels(path):
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def aggregate_daily(df):
    """Агрегация почасовых labels в маршрут × день."""
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    return (
        result.groupby(["route", "date"], as_index=False)
        .agg(
            daily_sum=("boardings", "sum"),
            daily_average=("boardings", "mean"),
        )
        .sort_values(["route", "date"])
        .reset_index(drop=True)
    )


def make_daily_grid(start="2025-01-01", end="2025-12-31"):
    return pd.MultiIndex.from_product(
        [ROUTES, pd.date_range(start, end, freq="D")],
        names=["route", "date"],
    ).to_frame(index=False)


def add_calendar_features(df):
    result = df.copy()
    dt = pd.to_datetime(result["date"])
    result["route"] = result["route"].astype(int)
    result["dayofweek"] = dt.dt.dayofweek.astype(int)
    result["day"] = dt.dt.day.astype(int)
    result["month"] = dt.dt.month.astype(int)
    result["dayofyear"] = dt.dt.dayofyear.astype(int)
    result["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    result["quarter"] = dt.dt.quarter.astype(int)
    result["is_weekend"] = result["dayofweek"].isin([5, 6]).astype(int)
    result["is_month_start"] = dt.dt.is_month_start.astype(int)
    result["is_month_end"] = dt.dt.is_month_end.astype(int)

    holidays = pd.to_datetime([
        "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
        "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08",
        "2025-02-23", "2025-03-08", "2025-05-01", "2025-05-02",
        "2025-05-08", "2025-05-09", "2025-06-12", "2025-06-13",
        "2025-11-03", "2025-11-04", "2025-12-31",
    ])
    result["is_holiday"] = dt.isin(holidays).astype(int)
    result["is_off_day"] = (
        (result["is_weekend"] == 1) | (result["is_holiday"] == 1)
    ).astype(int)
    result["is_preholiday"] = (dt + pd.Timedelta(days=1)).isin(holidays).astype(int)

    for column, period in [("dayofweek", 7), ("month", 12), ("dayofyear", 365.25)]:
        result[f"{column}_sin"] = np.sin(2 * np.pi * result[column] / period)
        result[f"{column}_cos"] = np.cos(2 * np.pi * result[column] / period)
    return result


def feature_columns(df):
    return [column for column in df.columns if column not in {"date", "daily_sum", "daily_average"}]


# =========================
# 3. Модели и валидация
# =========================


def make_model():
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=700,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=6.0,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def wape_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true).sum()
    return 0.0 if denominator == 0 else max(0.0, 1.0 - np.abs(y_true - y_pred).sum() / denominator)


def fit_two_models(train_df, predict_df, train_end="2025-08-31"):
    train_mask = train_df["date"] <= pd.Timestamp(train_end)
    features = feature_columns(train_df)
    predictions = predict_df[["route", "date"]].copy()

    for target in ["daily_sum", "daily_average"]:
        model = make_model()
        model.fit(train_df.loc[train_mask, features], train_df.loc[train_mask, target])
        predictions[f"{target}_prediction"] = np.maximum(
            0, model.predict(predict_df[features])
        )
    return predictions


def plot_all_routes(val_df, route_scores, pred_col, title, ncols=2):
    """Факт vs прогноз по каждому маршруту в сетке субплотов."""
    n = len(ROUTES)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4.5 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for i, route in enumerate(ROUTES):
        ax = axes[i]
        route_df = val_df[val_df["route"] == route].sort_values("date")
        ax.plot(route_df["date"], route_df["actual"], label="Факт", color="#1f77b4", linewidth=1.8)
        ax.plot(route_df["date"], route_df[pred_col], label="Прогноз", color="#ff7f0e", linestyle="--", linewidth=1.8)
        ax.set_title(f"Маршрут №{route} | WAPE: {route_scores.get(route, np.nan):.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Дата")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(loc="upper right", fontsize=8)

    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()


def validate(train_features, predictions, validation_start="2025-09-01"):
    val = train_features[train_features["date"] >= pd.Timestamp(validation_start)][
        ["route", "date", "daily_sum", "daily_average"]
    ].merge(predictions, on=["route", "date"], how="left")

    scores = {}
    for target in ["daily_sum", "daily_average"]:
        val["actual"] = val[target]
        pred_col = f"{target}_prediction"
        route_scores = {
            route: wape_score(
                group[target], group[pred_col]
            )
            for route, group in val.groupby("route")
        }
        scores[target] = route_scores
        print(f"{target}: общий WAPE-score = {wape_score(val[target], val[pred_col]):.4f}")
        plot_all_routes(val, route_scores, pred_col, target)
    return scores


# =========================
# 4. Запуск
# =========================


def make_prediction_submission(daily_predictions):
    return daily_predictions[
        ["route", "date", "daily_sum_prediction", "daily_average_prediction"]
    ].copy()


def main():
    train_raw = read_labels(TRAIN_PATH)
    test_raw = read_labels(TEST_PATH)
    observed = pd.concat([train_raw, test_raw], ignore_index=True)
    observed_daily = aggregate_daily(observed)

    # Полная дневная сетка: январь–октябрь с target, ноябрь–декабрь без target.
    daily = make_daily_grid().merge(observed_daily, on=["route", "date"], how="left")
    daily_features = add_calendar_features(daily)
    known = daily_features["daily_sum"].notna()

    # Локальная валидация на сентябре–октябре.
    validation_predictions = fit_two_models(
        daily_features[known],
        daily_features[daily_features["date"] >= pd.Timestamp("2025-09-01")],
        train_end="2025-08-31",
    )
    validate(daily_features[known], validation_predictions)

    # Финальное обучение на всём январе–октябре и прогноз ноября–декабря.
    future = daily_features[daily_features["date"] >= pd.Timestamp("2025-11-01")].copy()
    predictions = fit_two_models(daily_features[known], future, train_end="2025-10-31")
    predictions["date"] = pd.to_datetime(predictions["date"])

    # Новый обогащённый train: исходный target + фактические дневные признаки.
    train_enriched = train_raw.copy()
    train_enriched["date"] = pd.to_datetime(train_enriched["date"])
    train_enriched = train_enriched.merge(observed_daily, on=["route", "date"], how="left")
    train_enriched.to_csv(os.path.join(OUTPUT_DIR, "train_enriched.csv"), sep=";", index=False)

    # Новый обогащённый test: полный прогнозный период, target отсутствует.
    test_enriched = make_daily_grid("2025-11-01", "2025-12-31").merge(
        predictions, on=["route", "date"], how="left"
    )
    test_enriched.to_csv(os.path.join(OUTPUT_DIR, "test_enriched.csv"), sep=";", index=False)

    submission = make_prediction_submission(predictions)
    submission["date"] = submission["date"].dt.strftime("%Y-%m-%d")
    submission["daily_sum_prediction"] = submission["daily_sum_prediction"].round(2)
    submission["daily_average_prediction"] = submission["daily_average_prediction"].round(4)
    submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), sep=";", index=False)

    print(f"Готово: {OUTPUT_DIR}/train_enriched.csv ({len(train_enriched)} строк)")
    print(f"Готово: {OUTPUT_DIR}/test_enriched.csv ({len(test_enriched)} строк)")
    print(f"Готово: {OUTPUT_DIR}/submission.csv ({len(submission)} строк)")


if __name__ == "__main__":
    main()
