# Вставить целиком в одну ячейку Google Colab.

import importlib.util
import os
import subprocess
import sys
import warnings

if any(importlib.util.find_spec(name) is None for name in ("lightgbm", "xgboost", "catboost")):
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "lightgbm", "xgboost", "catboost", "pandas", "numpy",
        "matplotlib", "seaborn",
    ])

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from IPython.display import display


# =========================
# 1. Настройки
# =========================

DATA_DIR = "/content/dataset"
OUTPUT_PATH = "/content/submission_blend.csv"

TRAIN_PATH = os.path.join(DATA_DIR, "labels_day_train.csv")
TEST_PATH = os.path.join(DATA_DIR, "labels_day_test.csv")

ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
RANDOM_STATE = 42

# Rolling validation:
# train до начала месяца, validation = полный следующий месяц.
FOLDS = [
    ("2025-01-01", "2025-06-01", "2025-06-30"),
    ("2025-01-01", "2025-07-01", "2025-07-31"),
    ("2025-01-01", "2025-08-01", "2025-08-31"),
    ("2025-01-01", "2025-09-01", "2025-09-30"),
    ("2025-01-01", "2025-10-01", "2025-10-31"),
]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid")


# =========================
# 2. Метрика и признаки
# =========================

def wape_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return 0.0
    return max(0.0, 1.0 - np.abs(y_true - y_pred).sum() / denominator)


def ru_holidays():
    return pd.to_datetime([
        "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
        "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08",
        "2025-02-23", "2025-03-08", "2025-05-01", "2025-05-02",
        "2025-05-08", "2025-05-09", "2025-06-12", "2025-06-13",
        "2025-11-03", "2025-11-04", "2025-12-31",
    ])


def add_calendar_features(df):
    df = df.copy()
    dt = pd.to_datetime(df["date"])
    df["hour_val"] = df["hour"].astype(int)
    df["dayofweek"] = dt.dt.dayofweek.astype(int)
    df["day"] = dt.dt.day.astype(int)
    df["month"] = dt.dt.month.astype(int)
    df["dayofyear"] = dt.dt.dayofyear.astype(int)
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["quarter"] = dt.dt.quarter.astype(int)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    holidays = ru_holidays()
    df["is_holiday"] = dt.isin(holidays).astype(int)
    df["is_off_day"] = (
        (df["is_weekend"] == 1) | (df["is_holiday"] == 1)
    ).astype(int)
    df["is_preholiday"] = (dt + pd.Timedelta(days=1)).isin(holidays).astype(int)

    for period, value in [
        ("hour", df["hour_val"]),
        ("dow", df["dayofweek"]),
        ("month", df["month"]),
    ]:
        max_value = {"hour": 24, "dow": 7, "month": 12}[period]
        df[f"{period}_sin"] = np.sin(2 * np.pi * value / max_value)
        df[f"{period}_cos"] = np.cos(2 * np.pi * value / max_value)

    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    hour_in_week = df["dayofweek"] * 24 + df["hour_val"]
    for k in [1, 2, 3, 4]:
        df[f"fourier_week_sin_{k}"] = np.sin(2 * np.pi * k * hour_in_week / 168)
        df[f"fourier_week_cos_{k}"] = np.cos(2 * np.pi * k * hour_in_week / 168)
    return df


def add_profiles(df, train_mask):
    df = df.copy()
    source = df.loc[train_mask]

    group_specs = [
        (
            ["route", "hour", "dayofweek"],
            ["profile_rhd_mean", "profile_rhd_median"],
        ),
        (
            ["route", "hour", "is_off_day"],
            ["profile_rho_mean", "profile_rho_median"],
        ),
        (
            ["route", "hour"],
            ["profile_rh_mean", "profile_rh_median"],
        ),
        (
            ["route", "dayofweek"],
            ["profile_rd_mean"],
        ),
    ]

    for keys, names in group_specs:
        aggregations = {}
        for name in names:
            aggregations[name] = "median" if "median" in name else "mean"
        stats = source.groupby(keys)["boardings"].agg(**aggregations).reset_index()
        df = df.merge(stats, on=keys, how="left")

    profile_columns = [c for c in df.columns if c.startswith("profile_")]
    df[profile_columns] = df[profile_columns].fillna(0)
    return df


def make_grid():
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    return pd.MultiIndex.from_product(
        [ROUTES, dates, range(24)],
        names=["route", "date", "hour"],
    ).to_frame(index=False)


def prepare_data():
    train = pd.read_csv(TRAIN_PATH, sep=";", encoding="utf-8-sig")
    test = pd.read_csv(TEST_PATH, sep=";", encoding="utf-8-sig")
    observed = pd.concat([train, test], ignore_index=True)
    observed["date"] = pd.to_datetime(observed["date"])
    observed = (
        observed.groupby(["route", "date", "hour"], as_index=False)["boardings"]
        .sum()
    )

    grid = make_grid()
    grid["date"] = pd.to_datetime(grid["date"])
    data = grid.merge(
        observed,
        on=["route", "date", "hour"],
        how="left",
    )
    data["boardings"] = data["boardings"].fillna(0).astype(float)
    return add_calendar_features(data)


# =========================
# 3. Модели
# =========================

LGB_PARAMS = {
    "objective": "poisson",
    "n_estimators": 1600,
    "learning_rate": 0.01808007542584232,
    "num_leaves": 16,
    "max_depth": 3,
    "min_child_samples": 90,
    "subsample": 0.9160972855471983,
    "subsample_freq": 1,
    "colsample_bytree": 0.7305463944096473,
    "reg_alpha": 0.0035005040816287877,
    "reg_lambda": 3.537920492835122,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}

XGB_PARAMS = {
    "objective": "count:poisson",
    "n_estimators": 1300,
    "learning_rate": 0.025,
    "max_depth": 4,
    "min_child_weight": 10,
    "subsample": 0.90,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.01,
    "reg_lambda": 4.0,
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
}

CAT_PARAMS = {
    "loss_function": "RMSE",
    "iterations": 1300,
    "learning_rate": 0.025,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_seed": RANDOM_STATE,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
}


def make_models():
    return {
        "lightgbm": lgb.LGBMRegressor(**LGB_PARAMS),
        "xgboost": XGBRegressor(**XGB_PARAMS),
        "catboost": CatBoostRegressor(**CAT_PARAMS),
    }


def feature_columns(df):
    return [
        c for c in df.columns
        if c not in {"route", "date", "boardings"}
    ]


def fit_predict(train_df, val_df, features):
    predictions = {}
    models = make_models()

    for name, model in models.items():
        model.fit(train_df[features], train_df["boardings"])
        predictions[name] = np.maximum(
            0,
            model.predict(val_df[features]),
        )
        print(f"{name}: готово")

    return predictions


def choose_blend_weights(y_true, predictions):
    names = list(predictions)
    matrix = np.column_stack([predictions[name] for name in names])
    best = None

    # Сетка весов с шагом 0.05. Это дешевле и прозрачнее Optuna.
    for w_lgb in np.arange(0, 1.001, 0.05):
        for w_xgb in np.arange(0, 1.001 - w_lgb, 0.05):
            w_cat = 1.0 - w_lgb - w_xgb
            if w_cat < -1e-9:
                continue
            weights = np.array([w_lgb, w_xgb, w_cat])
            pred = np.maximum(0, matrix @ weights)
            score = wape_score(y_true, pred)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "weights": dict(zip(names, weights)),
                }
    return best


def blend(predictions, weights):
    return np.maximum(
        0,
        sum(predictions[name] * weight for name, weight in weights.items()),
    )


# =========================
# 4. Rolling validation
# =========================

full_df = prepare_data()
features = feature_columns(full_df)
oof_parts = []
fold_reports = []

print(f"Фичей: {len(features)}")
print(f"Строк в полной сетке: {len(full_df)}")

for fold_id, (train_start, val_start, val_end) in enumerate(FOLDS, start=1):
    train_mask = (
        (full_df["date"] >= train_start)
        & (full_df["date"] < val_start)
    )
    val_mask = (
        (full_df["date"] >= val_start)
        & (full_df["date"] < pd.Timestamp(val_end) + pd.Timedelta(days=1))
    )

    fold_df = add_profiles(full_df, train_mask)
    train_part = fold_df.loc[train_mask].copy()
    val_part = fold_df.loc[val_mask].copy()
    fold_features = feature_columns(fold_df)

    print(f"\nFold {fold_id}: train < {val_start}, validation {val_start} — {val_end}")
    preds = fit_predict(train_part, val_part, fold_features)
    y = val_part["boardings"].to_numpy()
    best = choose_blend_weights(y, preds)
    val_part["blend_pred"] = blend(preds, best["weights"])

    print("Веса:", best["weights"])
    print(f"Fold WAPE-score: {best['score']:.5f}")

    val_part["fold_id"] = fold_id
    for name, pred in preds.items():
        val_part[f"{name}_pred"] = pred

    fold_reports.append({
        "fold": fold_id,
        "validation": f"{val_start} — {val_end}",
        "score": best["score"],
        **best["weights"],
    })
    oof_parts.append(val_part)

all_oof = pd.concat(oof_parts, ignore_index=True)
best_fold_report = max(fold_reports, key=lambda row: row["score"])
best_fold_id = best_fold_report["fold"]
oof = all_oof[all_oof["fold_id"] == best_fold_id].copy()

print(
    f"\nВыбран лучший validation split: fold {best_fold_id}, "
    f"{best_fold_report['validation']}, "
    f"score={best_fold_report['score']:.5f}"
)

oof_score = wape_score(oof["boardings"], oof["blend_pred"])
oof_model_scores = {
    name: wape_score(oof["boardings"], oof[f"{name}_pred"])
    for name in ["lightgbm", "xgboost", "catboost"]
}

oof_best = choose_blend_weights(
    oof["boardings"].to_numpy(),
    {name: oof[f"{name}_pred"].to_numpy() for name in oof_model_scores},
)
oof["final_blend_pred"] = blend(
    {name: oof[f"{name}_pred"].to_numpy() for name in oof_model_scores},
    oof_best["weights"],
)

print("\n" + "=" * 70)
print("ROLLING VALIDATION")
print("=" * 70)
display(pd.DataFrame(fold_reports).round(5))
print("OOF scores отдельных моделей:", {
    k: round(v, 5) for k, v in oof_model_scores.items()
})
print("OOF оптимальные веса:", oof_best["weights"])
print(f"OOF WAPE-score blend: {wape_score(oof['boardings'], oof['final_blend_pred']):.5f}")
print("=" * 70)


# =========================
# 5. Сводка по маршрутам
# =========================

route_scores = {}
route_rows = []
for route in ROUTES:
    part = oof[oof["route"] == route]
    score = wape_score(part["boardings"], part["final_blend_pred"])
    route_scores[route] = score
    route_rows.append({
        "route": route,
        "wape_score": score,
        "actual_sum": part["boardings"].sum(),
        "rows": len(part),
    })

route_report = pd.DataFrame(route_rows)
display(route_report.round(5))


# =========================
# 6. Dataviz
# =========================

def plot_all_routes(val_df, scores, pred_col="final_blend_pred"):
    ncols = 2
    nrows = int(np.ceil(len(ROUTES) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18, 4.4 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    for i, route in enumerate(ROUTES):
        ax = axes[i]
        part = (
            val_df[val_df["route"] == route]
            .groupby("date")[["boardings", pred_col]]
            .sum()
            .reset_index()
        )
        ax.plot(part["date"], part["boardings"], label="Факт", linewidth=1.8)
        ax.plot(
            part["date"],
            part[pred_col],
            label="Blend",
            linewidth=1.8,
            linestyle="--",
        )
        ax.set_title(
            f"Маршрут №{route} | WAPE-score: {scores.get(route, np.nan):.4f}",
            fontweight="bold",
        )
        ax.set_xlabel("Дата")
        ax.set_ylabel("Посадки / день")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(fontsize=8)

    for i in range(len(ROUTES), len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def plot_hour_heatmaps(val_df):
    ncols = 2
    nrows = int(np.ceil(len(ROUTES) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18, 4.5 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    for i, route in enumerate(ROUTES):
        part = val_df[val_df["route"] == route].copy()
        part["weekday"] = part["date"].dt.dayofweek
        actual = part.pivot_table(
            index="weekday",
            columns="hour",
            values="boardings",
            aggfunc="mean",
        )
        pred = part.pivot_table(
            index="weekday",
            columns="hour",
            values="final_blend_pred",
            aggfunc="mean",
        )
        error = pred - actual
        sns.heatmap(
            error,
            ax=axes[i],
            cmap="RdBu_r",
            center=0,
            cbar=True,
        )
        axes[i].set_title(f"Маршрут №{route}: средняя ошибка, день недели × час")
        axes[i].set_xlabel("Час")
        axes[i].set_ylabel("День недели, 0=Пн")

    for i in range(len(ROUTES), len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def plot_global_dashboard(val_df, scores):
    daily = (
        val_df.groupby("date")[["boardings", "final_blend_pred"]]
        .sum()
        .reset_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)

    axes[0, 0].plot(daily["date"], daily["boardings"], label="Факт", linewidth=2)
    axes[0, 0].plot(
        daily["date"],
        daily["final_blend_pred"],
        label="Blend",
        linewidth=2,
        linestyle="--",
    )
    axes[0, 0].set_title(
        f"Суммарный дневной поток | WAPE-score: "
        f"{wape_score(daily['boardings'], daily['final_blend_pred']):.4f}"
    )
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis="x", rotation=35)

    score_series = pd.Series(scores).sort_index()
    axes[0, 1].bar(score_series.index.astype(str), score_series.values)
    axes[0, 1].set_title("WAPE-score по маршрутам")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xlabel("Маршрут")

    hourly = (
        val_df.groupby("hour")[["boardings", "final_blend_pred"]]
        .mean()
        .reset_index()
    )
    axes[1, 0].plot(hourly["hour"], hourly["boardings"], label="Факт", marker="o")
    axes[1, 0].plot(
        hourly["hour"],
        hourly["final_blend_pred"],
        label="Blend",
        marker="o",
    )
    axes[1, 0].set_title("Средний почасовой профиль всех маршрутов")
    axes[1, 0].set_xlabel("Час")
    axes[1, 0].legend()

    errors = val_df["final_blend_pred"] - val_df["boardings"]
    axes[1, 1].hist(errors, bins=60, color="#7c3aed", alpha=0.85)
    axes[1, 1].axvline(0, color="black", linewidth=1)
    axes[1, 1].set_title("Распределение ошибок blend")
    axes[1, 1].set_xlabel("Прогноз − факт")

    plt.show()


plot_global_dashboard(oof, route_scores)
plot_all_routes(oof, route_scores)
plot_hour_heatmaps(oof)


# =========================
# 7. Финальное обучение на train + test
# =========================

final_train_mask = full_df["date"] < "2025-11-01"
final_df = add_profiles(full_df, final_train_mask)
final_train = final_df.loc[final_train_mask].copy()
final_test = final_df.loc[~final_train_mask].copy()
final_features = feature_columns(final_df)

final_models = make_models()
final_predictions = {}

print("\nФинальное обучение на январе-октябре...")
for name, model in final_models.items():
    model.fit(final_train[final_features], final_train["boardings"])
    final_predictions[name] = np.maximum(
        0,
        model.predict(final_test[final_features]),
    )
    print(f"{name}: готово")

final_pred = blend(final_predictions, oof_best["weights"])

submission = final_test[["route", "date", "hour"]].copy()
submission["prediction"] = np.round(final_pred).astype(int)
submission["date"] = submission["date"].dt.strftime("%Y-%m-%d")
submission = submission.sort_values(["route", "date", "hour"]).reset_index(drop=True)
submission.to_csv(OUTPUT_PATH, sep=";", index=False, encoding="utf-8")

print("\n" + "=" * 70)
print(f"Submission сохранен: {OUTPUT_PATH}")
print(f"Строк: {len(submission)}")
print(f"Blend weights: {oof_best['weights']}")
print("=" * 70)

assert len(submission) == 14640
assert set(submission["route"]) == set(ROUTES)
assert submission[["route", "date", "hour"]].duplicated().sum() == 0
assert submission["prediction"].ge(0).all()
print("Проверка submission пройдена.")

# Для скачивания в Colab:
# from google.colab import files
# files.download(OUTPUT_PATH)
