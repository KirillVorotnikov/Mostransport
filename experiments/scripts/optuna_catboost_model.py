import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = "/kaggle/input/datasets/sauzzeth/dataset"
TRAIN_LABELS = f"{DATA_DIR}/labels_day_train.csv"
TEST_LABELS = f"{DATA_DIR}/labels_day_test.csv"
ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
N_TRIALS = 60
VALIDATION_PLOT = "/kaggle/working/validation_all_routes_optuna.png"
WEATHER_DATASET_DIR = Path("/kaggle/input/datasets/sauzzeth/moscow-weather-2025")
WEATHER_FILENAME = "moscow_weather_2025.csv"

BASE_MODEL_PARAMS = {
    "iterations": 1500,
    "random_seed": 42,
    "thread_count": -1,
    "verbose": False,
    "allow_writing_files": False,
    "bootstrap_type": "Bernoulli",
}


def calculate_wape_score(y_true, y_pred):
    denominator = np.sum(y_true)
    if denominator == 0:
        return 0.0
    return max(0.0, 1.0 - np.sum(np.abs(y_true - y_pred)) / denominator)


def quantile_loss_candidates():
    return [f"Quantile:alpha={alpha / 10:.1f}" for alpha in range(1, 10)]


def get_loss_function(trial):
    loss_family = trial.suggest_categorical(
        "loss_family",
        ["RMSE", "MAE", "Huber", *quantile_loss_candidates()],
    )
    if loss_family == "Huber":
        return f"Huber:delta={trial.suggest_float('huber_delta', 0.5, 5.0)}"
    return loss_family


def get_model_params(trial=None):
    if trial is None:
        loss = "Quantile:alpha=0.7"
        return {
            **BASE_MODEL_PARAMS,
            "loss_function": loss,
            "eval_metric": loss,
            "learning_rate": 0.03,
            "depth": 4,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "subsample": 0.8,
            "rsm": 0.8,
        }

    loss = get_loss_function(trial)
    return {
        **BASE_MODEL_PARAMS,
        "loss_function": loss,
        "eval_metric": loss,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "rsm": trial.suggest_float("rsm", 0.6, 1.0),
    }


def get_russian_holidays_2025():
    return set(pd.to_datetime([
        "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
        "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08",
        "2025-02-23", "2025-03-08", "2025-05-01", "2025-05-02",
        "2025-05-08", "2025-05-09", "2025-06-12", "2025-06-13",
        "2025-11-03", "2025-11-04", "2025-12-31",
    ]))


def get_russian_non_working_days_2025():
    return set(pd.to_datetime([
        "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
        "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08",
        "2025-01-11", "2025-01-12", "2025-01-18", "2025-01-19",
        "2025-01-25", "2025-01-26", "2025-02-01", "2025-02-02",
        "2025-02-08", "2025-02-09", "2025-02-15", "2025-02-16",
        "2025-02-22", "2025-02-23", "2025-03-01", "2025-03-02",
        "2025-03-08", "2025-03-09", "2025-03-15", "2025-03-16",
        "2025-03-22", "2025-03-23", "2025-03-29", "2025-03-30",
        "2025-04-05", "2025-04-06", "2025-04-12", "2025-04-13",
        "2025-04-19", "2025-04-20", "2025-04-26", "2025-04-27",
        "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04",
        "2025-05-08", "2025-05-09", "2025-05-10", "2025-05-11",
        "2025-05-17", "2025-05-18", "2025-05-24", "2025-05-25",
        "2025-05-31", "2025-06-01", "2025-06-07", "2025-06-08",
        "2025-06-12", "2025-06-13", "2025-06-14", "2025-06-15",
        "2025-06-21", "2025-06-22", "2025-06-28", "2025-06-29",
        "2025-07-05", "2025-07-06", "2025-07-12", "2025-07-13",
        "2025-07-19", "2025-07-20", "2025-07-26", "2025-07-27",
        "2025-08-02", "2025-08-03", "2025-08-09", "2025-08-10",
        "2025-08-16", "2025-08-17", "2025-08-23", "2025-08-24",
        "2025-08-30", "2025-08-31", "2025-09-06", "2025-09-07",
        "2025-09-13", "2025-09-14", "2025-09-20", "2025-09-21",
        "2025-09-27", "2025-09-28", "2025-10-04", "2025-10-05",
        "2025-10-11", "2025-10-12", "2025-10-18", "2025-10-19",
        "2025-10-25", "2025-10-26", "2025-11-02", "2025-11-03",
        "2025-11-04", "2025-11-08", "2025-11-09", "2025-11-15",
        "2025-11-16", "2025-11-22", "2025-11-23", "2025-11-29",
        "2025-11-30", "2025-12-06", "2025-12-07", "2025-12-13",
        "2025-12-14", "2025-12-20", "2025-12-21", "2025-12-27",
        "2025-12-28", "2025-12-31",
        "2025-03-07", "2025-04-30", "2025-06-11", "2025-11-01",
    ]))


def create_full_grid():
    return pd.MultiIndex.from_product(
        [ROUTES, pd.date_range("2025-01-01", "2025-12-31"), range(24)],
        names=["route", "date", "hour"],
    ).to_frame(index=False)


def extract_seasonal_features(df):
    df = df.copy()
    dt = pd.to_datetime(df["date"])
    df["hour_val"] = df["hour"]
    df["dayofweek"] = dt.dt.dayofweek
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["dayofyear"] = dt.dt.dayofyear
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["quarter"] = dt.dt.quarter
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)

    holidays = get_russian_holidays_2025()
    non_working_days = get_russian_non_working_days_2025()
    df["is_holiday"] = dt.isin(holidays).astype(int)
    df["is_non_working_day"] = dt.isin(non_working_days).astype(int)
    df["is_off_day"] = df["is_non_working_day"]
    df["is_preholiday"] = (
        (dt + pd.Timedelta(days=1)).isin(holidays).astype(int)
    )

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_val"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_val"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    hour_in_week = df["dayofweek"] * 24 + df["hour_val"]
    for k in (1, 2, 3):
        df[f"fourier_week_sin_{k}"] = np.sin(2 * np.pi * k * hour_in_week / 168)
        df[f"fourier_week_cos_{k}"] = np.cos(2 * np.pi * k * hour_in_week / 168)

    hour_in_year = (df["dayofyear"] - 1) * 24 + df["hour_val"]
    for k in (1, 2):
        df[f"fourier_year_sin_{k}"] = np.sin(2 * np.pi * k * hour_in_year / 8766)
        df[f"fourier_year_cos_{k}"] = np.cos(2 * np.pi * k * hour_in_year / 8766)
    return df


def add_historical_profiles(df, train_mask):
    train = df.loc[train_mask]
    result = df.copy()
    profiles = [
        (["route", "hour", "dayofweek"], {
            "profile_rhd_mean": "mean",
            "profile_rhd_median": "median",
            "profile_rhd_std": "std",
        }),
        (["route", "hour", "is_off_day"], {
            "profile_rho_mean": "mean",
            "profile_rho_median": "median",
        }),
        (["route", "hour"], {
            "profile_rh_mean": "mean",
            "profile_rh_median": "median",
        }),
    ]
    for keys, aggregations in profiles:
        profile = train.groupby(keys)["boardings"].agg(**aggregations).reset_index()
        result = result.merge(profile, on=keys, how="left")
    return result


def load_weather_features():
    if WEATHER_DATASET_DIR.is_file():
        weather_path = WEATHER_DATASET_DIR
    else:
        matches = list(WEATHER_DATASET_DIR.glob("**/*.csv"))
        if not matches:
            raise FileNotFoundError(f"В датасете {WEATHER_DATASET_DIR} не найден CSV с погодой")
        weather_path = next(
            (path for path in matches if path.name == WEATHER_FILENAME), matches[0]
        )

    weather = pd.read_csv(weather_path, parse_dates=["date"])
    required = {"date", "hour", "temperature", "precipitation", "wind", "thunder"}
    missing = required - set(weather.columns)
    if missing:
        raise ValueError(f"В {weather_path} отсутствуют колонки: {sorted(missing)}")
    return weather[["date", "hour", "temperature", "precipitation", "wind", "thunder"]]


def load_data():
    train_raw = pd.read_csv(TRAIN_LABELS, sep=";")
    test_raw = pd.read_csv(TEST_LABELS, sep=";")
    historical = pd.concat([train_raw, test_raw], ignore_index=True)
    historical["date"] = pd.to_datetime(historical["date"])

    full_df = create_full_grid().merge(
        historical[["route", "date", "hour", "boardings"]],
        on=["route", "date", "hour"],
        how="left",
    )
    full_df["boardings"] = full_df["boardings"].fillna(0)
    return extract_seasonal_features(full_df).merge(
        load_weather_features(), on=["date", "hour"], how="left"
    )


def visualize_validation(val_df):
    scores = {}
    for route in ROUTES:
        route_df = val_df[val_df["route"] == route]
        scores[route] = calculate_wape_score(route_df["boardings"], route_df["val_pred"])

    fig, axes = plt.subplots(2, 5, figsize=(24, 9), sharey=False)
    fig.suptitle("CatBoost + Optuna: факт и прогноз по всем маршрутам", fontsize=18)
    for ax, route in zip(axes.flat, ROUTES):
        daily = val_df[val_df["route"] == route].groupby("date")[["boardings", "val_pred"]].sum().reset_index()
        ax.plot(daily["date"], daily["boardings"], label="Факт", linewidth=1.7)
        ax.plot(daily["date"], daily["val_pred"], label="Прогноз", linestyle="--", linewidth=1.5)
        ax.set_title(f"Маршрут {route} | WAPE {scores[route]:.4f}")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Посадки в день")
    axes[1, 0].set_ylabel("Посадки в день")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(VALIDATION_PLOT, dpi=150, bbox_inches="tight")
    plt.show()
    return scores


def main():
    full_df = load_data()
    train_mask = (full_df["date"] >= "2025-01-01") & (full_df["date"] <= "2025-08-31")
    val_mask = (full_df["date"] >= "2025-09-01") & (full_df["date"] <= "2025-10-31")

    prepared = add_historical_profiles(full_df, train_mask)
    feature_cols = [c for c in prepared.columns if c not in {"route", "date", "boardings"}]
    train_df = prepared.loc[train_mask]
    val_df = prepared.loc[val_mask].copy()

    def objective(trial):
        model = CatBoostRegressor(**get_model_params(trial))
        model.fit(
            train_df[feature_cols], train_df["boardings"],
            eval_set=(val_df[feature_cols], val_df["boardings"]),
            early_stopping_rounds=50,
        )
        predictions = np.maximum(0, model.predict(val_df[feature_cols]))
        return calculate_wape_score(val_df["boardings"].values, predictions)

    study = optuna.create_study(direction="maximize", study_name="catboost_wape")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("Лучший WAPE:", f"{study.best_value:.6f}")
    print("Лучшие параметры:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    model = CatBoostRegressor(**get_model_params_from_best(study.best_params))
    model.fit(
        train_df[feature_cols], train_df["boardings"],
        eval_set=(val_df[feature_cols], val_df["boardings"]),
        early_stopping_rounds=50,
    )
    val_df["val_pred"] = np.maximum(0, model.predict(val_df[feature_cols]))

    overall_score = calculate_wape_score(val_df["boardings"], val_df["val_pred"])
    print(f"ОБЩИЙ WAPE SCORE (Sep-Oct): {overall_score:.4f}")
    print(f"Лучшее число итераций: {model.get_best_iteration() + 1}")
    print(f"График сохранен: {VALIDATION_PLOT}")
    visualize_validation(val_df)


def get_model_params_from_best(best_params):
    trial_params = dict(best_params)
    if trial_params.get("loss_family") == "Huber":
        loss = f"Huber:delta={trial_params['huber_delta']}"
    else:
        loss = trial_params["loss_family"]
    return {
        **BASE_MODEL_PARAMS,
        "loss_function": loss,
        "eval_metric": loss,
        "learning_rate": trial_params["learning_rate"],
        "depth": trial_params["depth"],
        "l2_leaf_reg": trial_params["l2_leaf_reg"],
        "random_strength": trial_params["random_strength"],
        "subsample": trial_params["subsample"],
        "rsm": trial_params["rsm"],
    }


if __name__ == "__main__":
    main()
