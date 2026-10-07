import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Настройки отрисовки графиков в ноутбуке
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

# ==========================================
# 1. КОНСТАНТЫ И ПУТИ К ДАННЫМ
# ==========================================
DATA_DIR = "/kaggle/input/datasets/sauzzeth/dataset"
OUTPUT_PATH = "/kaggle/working/submission.csv"

TRAIN_LABELS = f"{DATA_DIR}/labels_day_train.csv"
TEST_LABELS  = f"{DATA_DIR}/labels_day_test.csv"

ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]

# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И МЕТРИКИ
# ==========================================
def calculate_wape_score(y_true, y_pred):
    """
    Вычисление WAPE-score = max(0, 1 - sum(|y - y_hat|) / sum(y))
    """
    denom = np.sum(y_true)
    if denom == 0:
        return 0.0
    wape = np.sum(np.abs(y_true - y_pred)) / denom
    return max(0.0, 1.0 - wape)

def get_russian_holidays_2025():
    """
    Официальные нерабочие праздничные и переносы в РФ на 2025 год
    """
    holidays_2025 = [
        "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
        "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08",
        "2025-02-23", "2025-03-08", "2025-05-01", "2025-05-02",
        "2025-05-08", "2025-05-09", "2025-06-12", "2025-06-13",
        "2025-11-03", "2025-11-04", "2025-12-31"
    ]
    return set(pd.to_datetime(holidays_2025))

def create_full_grid():
    """
    Создание полной сетки ключей (Маршрут x Дата x Час)
    для всего периода 2025-01-01 ... 2025-12-31
    """
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    hours = list(range(24))
    
    grid = pd.MultiIndex.from_product(
        [ROUTES, dates, hours], 
        names=["route", "date", "hour"]
    ).to_frame().reset_index(drop=True)
    
    return grid

# ==========================================
# 3. ГЕНЕРАЦИЯ СЕЗОННЫХ ПРИЗНАКОВ
# ==========================================
def extract_seasonal_features(df):
    """
    Генерация календарных и гармонических признаков
    """
    df = df.copy()
    dt = pd.to_datetime(df["date"])
    
    # Календарные атрибуты
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
    
    # Праздники и переносы
    ru_holidays = get_russian_holidays_2025()
    df["is_holiday"] = dt.isin(ru_holidays).astype(int)
    df["is_off_day"] = ((df["is_weekend"] == 1) | (df["is_holiday"] == 1)).astype(int)
    df["is_preholiday"] = (dt + pd.Timedelta(days=1)).isin(ru_holidays).astype(int)
    
    # Тригонометрическое кодирование (Sin/Cos)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_val"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_val"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    
    # Гармоники Фурье
    hour_in_week = df["dayofweek"] * 24 + df["hour_val"]
    for k in [1, 2, 3]:
        df[f"fourier_week_sin_{k}"] = np.sin(2 * np.pi * k * hour_in_week / 168.0)
        df[f"fourier_week_cos_{k}"] = np.cos(2 * np.pi * k * hour_in_week / 168.0)
        
    hour_in_year = (df["dayofyear"] - 1) * 24 + df["hour_val"]
    for k in [1, 2]:
        df[f"fourier_year_sin_{k}"] = np.sin(2 * np.pi * k * hour_in_year / 8766.0)
        df[f"fourier_year_cos_{k}"] = np.cos(2 * np.pi * k * hour_in_year / 8766.0)
        
    return df

# ==========================================
# 4. РАСЧЕТ ИСТОРИЧЕСКИХ ПРОФИЛЕЙ (TARGET ENCODING)
# ==========================================
def add_historical_profiles(df, train_mask):
    """
    Расчет средних/медиан целевой переменной строго по train_mask для исключения утечки
    """
    df = df.copy()
    train_stats_data = df[train_mask]
    
    # 1. Route x Hour x DayOfWeek
    group_rhd = train_stats_data.groupby(["route", "hour", "dayofweek"])["boardings"].agg(
        profile_rhd_mean="mean",
        profile_rhd_median="median",
        profile_rhd_std="std"
    ).reset_index()

    # 2. Route x Hour x IsOffDay
    group_rho = train_stats_data.groupby(["route", "hour", "is_off_day"])["boardings"].agg(
        profile_rho_mean="mean",
        profile_rho_median="median"
    ).reset_index()

    # 3. Route x Hour
    group_rh = train_stats_data.groupby(["route", "hour"])["boardings"].agg(
        profile_rh_mean="mean",
        profile_rh_median="median"
    ).reset_index()

    df = df.merge(group_rhd, on=["route", "hour", "dayofweek"], how="left")
    df = df.merge(group_rho, on=["route", "hour", "is_off_day"], how="left")
    df = df.merge(group_rh, on=["route", "hour"], how="left")
    
    return df

# ==========================================
# 5. ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ В НОУТБУКЕ
# ==========================================
def visualize_validation_results(val_df, route_scores, overall_wape):
    """
    Отображение агрегированного и помаршрутного сравнения факта и прогноза прямо в ячейке
    """
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1.2])

    # 1. Агрегированный суточный пассажиропоток (суммарно по всем маршрутам)
    ax1 = fig.add_subplot(gs[0, :])
    daily = val_df.groupby("date")[["boardings", "val_pred"]].sum().reset_index()
    ax1.plot(daily["date"], daily["boardings"], label="Факт (boardings)", color="#1f77b4", linewidth=2.5)
    ax1.plot(daily["date"], daily["val_pred"], label="Прогноз (prediction)", color="#ff7f0e", linewidth=2, linestyle="--")
    ax1.set_title(f"Суммарный суточный пассажиропоток (Все маршруты) | Общий WAPE Score: {overall_wape:.4f}", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Дата")
    ax1.set_ylabel("Число посадок в день")
    ax1.legend(loc="upper right", frameon=True)

    # 2. WAPE Score по каждому маршруту (Bar Chart)
    ax2 = fig.add_subplot(gs[1, 0])
    routes_str = [str(r) for r in route_scores.keys()]
    scores = list(route_scores.values())
    bars = ax2.bar(routes_str, scores, color="#2ca02c", alpha=0.85, edgecolor="black")
    ax2.axhline(0.48, color="red", linestyle=":", linewidth=1.5, label="Базовый порог (~0.48)")
    ax2.set_title("WAPE-Score по маршрутам (Сентябрь-Октябрь)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Маршрут")
    ax2.set_ylabel("WAPE-Score")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right")
    
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    # 3. Почасовой срез за неделю (1–7 сентября)
    ax3 = fig.add_subplot(gs[1, 1])
    sample_mask = (val_df["date"] >= "2025-09-01") & (val_df["date"] <= "2025-09-07")
    sample_df = val_df[sample_mask].groupby("datetime")[["boardings", "val_pred"]].sum().reset_index()
    ax3.plot(sample_df["datetime"], sample_df["boardings"], label="Факт", color="#1f77b4", alpha=0.8)
    ax3.plot(sample_df["datetime"], sample_df["val_pred"], label="Прогноз", color="#d62728", linestyle="--")
    ax3.set_title("Почасовой профиль (1–7 Сентября 2025)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Дата и время")
    ax3.set_ylabel("Посадки в час")
    ax3.tick_params(axis='x', rotation=30)
    ax3.legend()

    # 4. Сравнение суточной динамики для маршрутов №17 и №11
    top_routes = [17, 11]
    for idx, r in enumerate(top_routes):
        ax = fig.add_subplot(gs[2, idx])
        r_df = val_df[val_df["route"] == r].groupby("date")[["boardings", "val_pred"]].sum().reset_index()
        score_r = route_scores.get(r, 0.0)
        ax.plot(r_df["date"], r_df["boardings"], label="Факт", color="#1f77b4", linewidth=1.8)
        ax.plot(r_df["date"], r_df["val_pred"], label="Прогноз", color="#ff7f0e", linestyle="--", linewidth=1.8)
        ax.set_title(f"Маршрут №{r} | WAPE Score: {score_r:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Суточные посадки")
        ax.legend()

    plt.tight_layout()
    plt.show()

# ==========================================
# 6. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ==========================================
df_train_raw = pd.read_csv(TRAIN_LABELS, sep=";")
df_test_raw  = pd.read_csv(TEST_LABELS, sep=";")

df_historical_raw = pd.concat([df_train_raw, df_test_raw], ignore_index=True)
df_historical_raw["date"] = pd.to_datetime(df_historical_raw["date"])

# Формируем единую сетку на весь 2025 год
full_df = create_full_grid()
full_df["date"] = pd.to_datetime(full_df["date"])

full_df = full_df.merge(
    df_historical_raw[["route", "date", "hour", "boardings"]], 
    on=["route", "date", "hour"], 
    how="left"
)
full_df["boardings"] = full_df["boardings"].fillna(0)
full_df = extract_seasonal_features(full_df)

# ==========================================
# 7.0 ПОДБОР ГИПЕРПАРАМЕТРОВ (OPTUNA)
# ==========================================
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Готовим фичи/сплиты один раз, чтобы не пересчитывать в каждом trial
val_train_mask = (full_df["date"] >= "2025-01-01") & (full_df["date"] <= "2025-08-31")
val_full_df = add_historical_profiles(full_df, train_mask=val_train_mask)

ignore_cols = ["route", "date", "boardings"]
feature_cols = [c for c in val_full_df.columns if c not in ignore_cols]

train_split = val_full_df[val_train_mask]
val_split = val_full_df[(val_full_df["date"] >= "2025-09-01") & (val_full_df["date"] <= "2025-10-31")].copy()

FIXED_QUANTILE_ALPHA = 0.7  # зафиксировано как лучшее значение

def objective(trial):
    # Перебираем разные функции потерь
    loss_type = trial.suggest_categorical(
        "objective", ["quantile", "regression", "regression_l1", "huber", "fair", "poisson"]
    )

    params = {
        "objective": loss_type,
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Доп. параметры, специфичные для отдельных лоссов
    if loss_type == "quantile":
        params["alpha"] = FIXED_QUANTILE_ALPHA
        params["metric"] = "quantile"
    elif loss_type == "huber":
        params["alpha"] = trial.suggest_float("huber_alpha", 0.5, 5.0)  # порог Huber, не путать с quantile alpha
        params["metric"] = "huber"
    elif loss_type == "fair":
        params["metric"] = "fair"
    elif loss_type == "poisson":
        params["metric"] = "poisson"
    else:
        params["metric"] = "mae" if loss_type == "regression_l1" else "rmse"

    model = lgb.LGBMRegressor(**params)
    model.fit(
        train_split[feature_cols], train_split["boardings"],
        eval_set=[(val_split[feature_cols], val_split["boardings"])],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    preds = np.maximum(0, model.predict(val_split[feature_cols]))
    score = calculate_wape_score(val_split["boardings"].values, preds)

    return score  # максимизируем WAPE-score

N_TRIALS = 60  # можно увеличить при наличии времени/ресурсов

study = optuna.create_study(direction="maximize", study_name="lgb_wape_tuning")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("=" * 60)
print(f"Лучший WAPE-score на валидации: {study.best_value:.4f}")
print("Лучшие параметры:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
print("=" * 60)

# Собираем финальный словарь параметров из лучшего trial
best_params = study.best_params.copy()
best_loss = best_params.pop("objective")

lgb_params = {
    "objective": best_loss,
    "boosting_type": "gbdt",
    "n_estimators": 1500,
    "learning_rate": best_params["learning_rate"],
    "num_leaves": best_params["num_leaves"],
    "max_depth": best_params["max_depth"],
    "min_child_samples": best_params["min_child_samples"],
    "subsample": best_params["subsample"],
    "colsample_bytree": best_params["colsample_bytree"],
    "reg_alpha": best_params["reg_alpha"],
    "reg_lambda": best_params["reg_lambda"],
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

if best_loss == "quantile":
    lgb_params["alpha"] = FIXED_QUANTILE_ALPHA
    lgb_params["metric"] = "quantile"
elif best_loss == "huber":
    lgb_params["alpha"] = best_params["huber_alpha"]
    lgb_params["metric"] = "huber"
elif best_loss == "fair":
    lgb_params["metric"] = "fair"
elif best_loss == "poisson":
    lgb_params["metric"] = "poisson"
else:
    lgb_params["metric"] = "mae" if best_loss == "regression_l1" else "rmse"

# ==========================================
# 7. ЭТАП 1: ЛОКАЛЬНАЯ ВАЛИДАЦИЯ (Янв-Авг -> Сен-Окт)
# ==========================================
val_train_mask = (full_df["date"] >= "2025-01-01") & (full_df["date"] <= "2025-08-31")
val_full_df = add_historical_profiles(full_df, train_mask=val_train_mask)

ignore_cols = ["route", "date", "boardings"]
feature_cols = [c for c in val_full_df.columns if c not in ignore_cols]

train_split = val_full_df[val_train_mask]
val_split   = val_full_df[(val_full_df["date"] >= "2025-09-01") & (val_full_df["date"] <= "2025-10-31")].copy()

lgb_params = {
    "objective": "quantile",
    "alpha": 0.7,   # 0.55-0.65, подобрать по WAPE на валидации
    "metric": "quantile",
    "boosting_type": "gbdt",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}

model_val = lgb.LGBMRegressor(**lgb_params)
model_val.fit(
    train_split[feature_cols], train_split["boardings"],
    eval_set=[(val_split[feature_cols], val_split["boardings"])],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

val_preds = model_val.predict(val_split[feature_cols])
val_split["val_pred"] = np.maximum(0, val_preds)
val_split["datetime"] = pd.to_datetime(
    val_split["date"].dt.strftime("%Y-%m-%d") + " " + val_split["hour"].astype(str).str.zfill(2) + ":00:00"
)

# Расчет WAPE по каждому маршруту в отдельности
overall_wape_score = calculate_wape_score(val_split["boardings"].values, val_split["val_pred"].values)

route_wape_scores = {}
print("=" * 45)
print(f"{'МАРШРУТ':<10} | {'WAPE SCORE':<12} | {'ФАКТ СУММА':<12}")
print("-" * 45)

for r in ROUTES:
    sub = val_split[val_split["route"] == r]
    r_score = calculate_wape_score(sub["boardings"].values, sub["val_pred"].values)
    route_wape_scores[r] = r_score
    print(f"{r:<10} | {r_score:<12.4f} | {int(sub['boardings'].sum()):<12}")

print("=" * 45)
print(f"ОБЩИЙ WAPE SCORE (Sep-Oct): {overall_wape_score:.4f}")
print("=" * 45)

# Отрисовка графиков в ноутбуке
visualize_validation_results(val_split, route_wape_scores, overall_wape_score)

# ==========================================
# 8. ЭТАП 2: ФИНАЛЬНОЕ ОБУЧЕНИЕ И ПРОГНОЗ (Ноя-Дек)
# ==========================================
final_train_mask = full_df["date"] < "2025-11-01"
final_full_df = add_historical_profiles(full_df, train_mask=final_train_mask)

final_train_split = final_full_df[final_train_mask]
test_split        = final_full_df[(final_full_df["date"] >= "2025-11-01") & (final_full_df["date"] <= "2025-12-31")]

final_model = lgb.LGBMRegressor(**lgb_params)
final_model.fit(final_train_split[feature_cols], final_train_split["boardings"])

test_preds = final_model.predict(test_split[feature_cols])
test_preds = np.maximum(0, test_preds)

# Формирование файла ответа
submission = pd.DataFrame({
    "route": test_split["route"].astype(int),
    "date": test_split["date"].dt.strftime("%Y-%m-%d"),
    "hour": test_split["hour"].astype(int),
    "prediction": np.round(test_preds).astype(int)
})

submission = submission.sort_values(by=["route", "date", "hour"]).reset_index(drop=True)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
submission.to_csv(OUTPUT_PATH, sep=";", index=False)

print(f"Файл submission успешно сохранен в: {OUTPUT_PATH}")
print(f"Строк в submission: {submission.shape[0]}")

# ============================================================
# Лучший WAPE-score на валидации: 0.8770
# Лучшие параметры:
#   objective: poisson
#   learning_rate: 0.01808007542584232
#   num_leaves: 16
#   max_depth: 3
#   min_child_samples: 90
#   subsample: 0.9160972855471983
#   colsample_bytree: 0.7305463944096473
#   reg_alpha: 0.0035005040816287877
#   reg_lambda: 3.537920492835122
# ============================================================
# =============================================
# МАРШРУТ    | WAPE SCORE   | ФАКТ СУММА  
# ---------------------------------------------
# 1          | 0.8949       | 1136093     
# 5          | 0.0000       | 0           
# 7          | 0.8569       | 1350573     
# 11         | 0.8913       | 1907061     
# 12         | 0.9079       | 1983604     
# 17         | 0.9119       | 3063025     
# 25         | 0.8387       | 464537      
# 26         | 0.8550       | 1089964     
# 28         | 0.8356       | 602164      
# 50         | 0.7345       | 1157460     
# =============================================
# ОБЩИЙ WAPE SCORE (Sep-Oct): 0.8736
# =============================================