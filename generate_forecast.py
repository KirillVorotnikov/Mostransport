# generate_forecast.py
import argparse
import subprocess
import sys
import shutil
import importlib.util
import os
import random
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str((ROOT / ".hf-cache").resolve()))

def run_selective():
    print("🚀 [Selective] Запуск ансамбля ARIMA + Chronos-2 + CatBoost...")
    packages = []
    if importlib.util.find_spec("chronos") is None:
        packages += ["chronos-forecasting==2.3.2", "pyarrow>=15"]
    if importlib.util.find_spec("catboost") is None:
        packages += ["catboost==1.2.10"]
    if packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

    import numpy as np
    import pandas as pd
    import torch
    from catboost import CatBoostRegressor
    from chronos import Chronos2Pipeline

    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
    ACTIVE_ROUTES = [r for r in ROUTES if r != 5]
    SELECTED_ROUTES = [11, 12, 25, 28]
    CATBOOST_WEIGHTS = {11: 0.35, 12: 0.10, 25: 0.35, 28: 0.45}
    CHRONOS_WEIGHT = 0.20
    CHRONOS_QUANTILE = "0.7"
    KEYS = ["route", "date", "hour"]

    CATBOOST_PARAMS = {
        11: dict(loss_function="Quantile:alpha=0.7", iterations=1500, learning_rate=0.097, depth=4, l2_leaf_reg=0.0058, random_strength=0.011, bagging_temperature=0.39, border_count=239),
        12: dict(loss_function="MAE", iterations=1500, learning_rate=0.139, depth=4, l2_leaf_reg=0.134, random_strength=0.70, bagging_temperature=0.88, border_count=217),
        25: dict(loss_function="Quantile:alpha=0.7", iterations=1500, learning_rate=0.034, depth=5, l2_leaf_reg=0.311, random_strength=1.1e-08, bagging_temperature=0.43, border_count=122),
        28: dict(loss_function="RMSE", iterations=1500, learning_rate=0.048, depth=5, l2_leaf_reg=0.003, random_strength=8.7e-07, bagging_temperature=0.57, border_count=153),
    }
    for p in CATBOOST_PARAMS.values(): p.update(random_seed=SEED, thread_count=-1, verbose=False, allow_writing_files=False)

    def first_existing(paths, name):
        for p in map(Path, paths):
            if p.exists(): return p
        raise FileNotFoundError(name)

    TRAIN_PATH = first_existing(["data/kaggle_mstrans/labels_day_train.csv", "data/labels_day_train.csv"], "labels_day_train.csv")
    TEST_PATH = first_existing(["data/kaggle_mstrans/labels_day_test.csv", "data/labels_day_test.csv"], "labels_day_test.csv")
    TEMPLATE_PATH = first_existing(["data/kaggle_mstrans/test_submission.csv", "data/test_submission.csv"], "test_submission.csv")
    BASELINE_PATH = first_existing(["submission_arima_w30_route5_85k.csv", "data/submission_arima_w30_route5_85k.csv", "data/kaggle_mstrans/submission.csv"], "submission.csv")

    history = pd.concat([pd.read_csv(TRAIN_PATH, sep=";"), pd.read_csv(TEST_PATH, sep=";")], ignore_index=True)
    history["date"] = pd.to_datetime(history["date"])
    template = pd.read_csv(TEMPLATE_PATH, sep=";"); template["date"] = pd.to_datetime(template["date"])
    baseline = pd.read_csv(BASELINE_PATH, sep=";"); baseline["date"] = pd.to_datetime(baseline["date"])

    last_history_date = history["date"].max()
    future_start = last_history_date + pd.Timedelta(days=1)
    future_end = pd.Timestamp("2026-09-20")
    years_needed = list(range(history["date"].min().year, future_end.year + 1))
    
    # --- ЗАГРУЗКА ПРАЗДНИКОВ ИЗ consultant2026.json ---
    holidays_json_path = ROOT / "consultant2026.json"
    if not holidays_json_path.exists():
        holidays_json_path = ROOT / "data" / "consultant2026.json"
    
    if holidays_json_path.exists():
        try:
            with open(holidays_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Извлекаем список дат по ключу "holidays"
                holiday_strings = data.get("holidays", [])
                HOLIDAYS_SET = set(pd.to_datetime(holiday_strings))
            print(f"✅ Загружено {len(HOLIDAYS_SET)} праздничных дней из {holidays_json_path.name}")
        except Exception as e:
            print(f"⚠️ Ошибка чтения {holidays_json_path.name}: {e}. Используем базовый календарь.")
            HOLIDAYS_SET = set()
            for year in years_needed:
                HOLIDAYS_SET.update(pd.to_datetime([f"{year}-01-01", f"{year}-01-02", f"{year}-01-03", f"{year}-01-04", f"{year}-01-05", f"{year}-01-06", f"{year}-01-07", f"{year}-01-08", f"{year}-02-23", f"{year}-03-08", f"{year}-05-01", f"{year}-05-09", f"{year}-06-12", f"{year}-11-04", f"{year}-12-31"]))
    else:
        print("⚠️ Файл consultant2026.json не найден. Используем базовый календарь.")
        HOLIDAYS_SET = set()
        for year in years_needed:
            HOLIDAYS_SET.update(pd.to_datetime([f"{year}-01-01", f"{year}-01-02", f"{year}-01-03", f"{year}-01-04", f"{year}-01-05", f"{year}-01-06", f"{year}-01-07", f"{year}-01-08", f"{year}-02-23", f"{year}-03-08", f"{year}-05-01", f"{year}-05-09", f"{year}-06-12", f"{year}-11-04", f"{year}-12-31"]))

    def seasonal_features(frame):
        frame = frame.copy(); dt = frame["date"]
        frame["hour_val"] = frame["hour"]; frame["dayofweek"] = dt.dt.dayofweek; frame["month"] = dt.dt.month
        frame["dayofyear"] = dt.dt.dayofyear; frame["is_weekend"] = frame["dayofweek"].isin([5, 6]).astype(int)
        frame["is_holiday"] = dt.dt.date.isin([h.date() for h in HOLIDAYS_SET]).astype(int)
        frame["is_off_day"] = ((frame["is_weekend"] == 1) | (frame["is_holiday"] == 1)).astype(int)
        for name, period, value in [("hour", 24, frame["hour_val"]), ("dow", 7, frame["dayofweek"]), ("doy", 365.25, frame["dayofyear"]), ("month", 12, frame["month"])]:
            frame[f"{name}_sin"] = np.sin(2 * np.pi * value / period); frame[f"{name}_cos"] = np.cos(2 * np.pi * value / period)
        return frame

    def add_profiles(frame, train_mask):
        result = frame.copy(); source = result.loc[train_mask]
        for keys, tag, stats in [(["route", "hour", "dayofweek"], "rhd", ["mean", "median", "std"]), (["route", "hour", "is_off_day"], "rho", ["mean", "median"]), (["route", "hour"], "rh", ["mean", "median"])]:
            profile = source.groupby(keys)["boardings"].agg(stats).reset_index().rename(columns={s: f"profile_{tag}_{s}" for s in stats})
            result = result.merge(profile, on=keys, how="left", validate="many_to_one")
        return result

    cat_grid = pd.MultiIndex.from_product([SELECTED_ROUTES, pd.date_range(history["date"].min(), future_end), range(24)], names=KEYS).to_frame(index=False)
    cat_grid = cat_grid.merge(history[KEYS + ["boardings"]], on=KEYS, how="left", validate="one_to_one").fillna(0.0)
    cat_grid = seasonal_features(cat_grid)
    cat_train_mask = cat_grid["date"] <= last_history_date
    cat_grid = add_profiles(cat_grid, cat_train_mask)
    CAT_FEATURES = [c for c in cat_grid.columns if c not in {"route", "date", "boardings"}]

    cat_predictions = []
    for route in SELECTED_ROUTES:
        train_route = cat_grid.loc[cat_train_mask & cat_grid["route"].eq(route)]
        future_route = cat_grid.loc[(cat_grid["date"] > last_history_date) & cat_grid["route"].eq(route)]
        model = CatBoostRegressor(**CATBOOST_PARAMS[route])
        model.fit(train_route[CAT_FEATURES], train_route["boardings"])
        part = future_route[KEYS].copy(); part["catboost_prediction"] = np.maximum(0, model.predict(future_route[CAT_FEATURES]))
        cat_predictions.append(part)
    cat_predictions = pd.concat(cat_predictions, ignore_index=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device, dtype=dtype)
    
    history_dates = pd.date_range(history["date"].min(), last_history_date)
    future_dates = pd.date_range(future_start, future_end)
    daily_observed = history.groupby(["route", "date"], as_index=False)["boardings"].sum()
    daily = pd.MultiIndex.from_product([ACTIVE_ROUTES, history_dates], names=["route", "date"]).to_frame(index=False).merge(daily_observed, how="left").fillna(0.0)
    
    def calendar_frame(routes, dates):
        frame = pd.MultiIndex.from_product([routes, pd.DatetimeIndex(dates)], names=["item_id", "timestamp"]).to_frame(index=False)
        dt = frame["timestamp"]; dow, doy = dt.dt.dayofweek, dt.dt.dayofyear
        frame["dow_sin"] = np.sin(2 * np.pi * dow / 7); frame["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        frame["doy_sin"] = np.sin(2 * np.pi * doy / 365.25); frame["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        frame["is_off_day"] = dt.dt.date.isin([h.date() for h in HOLIDAYS_SET]).astype(float)
        return frame

    context_df = calendar_frame(ACTIVE_ROUTES, history_dates).merge(daily, left_on=["item_id", "timestamp"], right_on=["route", "date"], validate="one_to_one").drop(columns=["route", "date"]).rename(columns={"boardings": "target"})
    future_df = calendar_frame(ACTIVE_ROUTES, future_dates)
    chronos_daily = pipeline.predict_df(context_df, future_df=future_df, prediction_length=len(future_dates), quantile_levels=[0.7], batch_size=100, context_length=512, cross_learning=True, freq="D")
    chronos_daily = chronos_daily.rename(columns={"item_id": "route", "timestamp": "date"})
    chronos_daily[CHRONOS_QUANTILE] = chronos_daily[CHRONOS_QUANTILE].clip(lower=0)

    # --- ПОЛНАЯ ЗАМЕНА ФИНАЛЬНОГО БЛОКА (Гарантирует наличие 2025 и 2026 годов) ---

    # 1. Определяем полный диапазон дат для итогового файла: с 1 января 2025 по 20 сентября 2026
    start_date = pd.Timestamp("2025-01-01")
    end_date = pd.Timestamp("2026-09-20")
    full_dates = pd.date_range(start_date, end_date)

    # 2. Создаем полную сетку (route, date, hour) на весь этот период
    full_grid = pd.MultiIndex.from_product([ACTIVE_ROUTES, full_dates, range(24)], names=KEYS).to_frame(index=False)

    # 3. Подтягиваем базовый прогноз (baseline) для всех этих дат. 
    # Он заполнит 2025 год, а для 2026 года там будут NaN (что нормально, мы их заменим).
    baseline_subset = baseline[baseline["route"] != 5].rename(columns={"prediction": "baseline_val"})
    full_grid = full_grid.merge(baseline_subset, on=KEYS, how="left", validate="one_to_one")
    full_grid["baseline_val"] = full_grid["baseline_val"].fillna(0.0)

    # 4. Генерируем новые предикты (Chronos + CatBoost) для периода ПОСЛЕ last_history_date
    future_dates = pd.date_range(last_history_date + pd.Timedelta(days=1), end_date)
    
    # (Здесь используется уже рассчитанный выше chronos_daily и cat_predictions)
    # Собираем future_grid аналогично тому, как делали раньше, но явно для future_dates
    future_grid = pd.MultiIndex.from_product([ACTIVE_ROUTES, future_dates, range(24)], names=KEYS).to_frame(index=False)
    future_grid = future_grid.merge(baseline_subset, on=KEYS, how="left", validate="one_to_one")
    future_grid["baseline_val"] = future_grid["baseline_val"].fillna(0.0)

    # Применяем Chronos
    future_grid = future_grid.merge(
        chronos_daily[["route", "date", CHRONOS_QUANTILE]].rename(columns={CHRONOS_QUANTILE: "chronos_q"}),
        on=["route", "date"], how="left", validate="many_to_one"
    )
    daily_sums = future_grid.groupby(["route", "date"])["baseline_val"].transform("sum")
    future_grid["hour_share"] = np.divide(
        future_grid["baseline_val"], daily_sums,
        out=np.full(len(future_grid), 1/24),
        where=daily_sums.to_numpy() > 0
    )
    future_grid["chronos_hourly"] = future_grid["hour_share"] * future_grid["chronos_q"].fillna(0.0)
    future_grid["chronos_blend"] = (1 - CHRONOS_WEIGHT) * future_grid["baseline_val"] + CHRONOS_WEIGHT * future_grid["chronos_hourly"]

    # Применяем CatBoost
    future_grid = future_grid.merge(
        cat_predictions.rename(columns={"catboost_prediction": "cat_pred"}),
        on=KEYS, how="left", validate="one_to_one"
    )
    future_grid["cat_weight"] = future_grid["route"].map(CATBOOST_WEIGHTS).fillna(0.0)
    future_grid["new_pred"] = (1 - future_grid["cat_weight"]) * future_grid["chronos_blend"] + future_grid["cat_weight"] * future_grid["cat_pred"].fillna(0.0)

    # 5. Обновляем полную сетку: где есть новый предикт (конец 2025 + 2026), берем его. 
    # Иначе оставляем baseline (начало и середина 2025 года).
    full_grid = full_grid.merge(
        future_grid[["route", "date", "hour", "new_pred"]],
        on=KEYS, how="left", validate="one_to_one"
    )
    full_grid["final_prediction"] = full_grid["new_pred"].fillna(full_grid["baseline_val"])

    # 6. Форматируем и сохраняем итоговый файл
    final_output = full_grid[["route", "date", "hour", "final_prediction"]].copy()
    final_output = final_output.rename(columns={"final_prediction": "prediction"})
    
    # Безопасное преобразование в целые числа
    final_output["prediction"] = final_output["prediction"].fillna(0.0).clip(lower=0).round().astype(int)
    final_output["date"] = final_output["date"].dt.strftime("%Y-%m-%d")
    
    # Сортируем для удобства чтения
    final_output = final_output.sort_values(by=["route", "date", "hour"]).reset_index(drop=True)
    
    out_path = ROOT / "data" / "forecast_selective.csv"
    final_output.to_csv(out_path, sep=";", index=False)
    
    print(f"✅ [Selective] Сохранено: {out_path}")
    print(f"Всего строк: {len(final_output)}")
    print(f"Диапазон дат в файле: с {final_output['date'].min()} по {final_output['date'].max()}")
    print(f"Сумма прогноза: {final_output['prediction'].sum()}")

def run_lgbm():
    print("🚀 [LGBM] Запуск Residual LGBM (LSTM+TCN)...")
    script = ROOT / "notebooks" / "kaggle" / "residual_lgbm.py"
    if not script.exists():
        print("❌ [LGBM] Файл residual_lgbm.py не найден!")
        return
    
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    if result.stdout: print(result.stdout)
    if result.returncode != 0:
        print("❌ [LGBM] Ошибка выполнения:\n", result.stderr)
        return
        
    src = ROOT / "data" / "kaggle_mstrans" / "lstm_tcn_v5" / "residual_lgbm" / "submission.csv"
    dst = ROOT / "data" / "forecast_lgbm.csv"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f"✅ [LGBM] Сохранено: {dst}")
    else:
        print("❌ [LGBM] Итоговый submission.csv не найден")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["selective", "lgbm", "all"], default="all")
    args = parser.parse_args()

    if args.model in ("selective", "all"):
        try: run_selective()
        except Exception as e: print(f"❌ Ошибка Selective: {e}")
    if args.model in ("lgbm", "all"):
        try: run_lgbm()
        except Exception as e: print(f"❌ Ошибка LGBM: {e}")