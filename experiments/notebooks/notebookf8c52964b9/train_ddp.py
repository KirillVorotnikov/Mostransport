"""Год почасовых посадок из сырого CSV → следующая неделя.

Контекст — до 365 последних дней маршрута, каждый день это 24 числа посадок.
Цель — 168 часов сразу после этого окна. Прогноз стартует с копии предыдущей
недели, BERT добавляет поправку.
"""

import os
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoConfig, AutoModelForMaskedLM

OUT = Path("/kaggle/working")
ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
START = date(2025, 1, 1)
TRAIN_END = date(2025, 8, 31)
TEST_START = date(2025, 9, 1)
TEST_END = date(2025, 10, 31)
FORECAST_START = date(2025, 11, 1)
FORECAST_END = date(2025, 12, 31)
MAX_DAYS = 365
MIN_DAYS = 28
HORIZON = 7
BATCH = 4
EPOCHS_TRAIN = 5
EPOCHS_ALL = 1
ENC_LR = 1e-5
HEAD_LR = 1e-3
WARMUP = 40


def daterange(start, end):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def points_of(score):
    if score <= 0.48:
        return 0
    if score <= 0.60:
        return 1
    if score <= 0.70:
        return 2
    if score <= 0.80:
        return 3
    if score <= 0.88:
        return 4
    return 5


def pin_alibi(encoder):
    if "alibi" in encoder._buffers:
        return
    alibi = encoder.alibi.detach().clone()
    del encoder.alibi
    encoder.register_buffer("alibi", alibi)


class WeekBert(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained("mosaicml/mosaic-bert-base", trust_remote_code=True)
        mlm = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
        self.bert = mlm.bert
        hidden = config.hidden_size
        self.day_proj = nn.Sequential(
            nn.Linear(24, hidden),
            nn.LayerNorm(hidden),
        )
        self.head = nn.Linear(hidden, HORIZON * 24)
        self.mix = nn.Parameter(torch.tensor(-4.0))
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, days, mask):
        embeds = self.day_proj(torch.log1p(days))
        encoded = self.bert.encoder(embeds, mask, output_all_encoded_layers=False)
        hidden = encoded[-1]
        if hidden.shape[1] != days.shape[1]:
            raise RuntimeError(f"encoder вернул последовательность {tuple(hidden.shape)}")
        delta = self.head(hidden[:, -1].float())
        previous = days[:, -HORIZON:, :].reshape(days.shape[0], -1).float()
        correction = torch.sigmoid(self.mix) * delta
        return (previous + correction).clamp(min=0), previous, correction


class WeekSet(Dataset):
    def __init__(self, days, series, targets):
        self.days = days
        self.series = series
        self.targets = targets
        self.index = []
        for route_i, _route in enumerate(ROUTES):
            for target in targets:
                self.index.append((route_i, target))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        route_i, target = self.index[item]
        ctx = self.series[route_i][max(0, target - MAX_DAYS):target]
        padded = np.zeros((MAX_DAYS, 24), np.float32)
        mask = np.zeros(MAX_DAYS, np.float32)
        padded[-len(ctx):] = ctx
        mask[-len(ctx):] = 1.0
        future = self.series[route_i][target:target + HORIZON].reshape(-1)
        return (
            torch.from_numpy(padded),
            torch.from_numpy(mask),
            torch.from_numpy(future.copy()),
            len(ctx),
        )


def load_series():
    with open(OUT / "history.pkl", "rb") as handle:
        history = pickle.load(handle)
    with open(OUT / "test_actual.pkl", "rb") as handle:
        history.update(pickle.load(handle))
    days = list(daterange(START, TEST_END))
    position = {day: index for index, day in enumerate(days)}
    series = np.zeros((len(ROUTES), len(days), 24), np.float32)
    route_pos = {route: index for index, route in enumerate(ROUTES)}
    for (route, day, hour), value in history.items():
        row = route_pos.get(route)
        col = position.get(day)
        if row is None or col is None:
            continue
        series[row, col, hour] = value
    return days, series


def target_starts(days, last_day):
    starts = []
    for index, day in enumerate(days):
        if index < MIN_DAYS or index + HORIZON > len(days):
            continue
        if days[index + HORIZON - 1] <= last_day:
            starts.append(index)
    return starts


def make_loader(dataset, rank, world):
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=False)
    loader = DataLoader(dataset, batch_size=BATCH, sampler=sampler, num_workers=0, pin_memory=True)
    return loader, sampler


def fit(model, loader, sampler, optimizer, scaler, epochs, rank):
    step = 0
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_abs = base_abs = total_y = 0.0
        for days, mask, target, _ctx in loader:
            scale_lr = min(1.0, (step + 1) / WARMUP)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * scale_lr
            days = days.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred, previous, _correction = model(days, mask)
            loss = F.l1_loss(pred.float(), target.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_abs += (pred.detach() - target.float()).abs().sum().item()
            base_abs += (previous.detach() - target.float()).abs().sum().item()
            total_y += target.float().sum().item()
            step += 1
            if rank == 0 and (step == 1 or step % 50 == 0):
                mix = torch.sigmoid(model.module.mix).item()
                print(
                    f"step {step} epoch {epoch} loss {loss.item():.1f} mix {mix:.3f} "
                    f"gpu_mb {round(torch.cuda.memory_allocated() / 2**20)}",
                    flush=True,
                )
        if rank == 0:
            print(
                f"epoch {epoch} train WAPE {total_abs / max(total_y, 1):.4f} "
                f"copy-week WAPE {base_abs / max(total_y, 1):.4f}",
                flush=True,
            )


@torch.no_grad()
def score_next_weeks(model, days, series, start, end):
    model.eval()
    model_abs = base_abs = denom = 0.0
    cursor = start
    while cursor <= end:
        horizon_days = min(HORIZON, (end - cursor).days + 1)
        target = days.index(cursor)
        batch_days = []
        batch_mask = []
        actual = []
        previous = []
        for route_i in range(len(ROUTES)):
            ctx = series[route_i][max(0, target - MAX_DAYS):target]
            padded = np.zeros((MAX_DAYS, 24), np.float32)
            mask = np.zeros(MAX_DAYS, np.float32)
            padded[-len(ctx):] = ctx
            mask[-len(ctx):] = 1.0
            batch_days.append(padded)
            batch_mask.append(mask)
            actual.append(series[route_i][target:target + horizon_days].reshape(-1))
            previous.append(series[route_i][target - HORIZON:target].reshape(-1)[: horizon_days * 24])
        pred, _prev, _correction = model(
            torch.tensor(np.stack(batch_days)).cuda(),
            torch.tensor(np.stack(batch_mask)).cuda(),
        )
        pred = pred[:, : horizon_days * 24].float().cpu().numpy()
        y = np.stack(actual)
        base = np.stack(previous)
        model_abs += np.abs(pred - y).sum()
        base_abs += np.abs(base - y).sum()
        denom += y.sum()
        print(
            f"week {cursor.isoformat()} days {horizon_days} "
            f"wape {np.abs(pred - y).sum() / max(y.sum(), 1):.4f} "
            f"copy {np.abs(base - y).sum() / max(y.sum(), 1):.4f}",
            flush=True,
        )
        cursor += timedelta(days=7)
    return model_abs, base_abs, denom


@torch.no_grad()
def forecast_submission(model, days, series):
    model.eval()
    future_days = list(daterange(FORECAST_START, FORECAST_END))
    known = series.shape[1]
    extended = np.zeros((len(ROUTES), known + len(future_days), 24), np.float32)
    extended[:, :known] = series
    calendar = days + future_days
    predictions = {}
    cursor = FORECAST_START
    while cursor <= FORECAST_END:
        horizon_days = min(HORIZON, (FORECAST_END - cursor).days + 1)
        target = calendar.index(cursor)
        batch_days = []
        batch_mask = []
        for route_i in range(len(ROUTES)):
            ctx = extended[route_i][max(0, target - MAX_DAYS):target]
            padded = np.zeros((MAX_DAYS, 24), np.float32)
            mask = np.zeros(MAX_DAYS, np.float32)
            padded[-len(ctx):] = ctx
            mask[-len(ctx):] = 1.0
            batch_days.append(padded)
            batch_mask.append(mask)
        pred, _prev, _correction = model(
            torch.tensor(np.stack(batch_days)).cuda(),
            torch.tensor(np.stack(batch_mask)).cuda(),
        )
        pred = pred.float().cpu().numpy()
        for route_i, route in enumerate(ROUTES):
            flat = pred[route_i]
            for offset in range(horizon_days):
                day = cursor + timedelta(days=offset)
                hours = flat[offset * 24:(offset + 1) * 24]
                extended[route_i, target + offset] = hours
                for hour, value in enumerate(hours):
                    predictions[(route, day.isoformat(), hour)] = max(0.0, float(value))
        cursor += timedelta(days=7)
    return predictions


def write_submission(predictions):
    import pandas as pd

    submission = pd.read_csv(next(Path("/kaggle/input").rglob("test_submission.csv")), sep=";", encoding="utf-8-sig")
    filled = []
    missing = 0
    for row in submission.itertuples(index=False):
        key = (int(row.route), str(row.date)[:10], int(row.hour))
        if key not in predictions:
            missing += 1
            filled.append(0)
        else:
            filled.append(int(round(predictions[key])))
    submission["prediction"] = filled
    submission.to_csv(OUT / "submission.csv", sep=";", index=False)
    print(
        "submission",
        len(submission),
        "missing",
        missing,
        "mean",
        round(sum(filled) / len(filled), 1),
        "zeros",
        sum(value == 0 for value in filled),
        flush=True,
    )


def build_optimizer(model):
    encoder, head = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "day_proj" in name or name.endswith("head.weight") or name.endswith("head.bias") or name.endswith("mix"):
            head.append(param)
        else:
            encoder.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder, "lr": ENC_LR, "weight_decay": 0.01},
            {"params": head, "lr": HEAD_LR, "weight_decay": 0.0},
        ]
    )
    optimizer.param_groups[0]["base_lr"] = ENC_LR
    optimizer.param_groups[1]["base_lr"] = HEAD_LR
    return optimizer


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.manual_seed(42)
    days, series = load_series()
    train_targets = target_starts(days, TRAIN_END)
    all_targets = target_starts(days, TEST_END)
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=3))
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    if rank == 0:
        sep_i = days.index(TEST_START)
        print(
            f"days {days[0].isoformat()}..{days[-1].isoformat()} "
            f"train weeks {len(train_targets)} x {len(ROUTES)} routes "
            f"context on Sep 1 = {sep_i} days (cap {MAX_DAYS})",
            flush=True,
        )
        print(f"ddp world {world} device {torch.cuda.get_device_name(rank)}", flush=True)
    model = WeekBert()
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    pin_alibi(model.bert.encoder)
    model = model.cuda(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    optimizer = build_optimizer(model)
    try:
        scaler = torch.amp.GradScaler("cuda")
    except TypeError:
        scaler = torch.cuda.amp.GradScaler()

    if rank == 0:
        print("phase 1: next week from the raw hourly year, targets through Aug 31", flush=True)
    loader, sampler = make_loader(WeekSet(days, series, train_targets), rank, world)
    fit(model, loader, sampler, optimizer, scaler, EPOCHS_TRAIN, rank)
    del loader, sampler
    dist.barrier()

    if rank == 0:
        print("score Sep-Oct as successive next weeks, true history only, before autumn fit", flush=True)
        model_abs, base_abs, denom = score_next_weeks(model.module, days, series, TEST_START, TEST_END)
        wape = model_abs / max(denom, 1.0)
        score = max(0.0, 1.0 - wape)
        base_wape = base_abs / max(denom, 1.0)
        awarded = points_of(score)
        text = (
            f"next_week_wape={wape:.6f}\n"
            f"next_week_score={score:.6f}\n"
            f"next_week_points={awarded}\n"
            f"next_week_weighted={awarded * 2}\n"
            f"copy_week_wape={base_wape:.6f}\n"
            f"copy_week_score={max(0.0, 1.0 - base_wape):.6f}\n"
        )
        (OUT / "metrics.txt").write_text(text, encoding="utf-8")
        print(text, flush=True)
    dist.barrier()

    if rank == 0:
        print("phase 2: one epoch including Sep-Oct next-week targets", flush=True)
    loader, sampler = make_loader(WeekSet(days, series, all_targets), rank, world)
    fit(model, loader, sampler, optimizer, scaler, EPOCHS_ALL, rank)
    dist.barrier()
    if rank == 0:
        torch.save(model.module.state_dict(), OUT / "boarding_bert.pt")
        write_submission(forecast_submission(model.module, days, series))
        print((OUT / "metrics.txt").read_text(encoding="utf-8"), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
