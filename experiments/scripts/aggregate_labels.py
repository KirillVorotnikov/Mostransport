import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROUTE_RE = re.compile(r"(\d+)")
INTERVALS = (10, 15, 20, 30)
SPLIT_RANGES = {
    "train": ("2025-01-01", "2025-08-31"),
    "test": ("2025-09-01", "2025-10-31"),
}


def aggregate_file(input_path: Path, output_path: Path, minutes: int) -> None:
    counts = defaultdict(int)
    with input_path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter=";"):
            if row.get("validation_result") != "1":
                continue
            route_match = ROUTE_RE.search(row.get("ngpt_route", ""))
            try:
                timestamp = datetime.strptime(row["tran_date_time"], "%Y-%m-%d %H:%M:%S")
            except (KeyError, TypeError, ValueError):
                continue
            if route_match is None:
                continue
            minute = timestamp.minute // minutes * minutes
            key = (int(route_match.group(1)), timestamp.date().isoformat(), timestamp.hour, minute)
            counts[key] += 1

    write_labels(counts, output_path)


def aggregate_splits(input_paths, output_dir: Path) -> None:
    counts_by_split = {
        split: {minutes: defaultdict(int) for minutes in INTERVALS}
        for split in SPLIT_RANGES
    }
    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source, delimiter=";"):
                if row.get("validation_result") != "1":
                    continue
                route_match = ROUTE_RE.search(row.get("ngpt_route", ""))
                try:
                    timestamp = datetime.strptime(row["tran_date_time"], "%Y-%m-%d %H:%M:%S")
                except (KeyError, TypeError, ValueError):
                    continue
                if route_match is None:
                    continue
                route = int(route_match.group(1))
                date = timestamp.date().isoformat()
                split = next(
                    (
                        name
                        for name, (start_date, end_date) in SPLIT_RANGES.items()
                        if start_date <= date <= end_date
                    ),
                    None,
                )
                if split is None:
                    continue
                for minutes, counts in counts_by_split[split].items():
                    minute = timestamp.minute // minutes * minutes
                    counts[(route, date, timestamp.hour, minute)] += 1

    for split, counts_by_interval in counts_by_split.items():
        for minutes, counts in counts_by_interval.items():
            write_labels(counts, output_dir / f"labels_{minutes}min_{split}.csv")


def write_labels(counts, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter=";")
        writer.writerow(["route", "date", "interval", "boardings"])
        for route, date, hour, minute in sorted(counts):
            writer.writerow([route, date, f"{hour:02d}:{minute:02d}", counts[(route, date, hour, minute)]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate raw tram validations into minute intervals.")
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("labels"))
    args = parser.parse_args()

    aggregate_splits(
        [args.input_dir / "train.csv", args.input_dir / "test.csv"],
        args.output_dir,
    )
    for split in ("train", "test"):
        for minutes in INTERVALS:
            print(args.output_dir / f"labels_{minutes}min_{split}.csv")


if __name__ == "__main__":
    main()
