from pathlib import Path

import polars as pl


DATA_DIR = Path("/kaggle/input/datasets/kirillvorotnikov2007/mstrans")
OUT_DIR = Path("/kaggle/working")


def make_labels(input_path: Path, output_path: Path) -> None:
    data = (
        pl.scan_csv(
            input_path,
            separator=";",
            encoding="utf8",
            has_header=True,
            infer_schema_length=10_000,
            ignore_errors=True,
            schema_overrides={
                "tran_date_time": pl.String,
                "ngpt_route": pl.String,
                "garage_number": pl.String,
                "validation_result": pl.Int16,
            },
        )
        .with_columns(
            [
                pl.col("tran_date_time")
                .str.strptime(
                    pl.Datetime,
                    format="%Y-%m-%d %H:%M:%S",
                    strict=False,
                )
                .alias("timestamp"),
                pl.col("ngpt_route")
                .str.extract(r"(\d+)", 1)
                .cast(pl.Int16, strict=False)
                .alias("route"),
                pl.col("garage_number")
                .str.strip_chars()
                .replace("", None)
                .alias("garage_id"),
            ]
        )
        .with_columns(
            [
                pl.col("timestamp").dt.date().alias("date"),
                pl.col("timestamp").dt.hour().alias("hour"),
            ]
        )
        .filter(
            pl.col("route").is_not_null()
            & pl.col("date").is_not_null()
            & pl.col("hour").is_not_null()
        )
        .group_by(["route", "date", "hour"])
        .agg(
            [
                pl.col("validation_result")
                .eq(1)
                .cast(pl.UInt32)
                .sum()
                .alias("boardings"),
                # Любая запись валидации считается присутствием трамвая в часу.
                pl.col("garage_id").drop_nulls().n_unique().alias("active_trams"),
            ]
        )
        .filter(pl.col("boardings") > 0)
        .select(["route", "date", "hour", "boardings", "active_trams"])
        .sort(["route", "date", "hour"])
        .collect(engine="streaming")
    )

    data = data.with_columns(
        [
            pl.col("route").cast(pl.Int16),
            pl.col("hour").cast(pl.Int8),
            pl.col("boardings").cast(pl.UInt32),
            pl.col("active_trams").cast(pl.UInt16),
        ]
    )

    required_columns = {
        "route",
        "date",
        "hour",
        "boardings",
        "active_trams",
    }
    assert required_columns <= set(data.columns)
    assert data["active_trams"].ge(0).all()
    assert data["boardings"].min() >= 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.write_csv(output_path, separator=";")
    print(f"{output_path}: {data.height:,} rows")


if __name__ == "__main__":
    make_labels(
        DATA_DIR / "train.csv",
        OUT_DIR / "labels_day_train_active_trams.csv",
    )
    make_labels(
        DATA_DIR / "test.csv",
        OUT_DIR / "labels_day_test_active_trams.csv",
    )
