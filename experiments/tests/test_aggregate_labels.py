import csv
import importlib.util
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "aggregate_labels",
    Path(__file__).parents[1] / "scripts" / "aggregate_labels.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AggregateLabelsTest(unittest.TestCase):
    def test_groups_successful_boardings_by_interval(self):
        source = Path(self.__class__.__name__ + "_source.csv")
        output = Path(self.__class__.__name__ + "_labels.csv")
        try:
            with source.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["tran_date_time", "validation_result", "ngpt_route"],
                    delimiter=";",
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"tran_date_time": "2025-01-01 05:00:00", "validation_result": "1", "ngpt_route": "25 трамвай"},
                        {"tran_date_time": "2025-01-01 05:09:59", "validation_result": "1", "ngpt_route": "25 трамвай"},
                        {"tran_date_time": "2025-01-01 05:10:00", "validation_result": "2", "ngpt_route": "25 трамвай"},
                        {"tran_date_time": "2025-01-01 05:15:00", "validation_result": "1", "ngpt_route": "25 трамвай"},
                        {"tran_date_time": "2025-01-01 05:20:00", "validation_result": "1", "ngpt_route": "25 трамвай"},
                    ],
                )

            MODULE.aggregate_file(source, output, 10)

            with output.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file, delimiter=";"))
            self.assertEqual(
                rows,
                [
                    {"route": "25", "date": "2025-01-01", "interval": "05:00", "boardings": "2"},
                    {"route": "25", "date": "2025-01-01", "interval": "05:10", "boardings": "1"},
                    {"route": "25", "date": "2025-01-01", "interval": "05:20", "boardings": "1"},
                ],
            )
        finally:
            source.unlink(missing_ok=True)
            output.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
