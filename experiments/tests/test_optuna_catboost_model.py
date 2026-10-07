import ast
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).parents[1] / "scripts"
MODEL_FILES = (
    SCRIPT_DIR / "catboostweather.py",
    SCRIPT_DIR / "catboost_fixed.py",
    SCRIPT_DIR / "optuna_catboost_model.py",
    SCRIPT_DIR / "catboost_model.py",
)


class LossSearchTests(unittest.TestCase):
    def test_quantile_candidates_include_multiple_alphas(self):
        source = (SCRIPT_DIR / "optuna_catboost_model.py").read_text()
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "quantile_loss_candidates"
        )
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<test>", "exec"), namespace)
        self.assertEqual(
            namespace["quantile_loss_candidates"](),
            [f"Quantile:alpha={alpha / 10:.1f}" for alpha in range(1, 10)],
        )


class CalendarFeatureTests(unittest.TestCase):
    def test_calendar_features_include_short_days_and_remove_holiday(self):
        for filename in MODEL_FILES:
            source = Path(filename).read_text()
            tree = ast.parse(source)
            functions = {
                node.name: node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
            }
            holiday_source = ast.get_source_segment(
                source, functions["get_russian_holidays_2025"]
            )
            calendar_source = ast.get_source_segment(
                source, functions["get_russian_non_working_days_2025"]
            )
            feature_source = ast.get_source_segment(
                source, functions["extract_seasonal_features"]
            )

            self.assertIn("2025-03-07", calendar_source)  # short day
            self.assertIn("2025-05-03", calendar_source)  # weekend
            self.assertIn("2025-05-01", holiday_source)
            self.assertIn('["is_non_working_day"]', feature_source)
            self.assertIn('["is_off_day"]', feature_source)
            self.assertIn('["is_holiday"]', feature_source)
            self.assertIn('["is_preholiday"]', feature_source)
