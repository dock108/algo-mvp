#!/usr/bin/env python3
import unittest
import tempfile
import json
import os
import pandas as pd
import sys

# Add the script's directory to the Python path to allow direct import
# This assumes the test is run from the root of the project or a similar context
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "../../scripts")
sys.path.insert(0, SCRIPTS_DIR)

from choose_best import choose_best_parameters  # noqa E402 module level import not at top of file


class TestChooseBest(unittest.TestCase):
    def create_mock_metrics_file(self, data, file_format="csv", dir_path=None):
        if dir_path is None:
            dir_path = tempfile.gettempdir()

        suffix = ".csv" if file_format == "csv" else ".json"
        # Use NamedTemporaryFile to ensure it's cleaned up, but get its name for the script
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=suffix, dir=dir_path, newline=""
        ) as tmp_file:
            if file_format == "csv":
                df = pd.DataFrame(data)
                df.to_csv(tmp_file.name, index=False)
            elif file_format == "json":
                json.dump(data, tmp_file, indent=4)
            return tmp_file.name

    def test_choose_best_csv_basic_selection(self):
        """Test basic selection from a CSV file."""
        metrics_data = [
            {
                "parameter_set": '{"sma_window": 10, "rsi_period": 7}',
                "sharpe_ratio": 1.5,
                "total_return": 0.2,
                "max_drawdown": 0.05,
            },
            {
                "parameter_set": '{"sma_window": 20, "rsi_period": 14}',
                "sharpe_ratio": 2.0,
                "total_return": 0.25,
                "max_drawdown": 0.04,
            },  # Best
            {
                "parameter_set": '{"sma_window": 5, "rsi_period": 5}',
                "sharpe_ratio": 1.0,
                "total_return": 0.15,
                "max_drawdown": 0.06,
            },
        ]
        metrics_file = self.create_mock_metrics_file(metrics_data, file_format="csv")

        with tempfile.NamedTemporaryFile(mode="r", suffix=".json") as tmp_output_file:
            output_file_path = tmp_output_file.name

        try:
            choose_best_parameters(metrics_file, output_file_path)
            with open(output_file_path, "r") as f:
                result = json.load(f)

            expected_params = {"sma_window": 20, "rsi_period": 14}
            self.assertEqual(result, expected_params)
        finally:
            os.remove(metrics_file)
            if os.path.exists(output_file_path):
                os.remove(output_file_path)

    def test_choose_best_json_input(self):
        """Test selection from a JSON input file."""
        metrics_data = [
            {
                "parameter_set": {"sma_window": 10, "rsi_period": 7},
                "sharpe_ratio": 1.5,
                "total_return": 0.2,
                "max_drawdown": 0.05,
            },
            {
                "parameter_set": {"sma_window": 20, "rsi_period": 14},
                "sharpe_ratio": 2.0,
                "total_return": 0.25,
                "max_drawdown": 0.04,
            },  # Best
            {
                "parameter_set": {"sma_window": 5, "rsi_period": 5},
                "sharpe_ratio": 1.0,
                "total_return": 0.15,
                "max_drawdown": 0.06,
            },
        ]
        # Note: For JSON, parameter_set is already an object
        metrics_file = self.create_mock_metrics_file(metrics_data, file_format="json")

        with tempfile.NamedTemporaryFile(mode="r", suffix=".json") as tmp_output_file:
            output_file_path = tmp_output_file.name

        try:
            choose_best_parameters(metrics_file, output_file_path)
            with open(output_file_path, "r") as f:
                result = json.load(f)

            expected_params = {"sma_window": 20, "rsi_period": 14}
            self.assertEqual(result, expected_params)
        finally:
            os.remove(metrics_file)
            if os.path.exists(output_file_path):
                os.remove(output_file_path)

    def test_tie_breaking_order(self):
        """Test that tie-breaking follows the specified metric order."""
        metrics_data = [
            {
                "parameter_set": '{"param": "A"}',
                "sharpe_ratio": 2.0,
                "total_return": 0.20,
                "max_drawdown": 0.05,
            },
            {
                "parameter_set": '{"param": "B"}',
                "sharpe_ratio": 2.0,
                "total_return": 0.25,
                "max_drawdown": 0.04,
            },  # Best due to total_return
            {
                "parameter_set": '{"param": "C"}',
                "sharpe_ratio": 2.0,
                "total_return": 0.20,
                "max_drawdown": 0.03,
            },  # Then this one by max_drawdown
        ]
        metrics_file = self.create_mock_metrics_file(metrics_data, file_format="csv")
        with tempfile.NamedTemporaryFile(mode="r", suffix=".json") as tmp_output_file:
            output_file_path = tmp_output_file.name

        try:
            choose_best_parameters(metrics_file, output_file_path)
            with open(output_file_path, "r") as f:
                result = json.load(f)
            self.assertEqual(result, {"param": "B"})
        finally:
            os.remove(metrics_file)
            if os.path.exists(output_file_path):
                os.remove(output_file_path)

    def test_missing_metric_column(self):
        """Test behavior when a primary sort metric column is missing (should still run with others)."""
        # Missing 'sharpe_ratio', will sort by 'total_return' then 'max_drawdown'
        metrics_data = [
            {
                "parameter_set": '{"param": "X"}',
                "total_return": 0.20,
                "max_drawdown": 0.05,
            },
            {
                "parameter_set": '{"param": "Y"}',
                "total_return": 0.25,
                "max_drawdown": 0.04,
            },  # Best
            {
                "parameter_set": '{"param": "Z"}',
                "total_return": 0.15,
                "max_drawdown": 0.06,
            },
        ]
        metrics_file = self.create_mock_metrics_file(metrics_data, file_format="csv")
        with tempfile.NamedTemporaryFile(mode="r", suffix=".json") as tmp_output_file:
            output_file_path = tmp_output_file.name

        try:
            # Capture stdout to check for warnings (optional)
            choose_best_parameters(metrics_file, output_file_path)
            with open(output_file_path, "r") as f:
                result = json.load(f)
            self.assertEqual(result, {"param": "Y"})
        finally:
            os.remove(metrics_file)
            if os.path.exists(output_file_path):
                os.remove(output_file_path)

    def test_empty_metrics_file(self):
        """Test with an empty metrics file."""
        metrics_file = self.create_mock_metrics_file([], file_format="csv")
        with tempfile.NamedTemporaryFile(mode="r", suffix=".json") as tmp_output_file:
            output_file_path = tmp_output_file.name

        try:
            choose_best_parameters(metrics_file, output_file_path)
            # Expect output file not to be created or to be empty, script should print error
            # For this test, we check if the output file is not created or empty if it is.
            self.assertFalse(
                os.path.exists(output_file_path)
                and os.path.getsize(output_file_path) > 0
            )
        finally:
            os.remove(metrics_file)
            if os.path.exists(output_file_path):
                os.remove(output_file_path)

    def test_no_valid_metrics_for_sorting(self):
        """Test when no defined sortable metrics are in the file but parameter_set exists."""
        metrics_data = [
            {
                "parameter_set": '{"param": "Fallback1"}',
                "unknown_metric1": 10,
                "unknown_metric2": 20,
            },
            {
                "parameter_set": '{"param": "Fallback2"}',
                "unknown_metric1": 5,
                "unknown_metric2": 25,
            },
        ]
        metrics_file = self.create_mock_metrics_file(metrics_data, file_format="csv")
        with tempfile.NamedTemporaryFile(mode="r", suffix=".json") as tmp_output_file:
            output_file_path = tmp_output_file.name

        try:
            choose_best_parameters(metrics_file, output_file_path)
            with open(output_file_path, "r") as f:
                result = json.load(f)
            # Should pick the first row as a fallback
            self.assertEqual(result, {"param": "Fallback1"})
        finally:
            os.remove(metrics_file)
            if os.path.exists(output_file_path):
                os.remove(output_file_path)

    def test_parameter_extraction_direct_columns(self):
        """Test parameter extraction from columns prefixed with 'param_'."""
        metrics_data = [
            {
                "param_alpha": 0.1,
                "param_beta": 0.2,
                "sharpe_ratio": 1.5,
                "total_return": 0.2,
                "max_drawdown": 0.05,
            },
            {
                "param_alpha": 0.3,
                "param_beta": 0.4,
                "sharpe_ratio": 2.0,
                "total_return": 0.25,
                "max_drawdown": 0.04,
            },  # Best
        ]
        metrics_file = self.create_mock_metrics_file(metrics_data, file_format="csv")
        with tempfile.NamedTemporaryFile(mode="r", suffix=".json") as tmp_output_file:
            output_file_path = tmp_output_file.name

        try:
            choose_best_parameters(metrics_file, output_file_path)
            with open(output_file_path, "r") as f:
                result = json.load(f)
            expected_params = {"param_alpha": 0.3, "param_beta": 0.4}
            self.assertEqual(result, expected_params)
        finally:
            os.remove(metrics_file)
            if os.path.exists(output_file_path):
                os.remove(output_file_path)


if __name__ == "__main__":
    unittest.main()
