#!/usr/bin/env python3
import argparse
import json
import pandas as pd


# Helper function to sort by multiple metrics.
# Add more metrics and sorting orders as needed.
def sort_by_metrics(df, metrics_to_sort_by):
    """Sorts a DataFrame by a list of metrics.
    Args:
        df (pd.DataFrame): The DataFrame to sort.
        metrics_to_sort_by (list): A list of tuples, where each tuple
                                   contains the metric name and a boolean
                                   indicating ascending (True) or
                                   descending (False) order.
    Returns:
        pd.DataFrame: The sorted DataFrame.
    """
    # Validate that all required metrics exist in the dataframe
    column_names = []
    ascending_values = []

    for metric, ascending in metrics_to_sort_by:
        if metric not in df.columns:
            raise ValueError(
                f"Metric {metric} not found in DataFrame columns: {df.columns}"
            )
        column_names.append(metric)
        ascending_values.append(ascending)

    # Sort by all metrics at once to maintain hierarchical sorting
    return df.sort_values(by=column_names, ascending=ascending_values)


def choose_best_parameters(metrics_file, output_file):
    """
    Reads backtest metrics, selects the best parameters, and writes them to a JSON file.

    Args:
        metrics_file (str): Path to the CSV file containing backtest metrics.
                            Assumes columns like 'parameter_set', 'sharpe_ratio', 'total_return', 'max_drawdown'.
        output_file (str): Path to the JSON file where the best parameters will be saved.
    """
    try:
        # Load the metrics data
        # Adjust this part if your metrics are in a different format (e.g., JSON)
        if metrics_file.endswith(".csv"):
            df = pd.read_csv(metrics_file)
        elif metrics_file.endswith(".json"):
            # Assuming a list of records format for JSON
            df = pd.read_json(metrics_file, orient="records")
        else:
            print(
                f"Unsupported file format for metrics_file: {metrics_file}. Please use .csv or .json."
            )
            return

        if df.empty:
            print(f"No data found in {metrics_file}.")
            return

        # Define metrics and their sorting order for selecting the best parameters
        # Example: Sort by Sharpe Ratio (descending), then by Total Return (descending)
        # Add or modify these metrics based on your strategy's performance indicators
        metrics_to_sort_by = [
            ("sharpe_ratio", False),  # Higher Sharpe Ratio is better
            ("total_return", False),  # Higher Total Return is better
            ("max_drawdown", True),  # Lower Max Drawdown is better
        ]

        # Validate that all necessary metrics are present
        for metric, _ in metrics_to_sort_by:
            if metric not in df.columns:
                print(
                    f"Warning: Metric '{metric}' not found in {metrics_file}. It will be ignored for sorting."
                )

        # Filter out metrics not present in the DataFrame to avoid errors
        valid_metrics_to_sort_by = [
            (m, asc) for m, asc in metrics_to_sort_by if m in df.columns
        ]

        if not valid_metrics_to_sort_by:
            print("No valid metrics found for sorting. Cannot choose best parameters.")
            if not df.empty:
                # Fallback: if no valid metrics but 'parameter_set' exists, pick the first one.
                if "parameter_set" in df.columns:
                    best_params_series = df.iloc[0]
                    print(
                        f"Warning: Selecting the first parameter set as a fallback: {best_params_series['parameter_set']}"
                    )
                else:
                    print(
                        "Cannot determine best parameters due to missing metrics and 'parameter_set' column."
                    )
                    return
            else:
                return

        # Sort by the defined metrics
        sorted_df = sort_by_metrics(df.copy(), valid_metrics_to_sort_by)

        # The best parameters are in the first row after sorting
        best_params_series = sorted_df.iloc[0]

        # Convert the best parameters to a dictionary
        # Assuming 'parameter_set' column contains a JSON string or dict of the parameters
        # Or other columns directly represent parameters like 'param_window', 'param_threshold'
        best_params_dict = {}
        if "parameter_set" in best_params_series and isinstance(
            best_params_series["parameter_set"], (str, dict)
        ):
            try:
                # If parameter_set is a string, try to parse it as JSON
                if isinstance(best_params_series["parameter_set"], str):
                    best_params_dict = json.loads(best_params_series["parameter_set"])
                else:  # if it's already a dict
                    best_params_dict = best_params_series["parameter_set"]
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not parse 'parameter_set' as JSON: {best_params_series['parameter_set']}"
                )
                print("Assuming 'parameter_set' is a simple string identifier.")
                best_params_dict = {"identifier": best_params_series["parameter_set"]}
        else:
            # If no 'parameter_set' column, try to extract parameters from other columns
            # This part needs to be adapted based on how parameters are stored in your metrics file
            # For example, if you have columns like 'param_sma_window', 'param_threshold'
            param_columns = [
                col for col in best_params_series.index if col.startswith("param_")
            ]
            if param_columns:
                best_params_dict = best_params_series[param_columns].to_dict()
            else:
                # Fallback: use all columns if no 'parameter_set' or 'param_' columns found.
                print(
                    "Warning: 'parameter_set' column or columns prefixed with 'param_' not found."
                )
                print("Using all columns from the best performing row as parameters.")
                best_params_dict = best_params_series.to_dict()

        # Save the best parameters to the output file
        with open(output_file, "w") as f:
            json.dump(best_params_dict, f, indent=4)

        print(f"Best parameters selected and saved to {output_file}")
        print(f"Selected parameters: {json.dumps(best_params_dict, indent=4)}")

    except FileNotFoundError:
        print(f"Error: Metrics file not found at {metrics_file}")
    except pd.errors.EmptyDataError:
        print(f"Error: Metrics file is empty at {metrics_file}")
    except KeyError as e:
        print(f"Error: Missing expected column in metrics file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Selects the best parameters from backtest metrics."
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        required=True,
        help="Path to the CSV or JSON file containing backtest metrics.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the JSON file where the best parameters will be saved.",
    )
    args = parser.parse_args()
    choose_best_parameters(args.metrics_file, args.output_file)


if __name__ == "__main__":
    main()
