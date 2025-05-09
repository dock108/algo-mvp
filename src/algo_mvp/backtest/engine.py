"""Core backtesting engine powered by vectorbt."""

import importlib
import shutil
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd
import vectorbt as vbt
import yaml
from pydantic import BaseModel, Field, validator
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from tqdm import tqdm


# Base strategy class (abstract)
class Strategy:
    """Base class for all strategies."""

    def __init__(self, params: Dict[str, Any]):
        """Initialize with strategy parameters."""
        self.params = params

    def run(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Run strategy on data and return entry/exit signals."""
        raise NotImplementedError("Strategy subclass must implement run()")


# Pydantic models for config validation
class BacktestParams(BaseModel):
    """Parameters for a backtest configuration."""

    band_mult: Union[float, List[float]] = Field(
        ..., description="Band multiplier values"
    )
    atr_len: Union[int, List[int]] = Field(..., description="ATR length values")

    @validator("band_mult")
    def validate_band_mult(cls, v):
        if isinstance(v, list):
            for val in v:
                if not isinstance(val, (int, float)):
                    raise ValueError(
                        f"band_mult values must be numeric, got {type(val)}"
                    )
        elif not isinstance(v, (int, float)):
            raise ValueError(
                f"band_mult must be numeric or list of numerics, got {type(v)}"
            )
        return v

    @validator("atr_len")
    def validate_atr_len(cls, v):
        if isinstance(v, list):
            for val in v:
                if not isinstance(val, int):
                    raise ValueError(
                        f"atr_len values must be integers, got {type(val)}"
                    )
        elif not isinstance(v, int):
            raise ValueError(
                f"atr_len must be integer or list of integers, got {type(v)}"
            )
        return v


class BacktestConfig(BaseModel):
    """Configuration for a backtest run."""

    provider: str = Field(..., description="Data provider (alpaca or tradovate)")
    symbol: str = Field(..., description="Symbol to backtest")
    timeframe: str = Field(..., description="Timeframe for the data")
    strategy: str = Field(..., description="Strategy name")
    params: BacktestParams = Field(..., description="Strategy parameters")
    cash: float = Field(10000.0, description="Starting cash amount")
    commission: float = Field(0.0, description="Commission per trade")
    metrics: List[str] = Field(
        ["sharpe", "max_drawdown", "cagr", "win_rate", "expectancy"],
        description="Metrics to calculate",
    )

    @validator("provider")
    def validate_provider(cls, v):
        if v.lower() not in ["alpaca", "tradovate"]:
            raise ValueError(f"Provider must be 'alpaca' or 'tradovate', got '{v}'")
        return v.lower()

    @validator("metrics")
    def validate_metrics(cls, v):
        allowed_metrics = [
            "sharpe",
            "max_drawdown",
            "cagr",
            "win_rate",
            "expectancy",
            "exposure",
        ]
        for metric in v:
            if metric.lower() not in allowed_metrics:
                raise ValueError(
                    f"Metric '{metric}' not in allowed metrics: {allowed_metrics}"
                )
        return [m.lower() for m in v]


class BacktestEngine:
    """Vectorbt-powered backtesting engine."""

    def __init__(
        self, config_file: str, output_dir: Optional[str] = None, verbose: bool = False
    ):
        """Initialize the backtest engine with a configuration file.

        Args:
            config_file: Path to the YAML configuration file
            output_dir: Custom output directory (overrides default)
            verbose: Enable verbose logging
        """
        self.config_file = Path(config_file)
        self.verbose = verbose
        self.console = Console()

        # Load and validate config
        self.config = self._load_config()

        # Set output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = (
                Path("backtests")
                / self.config.strategy
                / self.config.symbol
                / timestamp
            )

        # Initialize strategy
        self.strategy_instance = self._load_strategy()

        # Initialize results containers
        self.results = []
        self.portfolios = []

    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            self.console.print(message)

    def _load_config(self) -> BacktestConfig:
        """Load and validate the YAML configuration."""
        try:
            with open(self.config_file, "r") as f:
                config_dict = yaml.safe_load(f)

            return BacktestConfig(**config_dict)
        except Exception as e:
            self.console.print(f"[red]Error loading configuration:[/red] {str(e)}")
            raise

    def _load_strategy(self) -> Type[Strategy]:
        """Dynamically load the strategy class based on config."""
        try:
            # Import the strategy module
            strategy_name = self.config.strategy.lower()
            module_path = f"algo_mvp.backtest.strategies.{strategy_name}"

            # Convert snake_case to CamelCase for class name
            class_name = (
                "".join(word.title() for word in strategy_name.split("_")) + "Strategy"
            )

            try:
                module = importlib.import_module(module_path)
                strategy_class = getattr(module, class_name)
                return strategy_class
            except (ImportError, AttributeError) as e:
                self.console.print(
                    f"[red]Strategy '{strategy_name}' not found: {str(e)}[/red]"
                )
                raise ValueError(
                    f"Strategy '{strategy_name}' not found or not properly implemented"
                )
        except Exception as e:
            self.console.print(f"[red]Error loading strategy:[/red] {str(e)}")
            raise

    def _load_data(self) -> pd.DataFrame:
        """Load price data from Parquet files."""
        base_path = Path("data") / self.config.provider / self.config.symbol
        timeframe_path = base_path / f"{self.config.timeframe}.parquet"

        # Check if the file exists
        if not timeframe_path.exists():
            self.console.print(f"[red]Data file not found:[/red] {timeframe_path}")
            raise FileNotFoundError(f"Data file not found: {timeframe_path}")

        # Load the Parquet file
        self._log(f"Loading data from: {timeframe_path}")
        df = pd.read_parquet(timeframe_path)

        # Basic validation
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [
            col for col in required_columns if col.lower() not in df.columns.str.lower()
        ]

        if missing_columns:
            self.console.print(
                f"[red]Missing required columns:[/red] {', '.join(missing_columns)}"
            )
            raise ValueError(f"Data missing required columns: {missing_columns}")

        # Ensure column names are standardized
        column_mapping = {col: col.lower() for col in df.columns}
        df = df.rename(columns=column_mapping)

        # Ensure DataFrame is sorted by index
        df = df.sort_index()

        self._log(
            f"Loaded {len(df)} rows of data from {df.index.min()} to {df.index.max()}"
        )
        return df

    def _expand_parameters(self) -> List[Dict[str, Any]]:
        """Expand parameter lists into a grid for sweep."""
        params = self.config.params.dict()
        param_grid = []

        # Track which parameters are lists for grid sweep
        param_lists = {}
        for key, value in params.items():
            if isinstance(value, list):
                param_lists[key] = value
            else:
                # Convert single values to lists for consistent processing
                param_lists[key] = [value]

        # Generate all combinations
        keys = list(param_lists.keys())
        values = list(param_lists.values())

        for combination in product(*values):
            param_dict = dict(zip(keys, combination))
            param_grid.append(param_dict)

        self._log(f"Parameter grid expanded to {len(param_grid)} combinations")
        return param_grid

    def _calculate_metrics(self, portfolio) -> Dict[str, float]:
        """Calculate performance metrics based on configuration."""
        metrics = {}

        for metric_name in self.config.metrics:
            if metric_name == "sharpe":
                metrics["sharpe"] = portfolio.sharpe_ratio()
            elif metric_name == "max_drawdown":
                metrics["max_drawdown"] = (
                    portfolio.drawdown().max() * 100
                )  # as percentage
            elif metric_name == "cagr":
                metrics["cagr"] = portfolio.annual_return() * 100  # as percentage
            elif metric_name == "win_rate":
                if len(portfolio.trades) > 0:
                    metrics["win_rate"] = (
                        portfolio.trades.win_rate * 100
                    )  # as percentage
                else:
                    metrics["win_rate"] = 0.0
            elif metric_name == "expectancy":
                if len(portfolio.trades) > 0:
                    metrics["expectancy"] = portfolio.trades.expectancy
                else:
                    metrics["expectancy"] = 0.0
            elif metric_name == "exposure":
                metrics["exposure"] = (
                    portfolio.positions.avg_coverage * 100
                )  # as percentage

        return metrics

    def _save_results(
        self, run_id: int, params: Dict[str, Any], portfolio, metrics: Dict[str, float]
    ):
        """Save backtest results to files."""
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save equity curve
        equity_file = self.output_dir / f"equity_{run_id}.csv"
        portfolio.equity.to_csv(equity_file)

        # Save interactive plot
        plot_file = self.output_dir / f"plot_{run_id}.html"
        fig = portfolio.plot()
        fig.write_html(str(plot_file))

        # Add metrics and parameters to results list
        result = {"run_id": run_id, **params, **metrics}
        self.results.append(result)

        # Store portfolio for further analysis
        self.portfolios.append(portfolio)

    def _save_metrics_summary(self):
        """Save a summary CSV of all run metrics."""
        if not self.results:
            return

        # Convert results to DataFrame
        metrics_df = pd.DataFrame(self.results)

        # Save to CSV
        metrics_file = self.output_dir / "metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)

        return metrics_df

    def _save_config_copy(self):
        """Save a copy of the config file to the output directory."""
        config_dest = self.output_dir / self.config_file.name
        shutil.copy2(self.config_file, config_dest)

    def _print_summary_table(self, metrics_df: pd.DataFrame):
        """Print a summary table of the backtest results."""
        table = Table(
            title=f"Backtest Results Summary - {self.config.strategy} on {self.config.symbol}"
        )

        # Add columns
        table.add_column("Run", justify="right")

        # Add parameter columns
        param_columns = [
            col
            for col in metrics_df.columns
            if col not in ["run_id"] + self.config.metrics
        ]
        for param in param_columns:
            table.add_column(param, justify="right")

        # Add metric columns
        for metric in self.config.metrics:
            if metric in metrics_df.columns:
                table.add_column(metric, justify="right")

        # Add rows
        for _, row in metrics_df.iterrows():
            table_row = [str(int(row["run_id"]))]

            # Add parameter values
            for param in param_columns:
                table_row.append(str(row[param]))

            # Add metric values with formatting
            for metric in self.config.metrics:
                if metric in row:
                    # Format numbers nicely
                    if metric in ["sharpe", "expectancy"]:
                        table_row.append(f"{row[metric]:.2f}")
                    elif metric in ["max_drawdown", "cagr", "win_rate", "exposure"]:
                        table_row.append(f"{row[metric]:.2f}%")
                    else:
                        table_row.append(str(row[metric]))

            table.add_row(*table_row)

        self.console.print(table)

    def run(self) -> bool:
        """Run the backtest with the provided configuration.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load price data
            data = self._load_data()

            # Expand parameters for grid sweep
            param_grid = self._expand_parameters()
            total_runs = len(param_grid)

            self._log(f"Starting backtest with {total_runs} parameter combinations")

            # Run backtest for each parameter combination
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=self.console if self.verbose else None,
            ) as progress:
                task = progress.add_task(
                    f"Running {total_runs} backtests...", total=total_runs
                )

                for run_id, params in enumerate(tqdm(param_grid, disable=self.verbose)):
                    # Create strategy instance with parameters
                    strategy_obj = self.strategy_instance(params)

                    # Run strategy to get entry/exit signals
                    signals = strategy_obj.run(data)

                    if "entries" not in signals or "exits" not in signals:
                        self.console.print(
                            f"[yellow]Warning:[/yellow] Run {run_id} did not produce valid entry/exit signals"
                        )
                        continue

                    # Run portfolio simulation
                    portfolio = vbt.Portfolio.from_signals(
                        data["close"],
                        signals["entries"],
                        signals["exits"],
                        init_cash=self.config.cash,
                        fees=self.config.commission,
                        freq=data.index.inferred_freq,
                    )

                    # Calculate metrics
                    metrics = self._calculate_metrics(portfolio)

                    # Save results
                    self._save_results(run_id, params, portfolio, metrics)

                    progress.update(task, advance=1)

            # Save metrics summary
            metrics_df = self._save_metrics_summary()

            # Save config copy
            self._save_config_copy()

            # Print summary
            if metrics_df is not None:
                self._print_summary_table(metrics_df)

            self._log(
                f"Backtest completed successfully. Results saved to {self.output_dir}"
            )
            return True

        except Exception as e:
            self.console.print(f"[red]Error running backtest:[/red] {str(e)}")
            import traceback

            self.console.print(traceback.format_exc())
            return False
