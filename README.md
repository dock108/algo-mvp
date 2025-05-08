# algo-mvp

Lean backtest + paper-trade stack for Mike (Dock108).

## Quick Start

1.  **Install dependencies:**
    ```bash
    poetry install
    ```

2.  **Run tests:**
    ```bash
    poetry run pytest
    ```

3.  **Launch Streamlit app (placeholder):**
    ```bash
    poetry run streamlit hello
    ```

## Historical Data Fetching

This project includes a module to download historical market data for stocks/ETFs (via Alpaca) and micro-futures (via Tradovate demo).

### Configuration
Data fetching is configured using YAML files. Sample configurations can be found in the `configs/` directory:
- `configs/sample_equity.yaml`
- `configs/sample_futures.yaml`

Key configuration options include:
- `provider`: `alpaca` or `tradovate`
- `symbol`: The ticker symbol (e.g., `AAPL` for Alpaca, `MESM25` for Tradovate).
- `timeframe`: Data granularity (e.g., `1Min`, `1Day` for Alpaca; `1Min` for Tradovate which resamples ticks).
- `start`: Start date (YYYY-MM-DD).
- `end`: End date (YYYY-MM-DD).
- `adjust`: (Alpaca only) `true` or `false` for split/dividend adjustment.

### CLI Usage

To fetch data, use the following command:

```bash
python -m algo_mvp.fetch --config path/to/your_config.yaml
```

**Example:**
```bash
python -m algo_mvp.fetch --config configs/sample_equity.yaml
```

**Optional flags:**
- `--force`: Force download even if data for the requested range seems to exist locally. This overwrites any existing data for the specified timeframe and range.
- `--verbose`: Enable verbose output for more details on the fetching process.

Fetched data is stored in Parquet files under the `data/` directory, organized by provider, symbol, and timeframe (e.g., `data/alpaca/AAPL/1Day.parquet` or `data/alpaca/AAPL/1Day/2023.parquet` if partitioned by year due to size).

### Environment Variables

API credentials must be set as environment variables:

**Alpaca:**
- `ALPACA_KEY_ID`: Your Alpaca API Key ID.
- `ALPACA_SECRET_KEY`: Your Alpaca Secret Key.
(These are for live or paper trading accounts; data fetching uses the SIP feed available with free accounts.)

**Tradovate (Demo):**
- `TRADOVATE_CLIENT_ID`: Your Tradovate demo client ID (optional, might depend on specific demo setup).
- `TRADOVATE_ACCESS_TOKEN`: Your Tradovate demo access token.

**Note:** Never commit your API keys or secrets to the repository. Use a `.env` file (add it to `.gitignore`) or manage environment variables through your system for local development.

### Supported Providers & Granularities

| Provider  | Granularities (Examples)        | Notes                                         |
|-----------|---------------------------------|-----------------------------------------------|
| Alpaca    | 1Min, 5Min, 15Min, 1H, 1Day     | Uses SIP feed (free tier for IEX data)        |
| Tradovate | 1Min, 5Min, 1H (from ticks)     | Uses demo env; resamples tick data to OHLCV   |
