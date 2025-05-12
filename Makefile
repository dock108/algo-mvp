build:
	docker compose build

up:
	docker compose up -d

logs:
	docker compose logs -f backend

down:
	docker compose down

# --- Backtest-to-Paper Workflow ---
.PHONY: paper
paper: clean_paper_artifacts fetch_and_backtest choose_parameters start_paper_session

.PHONY: fetch_and_backtest
fetch_and_backtest:
	@echo "Running backtest_last45.sh to fetch data and run backtests..."
	./scripts/backtest_last45.sh # Output: Assumes this script outputs metrics, e.g., to backtest_metrics.csv
	@echo "Data fetching and backtesting finished."

.PHONY: choose_parameters
choose_parameters:
	@echo "Running choose_best.py to select best parameters..."
	# Ensure that the output of backtest_last45.sh is available as input here.
	# Adjust --metrics-file if your backtest script saves results elsewhere/differently.
	# Adjust --output-file to where start_paper.sh expects it.
	./scripts/choose_best.py --metrics-file ./backtest_results/metrics.csv --output-file ./best_params.json
	@echo "Best parameters selected and saved to ./best_params.json."

.PHONY: start_paper_session
start_paper_session:
	@echo "Running start_paper.sh to generate configs and start paper trading..."
	# The start_paper.sh script takes the best_params.json path as an argument.
	./scripts/start_paper.sh ./best_params.json
	@echo "Paper trading session initiated."

# Optional: A target to clean up generated artifacts from the paper workflow
.PHONY: clean_paper_artifacts
clean_paper_artifacts:
	@echo "Cleaning up paper trading workflow artifacts..."
	rm -f ./best_params.json
	rm -rf ./generated_configs # Matches default CONFIG_OUTPUT_DIR in start_paper.sh
	# Add commands to clean up backtest results if they are consistently named/located
	# For example: rm -rf ./backtest_results/
	@echo "Paper trading artifacts cleaned."

# Ensure scripts are executable
# This could also be part of your setup or CI
.PHONY: ensure_scripts_executable
ensure_scripts_executable:
	chmod +x ./scripts/backtest_last45.sh
	chmod +x ./scripts/choose_best.py
	chmod +x ./scripts/start_paper.sh

# You might want to add ensure_scripts_executable as a dependency to the `paper` target
# For example: paper: ensure_scripts_executable clean_paper_artifacts ... (rest of dependencies)
# Or run it manually once: make ensure_scripts_executable

# Prepend this to an existing Makefile or create a new one.
# If prepending, make sure there are no conflicts with existing targets.
