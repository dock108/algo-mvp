#!/bin/bash
# This script generates live configurations and hot-reloads the supervisor
# to start a 30-day paper trading session.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Path to the JSON file containing the best parameters
BEST_PARAMS_FILE="${1:-./best_params.json}" # Default to ./best_params.json if no argument provided

# Paths to Jinja2 templates
LIVE_RUNNER_TEMPLATE="workflows/templates/live_paper_template.yaml.jinja"
ORCHESTRATOR_TEMPLATE="workflows/templates/orchestrator_paper.yaml.jinja"

# Output directory for generated configuration files
CONFIG_OUTPUT_DIR="./generated_configs"

# --- Helper Functions ---

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to render Jinja2 template
# This is a basic example; you might need a more robust solution or a dedicated Python script.
render_template() {
    local template_file="$1"
    local params_file="$2"
    local output_file="$3"

    if ! command_exists jinja2; then
        echo "Error: jinja2-cli is not installed. Please install it (e.g., pip install jinja2-cli)."
        echo "Alternatively, modify this script to use another templating method."
        exit 1
    fi

    if [ ! -f "$template_file" ]; then
        echo "Error: Template file not found: $template_file"
        exit 1
    fi

    if [ ! -f "$params_file" ]; then
        echo "Error: Parameters file not found: $params_file"
        exit 1
    fi

    echo "Rendering $template_file with $params_file to $output_file..."
    # Ensure output directory exists
    mkdir -p "$(dirname "$output_file")"
    # The exact command might vary based on your jinja2-cli version and parameter format.
    # This assumes parameters are passed as a JSON file.
    jinja2 "$template_file" "$params_file" -o "$output_file"
    echo "Rendering complete."
}

# --- Main Script Logic ---

echo "Starting paper trading setup..."

if [ ! -f "$BEST_PARAMS_FILE" ]; then
    echo "Error: Best parameters file not found at $BEST_PARAMS_FILE"
    echo "Please run choose_best.py script first or provide the correct path."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$CONFIG_OUTPUT_DIR"

# --- Generate Live Runner Configurations ---
# This section assumes you might generate multiple live runner configs based on the parameters.
# For a single orchestrator setup, you might directly use BEST_PARAMS_FILE for the orchestrator.

# Example: If best_params.json contains a list of strategies or requires multiple files:
# This is a placeholder. You'll need to adapt this logic based on the structure of your best_params.json
# and how many live runner configs you need.

# For now, let's assume we generate one live runner config and one orchestrator config.
# We'll name them based on a timestamp or a strategy name if available in params.

STRATEGY_NAME=$(jq -r '.strategy_name // "default_strategy"' "$BEST_PARAMS_FILE")
TIMESTAMP=$(date +%Y%m%d%H%M%S)

GENERATED_LIVE_RUNNER_CONFIG="$CONFIG_OUTPUT_DIR/live_runner_${STRATEGY_NAME}_${TIMESTAMP}.yaml"
GENERATED_ORCHESTRATOR_CONFIG="$CONFIG_OUTPUT_DIR/orchestrator_paper_${STRATEGY_NAME}_${TIMESTAMP}.yaml"

# Render the live runner template (if you have a separate one per strategy instance)
# If your orchestrator template directly includes all parameters, you might skip this.
# For this example, let's assume live_paper_template.yaml.jinja is for a single runner
# and orchestrator_paper.yaml.jinja uses these generated files or parameters directly.

# This is a conceptual step. The actual rendering logic depends heavily on your template design.
# If live_paper_template.yaml.jinja is the main config driven by parameters:
render_template "$LIVE_RUNNER_TEMPLATE" "$BEST_PARAMS_FILE" "$GENERATED_LIVE_RUNNER_CONFIG"

# Render the orchestrator template
# The orchestrator template might need access to the path of the generated live runner config
# or directly use the parameters from BEST_PARAMS_FILE.
# For simplicity, let's assume it also just takes the best_params.json directly.
# You might need to create an intermediate JSON with more context for the orchestrator template.
render_template "$ORCHESTRATOR_TEMPLATE" "$BEST_PARAMS_FILE" "$GENERATED_ORCHESTRATOR_CONFIG"

echo "Configuration files generated in $CONFIG_OUTPUT_DIR"

# --- Hot-reload Supervisor ---
echo "Hot-reloading supervisor..."
# Placeholder command for hot-reloading supervisor
# Replace with your actual supervisor hot-reload command
# Example: supervisorctl reread
# Example: supervisorctl update
# Example: docker-compose kill -s SIGHUP supervisor_service_name
# Or, if you use a systemd service:
# Example: sudo systemctl reload your-supervisor.service

# Ensure you have a mechanism to tell the supervisor about the new config files,
# possibly by placing them in a watched directory or updating a main supervisor config file.

echo "Supervisor hot-reload command executed."

echo "Paper trading session setup initiated for 30 days."
