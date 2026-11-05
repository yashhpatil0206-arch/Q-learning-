# Project 2: Q-Learning for Inventory Decisions (Time-Series Example)

## Objective
Use a short synthetic demand time series to demonstrate how a Q-Learning agent can learn a simple stock-replenishment policy. This is an educational simulation, not a real business forecasting or purchasing system.

## Requirements
- Python 3.8 or newer
- No external packages required

## Run
```bash
python q_learning_inventory.py
```
The script prints a day-by-day evaluation and creates `inventory_results.csv`.

## How it connects to time series
Demand is represented as an ordered sequence of daily observations with a repeating weekly pattern and random variation. The state includes a discretised recent-demand level and a weekday/weekend indicator. Q-Learning selects a replenishment action based on the current state.

## Reward design
The example gives a negative reward for ordering, holding excess stock, and unmet demand. The agent aims to maximise cumulative reward (equivalent to reducing these illustrative costs).

## Limitations
- The demand series is synthetic and small.
- The demonstration does not train a forecasting model.
- A real deployment would require historical data, validation, safety constraints, and careful cost estimation.

## Files
- `q_learning_inventory.py`: training and evaluation.
- `inventory_results.csv`: generated when the script is run.
