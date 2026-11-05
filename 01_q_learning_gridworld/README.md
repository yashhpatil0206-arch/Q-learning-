# Project 1: Q-Learning Gridworld

## Objective
Teach an agent to navigate a small grid from a start cell to a goal while avoiding blocked cells. The agent learns through trial and error using tabular Q-Learning.

## Requirements
- Python 3.8 or newer
- No external packages required

## Run
```bash
python q_learning_gridworld.py
```

## Concepts demonstrated
- State, action and reward
- Exploration versus exploitation (epsilon-greedy)
- Q-table
- Bellman update:
  Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]

## Expected output
A 4x4 policy grid. `S` is the start, `G` is the goal, `#` are blocked cells, and arrows show the learned action.

## Files
- `q_learning_gridworld.py`: complete training and policy display code.
