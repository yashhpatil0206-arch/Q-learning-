"""
Q-Learning for Simple Inventory Replenishment
Uses a small synthetic demand time series and learns a reorder decision.
Run: python q_learning_inventory.py
"""
import csv
import random
from collections import defaultdict

ACTIONS = [0, 1, 2]  # order 0, 2, or 4 units
ORDER_QTY = {0: 0, 1: 2, 2: 4}
ALPHA, GAMMA, EPSILON = 0.12, 0.90, 0.15
EPISODES = 3500
DAYS = 30
MAX_STOCK = 10

def make_demand_series(seed=11, days=DAYS):
    """Create reproducible weekly-pattern demand for demonstration."""
    rng = random.Random(seed)
    values = []
    for day in range(days):
        weekday_effect = [0, 1, 1, 0, 2, 3, 2][day % 7]
        values.append(max(0, 2 + weekday_effect + rng.choice([-1, 0, 0, 1])))
    return values

def state_key(stock, demand_level, day_type):
    # Discretised state: inventory, recent demand level, weekday/weekend flag.
    return (min(stock, MAX_STOCK), demand_level, day_type)

def demand_level(d):
    return 0 if d <= 2 else (1 if d <= 4 else 2)

def choose_action(q, state, rng):
    if rng.random() < EPSILON:
        return rng.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: q[(state, a)])

def train(demands):
    q = defaultdict(float)
    rng = random.Random(21)

    for _ in range(EPISODES):
        stock = 5
        previous_demand = demands[0]
        for day in range(DAYS):
            current_demand = demands[day]
            state = state_key(stock, demand_level(previous_demand), int(day % 7 in (5, 6)))
            action = choose_action(q, state, rng)
            ordered = ORDER_QTY[action]
            available = min(MAX_STOCK, stock + ordered)
            sold = min(available, current_demand)
            shortage = current_demand - sold
            ending_stock = available - sold

            # Illustrative costs: order, holding and unmet-demand penalty.
            reward = -(1.0 * ordered + 0.25 * ending_stock + 5.0 * shortage)
            next_day = (day + 1) % DAYS
            next_demand = demands[next_day]
            next_state = state_key(ending_stock, demand_level(current_demand),
                                   int(next_day % 7 in (5, 6)))

            old = q[(state, action)]
            future = max(q[(next_state, a)] for a in ACTIONS) if day < DAYS - 1 else 0.0
            q[(state, action)] = old + ALPHA * (reward + GAMMA * future - old)
            stock = ending_stock
            previous_demand = current_demand
    return q

def evaluate(q, demands):
    stock, total_reward = 5, 0.0
    rows = []
    for day, demand in enumerate(demands):
        state = state_key(stock, demand_level(demand), int(day % 7 in (5, 6)))
        action = max(ACTIONS, key=lambda a: q[(state, a)])
        order = ORDER_QTY[action]
        available = min(MAX_STOCK, stock + order)
        sold = min(available, demand)
        shortage = demand - sold
        ending = available - sold
        reward = -(order + 0.25 * ending + 5.0 * shortage)
        total_reward += reward
        rows.append((day + 1, demand, stock, order, sold, ending, shortage, round(reward, 2)))
        stock = ending
    return rows, total_reward

if __name__ == "__main__":
    demand = make_demand_series()
    q_table = train(demand)
    results, score = evaluate(q_table, demand)
    headers = ["day", "demand", "opening_stock", "order", "sold",
               "ending_stock", "shortage", "reward"]
    with open("inventory_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
    print("Day-by-day evaluation")
    for row in results:
        print(row)
    print(f"Total reward (higher is better): {score:.2f}")
    print("Saved results to inventory_results.csv")
