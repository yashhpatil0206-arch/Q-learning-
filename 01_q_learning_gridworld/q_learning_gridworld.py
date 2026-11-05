"""
Q-Learning Gridworld
A small tabular reinforcement-learning example using only Python's standard library.
Run: python q_learning_gridworld.py
"""
import random

ACTIONS = ["up", "down", "left", "right"]
MOVE = {
    "up": (-1, 0), "down": (1, 0),
    "left": (0, -1), "right": (0, 1)
}
ROWS, COLS = 4, 4
START = (0, 0)
GOAL = (3, 3)
OBSTACLES = {(1, 1), (2, 1)}
ALPHA = 0.1       # learning rate
GAMMA = 0.95      # discount factor
EPSILON = 0.15    # exploration probability
EPISODES = 2500

def valid_state(state):
    r, c = state
    return 0 <= r < ROWS and 0 <= c < COLS and state not in OBSTACLES

def step(state, action):
    dr, dc = MOVE[action]
    candidate = (state[0] + dr, state[1] + dc)
    if not valid_state(candidate):
        candidate = state
    if candidate == GOAL:
        return candidate, 10.0, True
    return candidate, -1.0, False

def choose_action(q_table, state):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: q_table[(state, a)])

def train():
    q_table = {(s, a): 0.0 for r in range(ROWS) for c in range(COLS)
               for s in [(r, c)] if valid_state(s) for a in ACTIONS}

    for _ in range(EPISODES):
        state = START
        for _ in range(100):
            action = choose_action(q_table, state)
            next_state, reward, done = step(state, action)
            old_value = q_table[(state, action)]
            future = 0.0 if done else max(q_table[(next_state, a)] for a in ACTIONS)
            q_table[(state, action)] = old_value + ALPHA * (
                reward + GAMMA * future - old_value
            )
            state = next_state
            if done:
                break
    return q_table

def show_policy(q_table):
    symbols = {"up": "^", "down": "v", "left": "<", "right": ">"}
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            s = (r, c)
            if s == START:
                row.append("S")
            elif s == GOAL:
                row.append("G")
            elif s in OBSTACLES:
                row.append("#")
            else:
                best = max(ACTIONS, key=lambda a: q_table[(s, a)])
                row.append(symbols[best])
        print(" ".join(row))

if __name__ == "__main__":
    random.seed(7)
    learned_q = train()
    print("Learned policy: S=start, G=goal, #=blocked")
    show_policy(learned_q)
