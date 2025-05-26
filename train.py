import numpy as np
import random

# Q-learning settings
alpha = 0.1       # learning rate
gamma = 0.9       # discount factor
epsilon = 0.1     # exploration rate
episodes = 50000

# Game definitions
def get_empty_cells(state):
    return [i for i, v in enumerate(state) if v == 0]

def state_to_index(state):
    index = 0
    for i in range(9):
        index *= 3
        index += state[i]
    return index

def check_winner(state):
    wins = [
        [0,1,2], [3,4,5], [6,7,8],
        [0,3,6], [1,4,7], [2,5,8],
        [0,4,8], [2,4,6]
    ]
    for line in wins:
        a, b, c = line
        if state[a] != 0 and state[a] == state[b] and state[a] == state[c]:
            return state[a]
    return 0 if 0 in state else -1  # 0: ongoing, -1: draw

q_table = np.zeros((3**9, 9))

for episode in range(episodes):
    state = [0]*9
    done = False
    while not done:
        s_index = state_to_index(state)

        if random.random() < epsilon:
            action = random.choice(get_empty_cells(state))
        else:
            q_values = q_table[s_index]
            valid_actions = get_empty_cells(state)
            action = max(valid_actions, key=lambda a: q_values[a])

        state[action] = 1  # AI move

        winner = check_winner(state)
        if winner != 0:
            reward = 1 if winner == 1 else 0
            q_table[s_index][action] += alpha * (reward - q_table[s_index][action])
            done = True
            continue

        # Simulate random opponent
        opponent_actions = get_empty_cells(state)
        if opponent_actions:
            opp_action = random.choice(opponent_actions)
            state[opp_action] = 2

        winner = check_winner(state)
        next_index = state_to_index(state)
        reward = -1 if winner == 2 else 0 if winner == -1 else 0
        max_future = np.max(q_table[next_index])
        q_table[s_index][action] += alpha * (reward + gamma * max_future - q_table[s_index][action])
        if winner != 0:
            done = True

np.save("q_table.npy", q_table)
print("Training complete. Q-table saved.")
