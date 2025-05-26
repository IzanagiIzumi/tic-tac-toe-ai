import numpy as np
import random

# Constants
BOARD_SIZE = 9
NUM_STATES = 3 ** BOARD_SIZE
NUM_ACTIONS = BOARD_SIZE
EPISODES = 50000
ALPHA = 0.1       # Learning rate
GAMMA = 0.9       # Discount factor
EPSILON = 1.0     # Initial exploration rate
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01

# Initialize Q-table
q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

def state_to_index(state):
    index = 0
    for i in range(BOARD_SIZE):
        index *= 3
        index += state[i]
    return index

def get_available_actions(state):
    return [i for i in range(BOARD_SIZE) if state[i] == 0]

def check_winner(state):
    lines = [
        [0,1,2], [3,4,5], [6,7,8],
        [0,3,6], [1,4,7], [2,5,8],
        [0,4,8], [2,4,6]
    ]
    for a, b, c in lines:
        if state[a] != 0 and state[a] == state[b] == state[c]:
            return state[a]
    return 0 if 0 in state else -1  # 0=game ongoing, -1=draw

def train():
    global EPSILON
    for episode in range(EPISODES):
        state = [0] * BOARD_SIZE
        done = False

        while not done:
            index = state_to_index(state)
            actions = get_available_actions(state)

            if random.random() < EPSILON:
                action = random.choice(actions)
            else:
                q_values = q_table[index]
                valid_q = [q_values[a] if a in actions else -np.inf for a in range(NUM_ACTIONS)]
                action = int(np.argmax(valid_q))

            state[action] = 1  # Player move
            winner = check_winner(state)
            if winner != 0:
                reward = 1 if winner == 1 else 0
                q_table[index][action] += ALPHA * (reward - q_table[index][action])
                break

            # AI (opponent) random move
            opponent_actions = get_available_actions(state)
            if not opponent_actions:
                break
            opponent_move = random.choice(opponent_actions)
            state[opponent_move] = 2

            winner = check_winner(state)
            reward = 0
            if winner == 2:
                reward = -1
                done = True
            elif winner == -1:
                reward = 0.5
                done = True

            next_index = state_to_index(state)
            next_best = max(q_table[next_index][a] for a in get_available_actions(state)) if get_available_actions(state) else 0
            q_table[index][action] += ALPHA * (reward + GAMMA * next_best - q_table[index][action])

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    np.save("q_table.npy", q_table)
    print("Training complete. Q-table saved.")

if __name__ == "__main__":
    train()
