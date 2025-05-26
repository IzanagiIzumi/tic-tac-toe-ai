import numpy as np
import os

BOARD_SIZE = 9
NUM_STATES = 3 ** BOARD_SIZE
NUM_ACTIONS = BOARD_SIZE
EPISODES = 50000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

def state_to_index(state):
    index = 0
    for i in range(BOARD_SIZE):
        index *= 3
        index += state[i]
    return index

def index_to_state(index):
    state = [0] * BOARD_SIZE
    for i in reversed(range(BOARD_SIZE)):
        state[i] = index % 3
        index //= 3
    return state

def available_actions(state):
    return [i for i, v in enumerate(state) if v == 0]

def check_winner(state):
    lines = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6],
    ]
    for line in lines:
        a, b, c = line
        if state[a] != 0 and state[a] == state[b] == state[c]:
            return state[a]
    return 0 if 0 in state else -1 

def train():
    for episode in range(EPISODES):
        state = [0] * BOARD_SIZE
        done = False

        while not done:
            idx = state_to_index(state)
            if np.random.rand() < EPSILON:
                action = np.random.choice(available_actions(state))
            else:
                q_values = q_table[idx]
                masked_q = np.where(np.array(state) == 0, q_values, -np.inf)
                action = np.argmax(masked_q)

            state[action] = 1 
            winner = check_winner(state)
            if winner == 1:
                q_table[idx][action] += ALPHA * (1 - q_table[idx][action])
                break
            elif winner == -1:
                q_table[idx][action] += ALPHA * (0.5 - q_table[idx][action])
                break

            opponent_actions = available_actions(state)
            if not opponent_actions:
                break
            opp_action = np.random.choice(opponent_actions)
            state[opp_action] = 2
            winner = check_winner(state)
            reward = -1 if winner == 2 else 0
            next_idx = state_to_index(state)

            q_table[idx][action] += ALPHA * (reward + GAMMA * np.max(q_table[next_idx]) - q_table[idx][action])

    np.save("q_table.npy", q_table)
    print("Training complete.")

if __name__ == "__main__":
    if not os.path.exists("q_table.npy"):
        train()
    else:
        print("q_table.npy already exists.")
