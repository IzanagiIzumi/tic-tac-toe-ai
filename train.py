import numpy as np
import random

# Initialize Q-table: 3^9 states, 9 possible moves
q_table = np.zeros((3**9, 9))

# Hyperparameters
alpha = 0.1    # learning rate
gamma = 0.9    # discount factor (future reward discount)
epsilon = 0.2  # exploration rate (random moves)

def state_to_index(state):
    index = 0
    for i in range(9):
        index *= 3
        index += state[i]
    return index

def check_winner(board):
    lines = [
        [0,1,2], [3,4,5], [6,7,8],  # rows
        [0,3,6], [1,4,7], [2,5,8],  # cols
        [0,4,8], [2,4,6]            # diagonals
    ]
    for a, b, c in lines:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if 0 not in board:
        return 0  # Draw
    return None  # Game ongoing

def available_moves(board):
    return [i for i, x in enumerate(board) if x == 0]

def make_move(board, move, player):
    new_board = board[:]
    new_board[move] = player
    return new_board

def choose_move(state_idx, available, q_table):
    # Epsilon-greedy policy
    if random.uniform(0, 1) < epsilon:
        return random.choice(available)
    else:
        q_values = q_table[state_idx]
        # Choose the best available move
        best_moves = [m for m in available if q_values[m] == max(q_values[mv] for mv in available)]
        return random.choice(best_moves)

def train(episodes=50000):
    global q_table
    for episode in range(episodes):
        board = [0]*9
        player = 1  # AI plays as 1
        states_actions = []

        while True:
            state_idx = state_to_index(board)
            moves = available_moves(board)

            if player == 1:
                # AI move (learning)
                move = choose_move(state_idx, moves, q_table)
            else:
                # Opponent move (random)
                move = random.choice(moves)

            board = make_move(board, move, player)

            winner = check_winner(board)
            next_state_idx = state_to_index(board)

            if player == 1:
                states_actions.append((state_idx, move, next_state_idx))

            if winner is not None:
                # Game ended, assign rewards
                for state_idx, move, next_state_idx in reversed(states_actions):
                    if winner == 1:
                        reward = 1  # AI won
                    elif winner == 0:
                        reward = 0.5  # Draw
                    else:
                        reward = -1  # AI lost

                    old_value = q_table[state_idx][move]
                    future_max = 0 if next_state_idx == state_idx else max(q_table[next_state_idx])
                    q_table[state_idx][move] = old_value + alpha * (reward + gamma * future_max - old_value)
                break

            # Switch player
            player = 2 if player == 1 else 1

        # Optional: print progress every 5000 episodes
        if (episode+1) % 5000 == 0:
            print(f"Episode {episode+1}/{episodes} complete")

if __name__ == "__main__":
    train(episodes=50000)
    np.save("q_table.npy", q_table)
    print("Training complete. Q-table saved to q_table.npy")
