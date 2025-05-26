import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


Q_TABLE_PATH = "q_table.npy"
if os.path.exists(Q_TABLE_PATH):
    q_table = np.load(Q_TABLE_PATH, allow_pickle=True)
else:
    q_table = np.zeros((3**9, 9))


EPSILON = 0.1       
ALPHA = 0.5          
GAMMA = 0.9        


current_game_history = []

def state_to_index(state):
    index = 0
    for i in range(9):
        index *= 3
        index += state[i]
    return index

def check_winner(board):
    lines = [
        [0,1,2], [3,4,5], [6,7,8],  
        [0,3,6], [1,4,7], [2,5,8],  
        [0,4,8], [2,4,6]            
    ]
    for a,b,c in lines:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if 0 not in board:
        return 0 
    return None 

@app.route("/", methods=["GET"])
def home():
    return "Tic Tac Toe AI is running!"

@app.route("/move", methods=["POST"])
def move():
    global current_game_history

    data = request.get_json()
    state = data.get("state", None)
    if not state or len(state) != 9:
        return jsonify({"error": "Invalid state"}), 400

    state_index = state_to_index(state)
    valid_moves = [i for i, v in enumerate(state) if v == 0]
    if not valid_moves:
        return jsonify({"error": "No valid moves"}), 400

    import random
    if random.random() < EPSILON:
        action = random.choice(valid_moves)
    else:
        q_values = q_table[state_index]

        action = max(valid_moves, key=lambda a: q_values[a])


    current_game_history.append((state_index, action))

    return jsonify({"move": action})

@app.route("/train", methods=["POST"])
def train():
    global current_game_history, q_table

    data = request.get_json()
    result = data.get("result", None)
    final_state = data.get("final_state", None)

    if result is None or final_state is None:
        return jsonify({"error": "Missing training data"}), 400

    if result == 2:
        reward = 1
    elif result == 1:
        reward = -1
    else:
        reward = 0.5

    for i in reversed(range(len(current_game_history))):
        state_idx, action = current_game_history[i]

        if i == len(current_game_history) - 1:
            max_next_q = 0
        else:
            next_state_idx, _ = current_game_history[i+1]
            max_next_q = max(q_table[next_state_idx])

        old_value = q_table[state_idx][action]
        q_table[state_idx][action] = old_value + ALPHA * (reward + GAMMA * max_next_q - old_value)

        reward *= GAMMA

    current_game_history = []

    np.save(Q_TABLE_PATH, q_table)

    return jsonify({"message": "Training complete"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
