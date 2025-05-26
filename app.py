import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

q_table = np.load("q_table.npy", allow_pickle=True)
ALPHA = 0.1
GAMMA = 0.9

def state_to_index(state):
    index = 0
    for i in range(9):
        index *= 3
        index += state[i]
    return index

def check_winner(state):
    lines = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6],
    ]
    for a, b, c in lines:
        if state[a] and state[a] == state[b] == state[c]:
            return state[a]
    return 0 if 0 in state else -1

@app.route("/", methods=["GET"])
def home():
    return "Tic Tac Toe AI is running!"

@app.route("/move", methods=["POST"])
def move():
    global q_table

    data = request.get_json()
    state = data["state"]
    state_index = state_to_index(state)

    q_values = q_table[state_index]
    valid_moves = [i for i in range(9) if state[i] == 0]
    masked_q = np.where(np.array(state) == 0, q_values, -np.inf)

    if len(valid_moves) == 0:
        return jsonify({"error": "No valid moves left."}), 400

    action = int(np.argmax(masked_q))
    if state[action] != 0:
        return jsonify({"error": "AI returned an invalid move."}), 500

    state[action] = 2
    reward = 1 if check_winner(state) == 2 else 0

    new_state_index = state_to_index(state)
    q_table[state_index][action] += ALPHA * (reward + GAMMA * np.max(q_table[new_state_index]) - q_table[state_index][action])

    np.save("q_table.npy", q_table)

    return jsonify({"move": action})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
