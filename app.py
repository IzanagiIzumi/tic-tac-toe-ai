import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained Q-table
try:
    q_table = np.load("q_table.npy", allow_pickle=True)
except FileNotFoundError:
    print("❌ q_table.npy not found! Make sure you've run train.py.")
    q_table = np.zeros((3**9, 9))  # fallback: empty table

def state_to_index(state):
    index = 0
    for i in range(9):
        index *= 3
        index += state[i]
    return index

@app.route("/", methods=["GET"])
def home():
    return "Tic Tac Toe AI is running!"

@app.route("/move", methods=["POST"])
def move():
    data = request.get_json()
    state = data.get("state")

    if not isinstance(state, list) or len(state) != 9:
        return jsonify({"error": "Invalid state"}), 400

    # Convert state to index
    state_index = state_to_index(state)

    # Get Q-values and filter for valid moves
    q_values = q_table[state_index]
    valid_moves = [i for i in range(9) if state[i] == 0]

    if not valid_moves:
        return jsonify({"error": "No valid moves"}), 400

    # Choose best move among valid ones
    best_move = max(valid_moves, key=lambda i: q_values[i])
    return jsonify({"move": best_move})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
