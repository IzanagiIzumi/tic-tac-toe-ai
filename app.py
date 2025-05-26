import os
from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained Q-table from file
q_table = np.load("q_table.npy", allow_pickle=True)

@app.route("/", methods=["GET"])
def home():
    return "Tic Tac Toe AI is running!"

@app.route("/move", methods=["POST"])
def move():
    data = request.get_json()
    state = tuple(data["state"])
    state_index = state_to_index(state)

    q_values = q_table[state_index]

    # Find valid (empty) positions
    valid_moves = [i for i, val in enumerate(state) if val == 0]

    if not valid_moves:
        return jsonify({"error": "No valid moves left"}), 400

    # Filter Q-values to only valid moves
    best_move = max(valid_moves, key=lambda move: q_values[move])

    return jsonify({"move": best_move})

def state_to_index(state):
    """Converts a 9-element board state into a base-3 index"""
    index = 0
    for val in state:
        index = index * 3 + val
    return index

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
