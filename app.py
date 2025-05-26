import os
from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)


q_table = np.load("q_table.npy", allow_pickle=True)

EPSILON = 0.1

@app.route("/", methods=["GET"])
def home():
    return "Tic Tac Toe AI is running!"

@app.route("/move", methods=["POST"])
def move():
    data = request.get_json()
    state = list(data.get("state", []))

    if not state or len(state) != 9:
        return jsonify({"error": "Invalid state"}), 400

    state_index = state_to_index(state)
    q_values = q_table[state_index]

    valid_moves = [i for i in range(9) if state[i] == 0]
    if not valid_moves:
        return jsonify({"error": "No valid moves"}), 400

    if random.random() < EPSILON:
        chosen_move = random.choice(valid_moves)
    else:
        chosen_move = max(valid_moves, key=lambda a: q_values[a])

    print(f"State: {state}")
    print(f"Valid moves: {valid_moves}")
    print(f"Chosen move: {chosen_move} (random choice: {random.random() < EPSILON})")

    return jsonify({"move": chosen_move})

def state_to_index(state):
    index = 0
    for i in range(9):
        index *= 3
        index += state[i]
    return index

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
