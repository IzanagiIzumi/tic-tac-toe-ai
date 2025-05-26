import os
from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained Q-table from file
q_table = np.load("q_table.npy", allow_pickle=True)

@app.route("/")
def home():
    return "Tic Tac Toe AI is running!"

@app.route("/move", methods=["POST"])
def move():
    try:
        data = request.get_json()
        state = tuple(data["state"])
        state_index = state_to_index(state)
        action = int(np.argmax(q_table[state_index]))
        return jsonify({"move": action})
    except Exception as e:
        print("Error in /move:", e)
        return jsonify({"error": "Invalid request or internal error"}), 400


def state_to_index(state):
    """
    Converts a 9-element state (with values 0, 1, or 2) into a unique index.
    This works by interpreting the state as a base-3 number.
    """
    index = 0
    for i in range(9):
        index *= 3
        index += state[i]
    return index

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
