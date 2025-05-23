from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)
q_table = np.load("q_table.npy", allow_pickle=True)

@app.route("/")
def home():
    return "Tic Tac Toe AI is running!"

@app.route("/move", methods=["POST"])
def move():
    data = request.get_json()
    state = tuple(data["state"])
    state_index = state_to_index(state)
    action = int(np.argmax(q_table[state_index]))
    return jsonify({"move": action})

def state_to_index(state):
    index = 0
    for i in range(9):
        index *= 3
        index += state[i]
    return index

if __name__ == "__main__":
    app.run()
