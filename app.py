import os
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = DQN()
model.load_state_dict(torch.load("dqn_tictactoe.pth", map_location=torch.device("cpu")))
model.eval()

@app.route("/", methods=["GET"])
def home():
    return "Tic Tac Toe AI (Deep Q-Learning) is running!"

@app.route("/move", methods=["POST"])
def move():
    data = request.get_json()
    state = data.get("state", None)

    if not state or len(state) != 9:
        return jsonify({"error": "Invalid state"}), 400

    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor).numpy()[0]

    for i in range(9):
        if state[i] != 0:
            q_values[i] = -float('inf')

    action = int(np.argmax(q_values))

    return jsonify({"move": action})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
