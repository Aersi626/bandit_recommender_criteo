from flask import Flask, request, jsonify
from src.bandits.linucb import LinUCB
from src.simulator import Simulator
from src.utils.data_loader import load_data

app = Flask(__name__)

# Load dummy simulator and agent (could preload trained states later)
data = load_data()
simulator = Simulator(data, context_features=["C1", "banner_pos"], num_arms=10)
agent = LinUCB(alpha=0.1)

@app.route("/recommend", methods=["POST"])
def recommend():
    content = request.json
    context = content.get("context")
    available_arms = content.get("available_arms")

    chosen_arm = agent.select_arm(context, available_arms)
    return jsonify({"chosen_arm": int(chosen_arm)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)