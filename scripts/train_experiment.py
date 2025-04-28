from src.utils.data_loader import load_data
from src.simulator import Simulator
from src.bandits.linucb import LinUCB
from src.bandits.thompson_sampling import ThompsonSampling
from src.bandits.epsilon_greedy import EpsilonGreedy
from src.evaluation.experiment_runner import run_experiment
from src.utils.data_loader import load_data

if __name__ == "__main__":
    data = load_data(
        path="data/criteo_sample.csv",
        context_features=["I1", "I2", "C1", "C2"],
        nrows=100_000
    )

    # Initialize simulator
    simulator = Simulator(
        data,
        context_features=["I1", "I2", "C1", "C2"],
        num_arms=10
    )

    agents = {
        "LinUCB": LinUCB(alpha=0.1),
        "ThompsonSampling": ThompsonSampling(),
        "EpsilonGreedy": EpsilonGreedy(epsilon=0.1)
    }

    run_experiment(simulator, agents)