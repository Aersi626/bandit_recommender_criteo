import numpy as np
import matplotlib.pyplot as plt
from src.evaluation.metrics import cumulative_reward, cumulative_regret

def run_experiment(simulator, agents, num_rounds=10000):
    results = {name: {"rewards": [], "regrets": []} for name in agents.keys()}
    optimal_ctr = 0.3  # assume max CTR possible is ~30% (adjustable)

    for name, agent in agents.items():
        simulator.reset()
        cumulative_rewards = []
        cumulative_regrets = []
        total_reward = 0

        for t in range(num_rounds):
            context, arms = simulator.get_context_and_candidates()
            chosen_arm = agent.select_arm(context, arms)
            reward = simulator.get_reward(chosen_arm)
            agent.update(context, chosen_arm, reward)

            total_reward += reward
            regret = optimal_ctr - reward

            cumulative_rewards.append(total_reward)
            cumulative_regrets.append(regret if t == 0 else cumulative_regrets[-1] + regret)

        results[name]["rewards"] = cumulative_rewards
        results[name]["regrets"] = cumulative_regrets

    # Plotting
    for name in agents.keys():
        plt.plot(results[name]["regrets"], label=f"{name} Regret")

    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.title("Bandit Algorithm Comparison")
    plt.show()

    return results