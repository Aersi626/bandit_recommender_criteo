import numpy as np

def cumulative_reward(rewards):
    """
    Calculate cumulative rewards over time.

    Args:
        rewards (list or np.array): list of reward per timestep.

    Returns:
        np.array: cumulative sum of rewards.
    """
    return np.cumsum(rewards)

def cumulative_regret(rewards, optimal_reward):
    """
    Calculate cumulative regret over time.

    Args:
        rewards (list or np.array): list of reward per timestep.
        optimal_reward (float): the maximum possible reward (e.g., best achievable CTR).

    Returns:
        np.array: cumulative sum of regrets.
    """
    rewards = np.array(rewards)
    regrets = optimal_reward - rewards
    return np.cumsum(regrets)

def compute_ctr(rewards):
    """
    Calculate Click-Through Rate (CTR) at each time step.

    Args:
        rewards (list or np.array): list of reward per timestep.

    Returns:
        np.array: CTR at each step.
    """
    rewards = np.array(rewards)
    ctrs = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
    return ctrs