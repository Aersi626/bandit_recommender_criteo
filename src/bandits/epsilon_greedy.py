import numpy as np
from src.bandits.base_bandit import BaseBandit

class EpsilonGreedy(BaseBandit):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.counts = {}
        self.values = {}

    def select_arm(self, context, available_arms):
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_arms)
        estimates = []
        for arm in available_arms:
            estimates.append(self.values.get(arm, 0))
        return available_arms[np.argmax(estimates)]

    def update(self, context, chosen_arm, reward):
        if chosen_arm not in self.counts:
            self.counts[chosen_arm] = 0
            self.values[chosen_arm] = 0.0
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward