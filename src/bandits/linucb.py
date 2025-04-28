import numpy as np
from src.bandits.base_bandit import BaseBandit

class LinUCB(BaseBandit):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.A = {}
        self.b = {}

    def select_arm(self, context, available_arms):
        context_vector = np.array(list(context.values()))
        p_values = []
        for arm in available_arms:
            if arm not in self.A:
                d = len(context_vector)
                self.A[arm] = np.identity(d)
                self.b[arm] = np.zeros(d)
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv.dot(self.b[arm])
            p = theta.dot(context_vector) + self.alpha * np.sqrt(context_vector.dot(A_inv).dot(context_vector))
            p_values.append(p)
        return available_arms[np.argmax(p_values)]

    def update(self, context, chosen_arm, reward):
        context_vector = np.array(list(context.values()))
        self.A[chosen_arm] += np.outer(context_vector, context_vector)
        self.b[chosen_arm] += reward * context_vector