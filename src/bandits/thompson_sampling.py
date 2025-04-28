import numpy as np
from src.bandits.base_bandit import BaseBandit

class ThompsonSampling(BaseBandit):
    def __init__(self, prior_alpha=1, prior_beta=1):
        self.alphas = {}
        self.betas = {}
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def select_arm(self, context, available_arms):
        samples = []
        for arm in available_arms:
            if arm not in self.alphas:
                self.alphas[arm] = self.prior_alpha
                self.betas[arm] = self.prior_beta
            sample = np.random.beta(self.alphas[arm], self.betas[arm])
            samples.append(sample)
        return available_arms[np.argmax(samples)]

    def update(self, context, chosen_arm, reward):
        if reward == 1:
            self.alphas[chosen_arm] += 1
        else:
            self.betas[chosen_arm] += 1