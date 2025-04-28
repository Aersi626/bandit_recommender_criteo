import pandas as pd
import numpy as np

class Simulator:
    def __init__(self, data, context_features, num_arms):
        self.data = data
        self.context_features = context_features
        self.num_arms = num_arms
        self.index = 0

    def reset(self):
        self.index = 0

    def get_context_and_candidates(self):
        row = self.data.iloc[self.index]
        context = row[self.context_features].to_dict()
        arms = np.random.choice(self.data['ad_id'].unique(), self.num_arms, replace=False)
        return context, arms

    def get_reward(self, chosen_arm):
        row = self.data.iloc[self.index]
        reward = 1 if row['ad_id'] == chosen_arm and row['click'] == 1 else 0
        self.index += 1
        return reward