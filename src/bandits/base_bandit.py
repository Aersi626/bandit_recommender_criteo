from abc import ABC, abstractmethod

class BaseBandit(ABC):
    @abstractmethod
    def select_arm(self, context, available_arms):
        pass

    @abstractmethod
    def update(self, context, chosen_arm, reward):
        pass