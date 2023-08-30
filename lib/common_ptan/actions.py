import numpy as np
import torch
import random

class ActionSelector:
    """
    Abstract class which converts scores to the actions
    """
    def __call__(self, scores):
        raise NotImplementedError


class ArgmaxActionSelector(ActionSelector):
    """
    Selects actions using argmax
    """
    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=0)

class EpsilonGreedyActionSelector(ActionSelector):
    def __init__(self, epsilon=0.05, selector=None):
        self.epsilon = epsilon
        self.selector = selector if selector is not None else ArgmaxActionSelector()

    def __call__(self, scores):
        scores = scores.cpu().numpy()
        assert isinstance(scores, np.ndarray)
        n_actions = scores.shape
        actions = self.selector(scores)
        if np.random.random() < self.epsilon:
            rand_actions = np.random.choice(n_actions[0])
            actions = rand_actions
        return actions
    
