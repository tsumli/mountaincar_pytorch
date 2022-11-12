from abc import ABCMeta, abstractmethod


class BaseAgent(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def select_action(self, state):
        raise NotImplementedError()

    @abstractmethod
    def learn(self, state, action, reward, next_state, episode, step):
        raise NotImplementedError()
