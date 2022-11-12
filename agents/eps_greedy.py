import math


class EpsGreedy:
    def __init__(
        self,
        eps_start: float = 0.5,
        eps_end: float = 1.0,
        eps_decay: float = 0.99,
    ) -> None:
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def get_threshold(self) -> float:
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done * self.eps_decay
        )
        return eps_threshold

    def update_step(self):
        self.steps_done += 1
