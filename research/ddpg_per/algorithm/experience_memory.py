# https://github.com/jaara/AI-blog
import numpy as np
from .sum_tree import SumTree


class ExperienceMemory:
    epsilon = 1e-5

    def __init__(self, capacity, degree):
        self.tree = SumTree(capacity)
        self.degree = degree

    def _get_priority(self, error):
        return (error + self.epsilon) ** self.degree

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def get_batch(self, n):
        batch = []
        segment = self.tree.total() / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            (idx, _, data) = self.tree.get(s)
            assert data is not None
            batch.append([idx, data])

        return batch

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
