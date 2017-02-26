# https://github.com/jaara/AI-blog
import numpy as np
from .sum_tree import SumTree


class ExperienceMemory:
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        assert len(sample) == 5  # s,a,r,s2,d
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
