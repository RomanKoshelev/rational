from common.events import EventSystem, Subscriber
from time import time


class Timer(Subscriber):
    def __init__(self):
        self.agent = None
        self.history = {}
        self._subscribe('algorithm.train_episode', Timer._on_train)

    def _on_train(self, _):
        e = 'train'
        self._add_history(e)
        d = self._get_last_interval(e)
        if d is not None:
            EventSystem.send('timer', {e: d})

    def _add_history(self, e):
        if e not in self.history.keys():
            self.history[e] = []
        self.history[e].append(time())

    def _get_last_interval(self, e):
        h = list(self.history.get(e, []))
        l = len(h) - 1
        return h[l] - h[l - 1] if l > 0 else None
