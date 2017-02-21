from common.events import Events
from time import time


class Timer(object):
    def __init__(self):
        self.agent = None
        self.history = {}
        self.subcribe()

    def subcribe(self):
        Events.subscribe('algorithm.train_episode', Timer.on_train, self)

    def on_train(self, _):
        e = 'train'
        self.add_history(e)
        d = self.last_interval(e)
        if d is not None:
            Events.send('timer', {e: d})

    def add_history(self, e):
        if e not in self.history.keys():
            self.history[e] = []
        self.history[e].append(time())

    def last_interval(self, e):
        h = list(self.history.get(e, []))
        l = len(h) - 1
        return h[l] - h[l - 1] if l > 0 else None
