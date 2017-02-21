from common.events import Subscriber


class Logger(Subscriber):
    def __init__(self):
        self.agent = None
        self._subscribe('algorithm.train_episode', Logger._on_train)
        self._subscribe('algorithm.eval', Logger._on_eval)
        self._subscribe('world.action', Logger._on_world_action)
        self._subscribe('timer', Logger._on_timer)

    # noinspection PyMethodMayBeStatic
    def _on_timer(self, times: dict):
        for k, v in times.items():
            print("\t%s:%4.2f" % (k, v), end='')

    def _on_train(self, info):
        if info['episode'] % 10 == 0:
            print()
        print("\n%4d:\tr:%+8.2f\tq:%+8.2f\txy:[%4.1f, %4.1f]" % (
            info['episode'],
            info['reward'],
            info['qmax'],
            self.agent[0],
            self.agent[1],
        ), end='')

    # noinspection PyMethodMayBeStatic
    def _on_eval(self, info):
        print("\te:%+8.2f\t%-6s" % (
            info['ave_reward'],
            'DONE' if info['ave_done'] > .5 else '',
        ), end='')

    def _on_world_action(self, info):
        self.agent = info['agent'].copy()
