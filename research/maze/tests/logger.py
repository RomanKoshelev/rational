from common.events import Events


class Logger(object):
    def __init__(self):
        self.agent = None
        self.subcribe()

    def subcribe(self):
        Events.subscribe('algorithm.train_episode', Logger.on_train, self)
        Events.subscribe('algorithm.eval', Logger.on_eval, self)
        Events.subscribe('world.action', Logger.on_world_action, self)
        Events.subscribe('timer', Logger.on_timer, self)

    # noinspection PyMethodMayBeStatic
    def on_timer(self, times: dict):
        for k, v in times.items():
            print("\t%s:%4.2f" % (k, v), end='')

    def on_train(self, info):
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
    def on_eval(self, info):
        print("\te:%+8.2f\t%-6s" % (
            info['ave_reward'],
            'DONE' if info['ave_done'] > .5 else '',
        ), end='')

    def on_world_action(self, info):
        self.agent = info['agent'].copy()
