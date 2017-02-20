from common.events import Events


class Logger(object):
    def __init__(self):
        self.agent = None
        self.subcribe()

    def subcribe(self):
        Events.subscribe('algorithm.train_episode', Logger.on_train, self)
        Events.subscribe('algorithm.eval', Logger.on_eval, self)
        Events.subscribe('world.action', Logger.on_world_action, self)

    def on_train(self, info):
        if info['episode'] % 10 == 0:
            print()
        print("\n%4d:\tnr:%.2f\t\tr:%+8.2f\tq:%+8.2f\txy:[%4.1f, %4.1f]" % (
            info['episode'],
            info['nrate'],
            info['reward'],
            info['qmax'],
            self.agent[0],
            self.agent[1],
        ), end='')

    # noinspection PyMethodMayBeStatic
    def on_eval(self, info):
        print("\t\teval:%+8.2f\t%-6s" % (
            info['ave_reward'],
            'DONE' if info['ave_done'] > .5 else '',
        ), end='')

    def on_world_action(self, info):
        # self._print_action(info['action'])
        self.agent = info['agent'].copy()

    def _print_action(self, action):
        if self.agent is not None:
            print("    pos: [%3.1f, %3.1f] act: [%3.1f, %3.1f]" % (
                self.agent[0],
                self.agent[1],
                action[0],
                action[1],
            ), end='\r')
