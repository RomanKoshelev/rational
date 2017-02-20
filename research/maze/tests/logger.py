from common.events import Events


class Logger(object):
    def __init__(self):
        self.agent = None

    def subcribe(self):
        Events.subscribe('algorithm.train_episode_end', Logger.on_algorithm_episode, self)
        Events.subscribe('world.action', Logger.on_world_action, self)

    def on_algorithm_episode(self, info):
        print("%4d:\tnr:%.2f\t\tr:%+8.2f\tq:%+8.2f\txy:[%4.1f, %4.1f]\t%-6s" % (
            info['episode'],
            info['nrate'],
            info['reward'],
            info['qmax'],
            self.agent[0],
            self.agent[1],
            'done' if info['done'] else '',
        ))

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
