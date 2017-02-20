from common.events import Events


class Logger(object):
    def __init__(self):
        self.agent = None
        self.episode = None

    def subcribe(self):
        Events.subscribe('algorithm.train_episode_end', Logger.on_algorithm_episode, self)
        Events.subscribe('world.action', Logger.on_world_action, self)

    def on_algorithm_episode(self, info):
        if self.episode is None or self.episode % 10 == 0:
            self.print_header()
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
        if self.agent is not None:
            print("    pos: [%3.1f, %3.1f] act: [%3.1f, %3.1f]" % (
                self.agent[0],
                self.agent[1],
                info['action'][0],
                info['action'][1],
            ), end='\r')
        self.agent = info['agent'].copy()

    @staticmethod
    def print_header():
        pass
        # print("\nEP\tNR\tREWARD\tQMax\tAGENT")
