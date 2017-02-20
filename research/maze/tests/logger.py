from common.events import Events


class Logger(object):
    @staticmethod
    def subcribe():
        Events.subscribe('train_episode_end', Logger.on_episode_end)

    @staticmethod
    def on_episode_end(info):
        print("%4d %6.2f %6.1f %7.1f" % (
            info['episode'],
            info['nrate'],
            info['reward'],
            info['qmax']
        ))
