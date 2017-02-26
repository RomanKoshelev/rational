from research.ddpg.config import config
from research.ddpg_per.algorithm.ddpg_per import DdpgPer


# algorithm
config['algorithm.class'] = DdpgPer

# prioritized experience replay
config['per.degree'] = .6
