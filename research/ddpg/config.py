from .algorithm.config import config
from .algorithm.ddpg import Ddpg
from .env.target_task import RandomTargetTask
from .env.target_world import TargetWorld

# algorithm
config['algorithm.class'] = Ddpg

# ddpg
config['ddpg.buffer_size'] = 15 * 1000
config['ddpg.actor_tau'] = 0.1  # 0.001
config['ddpg.actor_lr'] = 1e-5  # 1e-4
config['ddpg.critic_lr'] = 1e-3  # 1e-3
config['ddpg.critic_tau'] = 0.1  # 0.001

# world
config['world.class'] = TargetWorld
config['world.dim'] = 2
config['world.size'] = 20
config['world.agent_step'] = 1.5

# task
config['task.class'] = RandomTargetTask
config['task.done_dist'] = 1.
config['task.reward_done'] = 1000
config['task.reward_dist'] = 10

# train
config['train.episodes'] = 2000
config['train.steps'] = 10

