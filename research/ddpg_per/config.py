from .algorithm.config import config
from .env.target_task import RandomTargetTask
from .env.target_world import TargetWorld

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
config['train.episodes'] = 3000
config['train.steps'] = 10
