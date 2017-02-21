from research.maze.ddpg.config import config
from research.maze.worlds.target_world import TargetWorld

# world
config['world.class'] = TargetWorld
config['world.dim'] = 2
config['world.size'] = 20
config['world.agent_step'] = 1.

# task
config['task.done_dist'] = .5
config['task.reward_done'] = 1000
config['task.reward_dist'] = 10

# train
config['train.episodes'] = 2000
config['train.steps'] = 10
