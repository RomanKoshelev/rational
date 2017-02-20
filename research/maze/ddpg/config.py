from .noise_tools import linear_05_00

config = {
    # actor
    'ddpg.actor.h1': 400,
    'ddpg.actor.h2': 300,
    'ddpg.actor.l2': 0.0,
    'ddpg.actor.lr': 1e-4,
    'ddpg.actor.tau': 0.001,

    # critic
    'ddpg.critic.h1': 400,
    'ddpg.critic.h2': 300,
    'ddpg.critic.l2': 0.01,
    'ddpg.critic.lr': 1e-3,
    'ddpg.critic.tau': 0.001,

    # ruture reward decay
    'ddpg.gamma': 0.99,

    # train
    'ddpg.buffer_size': 100 * 1000,
    'ddpg.batch_size': 128,
    'ddpg.noise_sigma': .1,
    'ddpg.noise_theta': .01,
    'ddpg.noise_rate_method': linear_05_00,
}
