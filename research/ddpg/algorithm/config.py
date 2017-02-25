from .noise_tools import constant_1

config = {
    # actor
    'ddpg.actor_h1': 400,
    'ddpg.actor_h2': 300,
    'ddpg.actor_l2': 0.0,
    'ddpg.actor_lr': 1e-4,
    'ddpg.actor_tau': 0.001,

    # critic
    'ddpg.critic_h1': 400,
    'ddpg.critic_h2': 300,
    'ddpg.critic_l2': 0.01,
    'ddpg.critic_lr': 1e-3,
    'ddpg.critic_tau': 0.001,

    # ruture reward decay
    'ddpg.gamma': 0.99,

    # train
    'ddpg.buffer_size': 100 * 1000,
    'ddpg.batch_size': 128,

    # noise
    'ddpg.noise_sigma': .5,
    'ddpg.noise_theta': .15,
    'ddpg.noise_rate_method': constant_1,
}
