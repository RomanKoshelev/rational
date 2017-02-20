from .noise_tools import linear_05_00

config = {
    # actor
    'alg.actor.h1': 400,
    'alg.actor.h2': 300,
    'alg.actor.l2': 0.0,
    'alg.actor.lr': 0.0001,
    'alg.actor.tau': 0.001,

    # critic
    'alg.critic.h1': 400,
    'alg.critic.h2': 300,
    'alg.critic.l2': 0.01,
    'alg.critic.lr': 0.001,
    'alg.critic.tau': 0.001,

    # common
    'alg.gamma': 0.99,  # FUTURE REWARD DECAY

    # train
    'alg.buffer_size': 100 * 1000,
    'alg.batch_size': 128,
    'alg.noise_sigma': .1,
    'alg.noise_theta': .01,
    'alg.noise_rate_method': linear_05_00,
}
