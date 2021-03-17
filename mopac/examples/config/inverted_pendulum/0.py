params = {
    'type': 'MOPAC',
    'universe': 'gym',
    'domain': 'InvertedPendulum',
    'task': 'v2',

    'log_dir': '~/ray_mopac/',
    'exp_name': 'mopac_entropy0_mix5',

    'kwargs': {
        'n_epochs': 80, ## 20k steps
        'epoch_length': 250,
        'train_every_n_steps': 1,
        'n_train_repeat': 10,
        'eval_render_mode': None,
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'mopac': True,
        'valuefunc': True,
        'deterministic_obs': False,
        'deterministic_rewards': False,
        'rollout_schedule': [0, 500, 5, 5],
        'ratio_schedule': [0, 70, 0.05, 0.05],
        'rollout_batch_size': 10000,

        'model_train_freq': 250,
        'model_train_end_epoch': 70,
        'model_retain_epochs': 1,
        'num_networks': 7,
        'num_elites': 5,
        'target_entropy': 0,
        'max_model_t': None,
    }
}
