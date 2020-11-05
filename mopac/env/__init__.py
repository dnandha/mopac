import gym

MOPAC_ENVIRONMENT_SPECS = (
	{
        'id': 'AntTruncatedObs-v2',
        'entry_point': (f'mopac.env.ant:AntTruncatedObsEnv'),
    },
	{
        'id': 'HumanoidTruncatedObs-v2',
        'entry_point': (f'mopac.env.humanoid:HumanoidTruncatedObsEnv'),
    },
)

def register_mopac_environments():
    for mopac_environment in MOPAC_ENVIRONMENT_SPECS:
        gym.register(**mopac_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  MOPAC_ENVIRONMENT_SPECS)

    return gym_ids