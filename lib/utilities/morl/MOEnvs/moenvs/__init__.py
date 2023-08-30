from gym.envs.registration import register

register(
    id='dst-v0',
    entry_point='moenvs.dst:DeepSeaTreasureEnv',
    max_episode_steps=30,
)

register(
    id='ftn-v0',
    entry_point='moenvs.ftn:FruitTree',
)

register(
    id = 'MO-Ant-v2',
    entry_point = 'moenvs.MujocoEnvs:AntEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Hopper-v2',
    entry_point = 'moenvs.MujocoEnvs:HopperEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Hopper-v3',
    entry_point = 'moenvs.MujocoEnvs:HopperEnv3',
    max_episode_steps=500,
)

register(
    id = 'MO-HalfCheetah-v2',
    entry_point = 'moenvs.MujocoEnvs:HalfCheetahEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Walker2d-v2',
    entry_point = 'moenvs.MujocoEnvs:Walker2dEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Swimmer-v2',
    entry_point = 'moenvs.MujocoEnvs:SwimmerEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Humanoid-v2',
    entry_point = 'moenvs.MujocoEnvs:HumanoidEnv',
    max_episode_steps=1000,
)
