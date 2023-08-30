# Swimmer-v2 env
# two objectives
# forward speed, energy efficiency

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from os import path

class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.obj_dim = 2
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, model_path = path.join(path.abspath(path.dirname(__file__)), "assets/swimmer.xml"), frame_skip = 4)
        self.action_space_type = "Continuous"
        self.reward_space = np.zeros((2,))
    def step(self, a):
        info = self._get_info()
        ctrl_cost_coeff = 0.15
        xposbefore = self.sim.data.qpos[0]
        # if isinstance(a, (np.ndarray)):
        #     a = a[0]
        a = np.clip(a, -1, 1)
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = 0.3 - ctrl_cost_coeff * np.square(a).sum()
        ob = self._get_obs()
        return ob, np.array([reward_fwd, reward_ctrl],dtype=np.float32), False, info

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def _get_info(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return  np.concatenate([qpos.flat, qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv),
        )
        return self._get_obs()
