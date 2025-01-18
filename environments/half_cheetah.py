# HalfCheetah-v2 env
# two objectives
# running speed, energy efficiency

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from os import path


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.gonca_steps = 0
        self.obj_dim = 2
        mujoco_env.MujocoEnv.__init__(
            self,
            model_path=path.join(
                path.abspath(path.dirname(__file__)), "assets/half_cheetah.xml"
            ),
            frame_skip=5,
        )
        utils.EzPickle.__init__(self)

    def reset(self):
        self.gonca_steps = 0
        return super().reset()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)
        xposafter, ang = self.sim.data.qpos[0], self.sim.data.qpos[2]
        ob = self._get_obs()

        reward_run = ((xposafter - xposbefore) / self.dt) / 16
        reward_energy = 1 - np.square(action).sum() / self.action_space.shape[0]

        done = False
        self.gonca_steps += 1

        return (
            ob,
            0.0,
            done,
            {"obj": np.array([reward_run, reward_energy]), "Final_Position": xposafter},
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
