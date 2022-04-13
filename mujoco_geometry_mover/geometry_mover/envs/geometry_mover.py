import numpy as np
from gym import utils
from gym import spaces
from gym.envs.mujoco import mujoco_env


class GeometryMover(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, frame_skip=1, max_timesteps=100):
        utils.EzPickle.__init__(self)
        self.goal_state = [np.random.uniform(low=0.0, high=1.0), np.random.uniform(low=0.0, high=1.0)]
        self.camera_position = None
        self.step_count = 0
        self.max_timesteps = max_timesteps
        self.frame_skip = frame_skip
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
            "desired_goal": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        })
        mujoco_env.MujocoEnv.__init__(self,
                                      "/Users/federico/PycharmProjects/point_mover/mujoco_geometry_mover/geometry_mover/envs/geometry_mover.xml",
                                      self.frame_skip)
        # Linux version
        """mujoco_env.MujocoEnv.__init__(self,
                                      "/home/federima/point_mover/mujoco_geometry_mover/geometry_mover/envs/geometry_mover.xml",
                                      self.frame_skip)"""

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        pos = self.get_body_com("pointer")[:3]
        self.camera_position = [pos[0], pos[2]]
        distance = (np.sqrt(np.power((self.camera_position[0] - self.goal_state[0]), 2) +
                            np.power((self.camera_position[1] - self.goal_state[1]), 2)))
        cost = -1.0
        if distance <= 0.01:
            cost = 0.0
            done = True
        else:
            done = False

        self.step_count += 1
        obs = self._get_obs()

        return obs, cost, done, {}

    def _get_obs(self):
        return {"observation": np.concatenate([self.camera_position, self.goal_state]),
                "achieved_goal": np.array(self.camera_position),
                "desired_goal": np.array(self.goal_state)}

    def reset_model(self):
        pos = self.get_body_com("pointer")[:3]
        self.camera_position = [pos[0], pos[2]]
        self.goal_state = [np.random.uniform(low=0.0, high=1.0), np.random.uniform(low=0.0, high=1.0)]
        restart_position = [np.random.uniform(low=-1.0, high=1.0), np.random.uniform(low=-1.0, high=1.0)]
        self.set_state(np.array([restart_position[0], 1, restart_position[1], 0, 0, 0, 0]),
                       np.concatenate([self.goal_state, [0, 0, 0, 0]]))
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def _distance(self, a, b):
        return np.sqrt(np.power((a[0] - b[0]), 2) + np.power((a[1] - b[1]), 2))

    def compute_reward(self, achieved_goal, desired_goal, info):
        rewards = np.zeros(shape=(len(achieved_goal,)))
        for a, d, i in zip(achieved_goal, desired_goal, range(len(achieved_goal))):
            if self._distance(a, d) <= 0.01:
                rewards[i] = 0.0
            else:
                rewards[i] = -1.0

        return rewards
