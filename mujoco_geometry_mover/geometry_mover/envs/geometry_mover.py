import numpy as np
from copy import deepcopy
from gym import utils
from gym import spaces
from gym.envs.mujoco import mujoco_env


class GeometryMover(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, frame_skip=1, max_timesteps=100, on_linux=False):
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
        self.on_linux = on_linux
        if self.on_linux:
            mujoco_env.MujocoEnv.__init__(self,
                                          "/home/federima/point_mover/mujoco_geometry_mover/geometry_mover/envs/geometry_mover.xml",
                                          self.frame_skip)
        else:
            mujoco_env.MujocoEnv.__init__(self,
                                          "/Users/federico/PycharmProjects/point_mover/mujoco_geometry_mover/geometry_mover/envs/geometry_mover.xml",
                                          self.frame_skip)

    def step(self, a):
        new_pos = self.sim.data.get_body_xpos('pointer')
        new_pos = new_pos + np.array([a[0], 1, a[1]])
        self.sim.data.body_xpos[1] = new_pos
        self.sim.data.geom_xpos[1] = new_pos
        self.sim.forward()
        self.do_simulation(a, self.frame_skip)
        #pos = self.get_body_com("pointer")[:3]
        #self.model.geom_pos[1] = -deepcopy(pos)
        self.camera_position = [new_pos[0], new_pos[2]]
        distance = (np.sqrt(np.power((self.camera_position[0] - self.goal_state[0]), 2) +
                            np.power((self.camera_position[1] - self.goal_state[1]), 2)))
        cost = -1.0
        #cost = -0.5*distance
        if distance <= 0.05:
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
        #self.model.body_pos[1] = [np.random.uniform(low=0.0, high=1.0), 0.75, np.random.uniform(low=0.0, high=1.0)]
        pos = self.get_body_com("pointer")[:3]
        self.camera_position = [pos[0], pos[2]]
        #self.model.geom_pos[1] = -deepcopy(pos)
        self.goal_state = [np.random.uniform(low=0.0, high=1.0), np.random.uniform(low=0.0, high=1.0)]
        restart_position = [np.random.uniform(low=-1.0, high=1.0), np.random.uniform(low=-1.0, high=1.0)]
        # TODO: consider case of more than one object
        self.model.body_pos[2] = [np.random.uniform(low=-2.0, high=2.0), 0, np.random.uniform(low=2.0, high=4.0)]
        # self.model.body_quat[2] = [1, 0, np.random.randint(low=-90, high=90), 0]
        self.set_state(np.array([restart_position[0], restart_position[1]]),
                       np.array(self.goal_state))
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
