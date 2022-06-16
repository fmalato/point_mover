import gym
import time
import numpy as np
import pybullet as p
from copy import deepcopy
from gym import utils
from gym import spaces


class BulletGeometryMover(gym.Env):

    def __init__(self, max_timesteps, frame_skip=1, camera_distance=5, on_linux=False, limit_fps=False):
        super().__init__()
        self.goal_state = [np.random.uniform(low=0.0, high=1.0), np.random.uniform(low=0.0, high=1.0)]
        self.camera_position = None
        self.step_count = 0
        self.max_timesteps = max_timesteps
        self.frame_skip = frame_skip
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        })
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(2,), dtype=np.float32)
        self.on_linux = on_linux
        if not self.on_linux:
            self.connection = p.connect(p.GUI)
        else:
            self.connection = p.connect(p.DIRECT)
        self.limit_fps = limit_fps
        self.boxId = p.loadMJCF("bullet_geometry_mover/bullet_geometry_mover/envs/bullet_geometry_mover.xml")
        self.camera_distance = camera_distance
        # Debugging variables
        self.last_action = None
        self.last_distance = None
        self.current_obs = None
        self.num_envs = 1
        self.rod_pos = None

    def step(self, a):
        # For debugging purposes
        done = False
        self.last_action = a
        pointerPos, pointerOrn = p.getBasePositionAndOrientation(self.boxId[0])
        new_pointer_pos = [np.clip(pointerPos[0] + a[0], -3.0, 3.0),
                           pointerPos[1],
                           np.clip(pointerPos[2] + a[1], -3.0, 3.0)]

        p.resetBasePositionAndOrientation(self.boxId[0], new_pointer_pos, pointerOrn)
        p.resetDebugVisualizerCamera(self.camera_distance, 0, 0, new_pointer_pos)
        # TODO: change to relative coordinates    DONE
        # TODO: DEBUG: value maps from actor + critic
        self.camera_position = [new_pointer_pos[0], new_pointer_pos[2]]
        # self.camera_position = [new_pointer_pos[0], new_pointer_pos[2]]
        distance = (np.sqrt(np.power((self.camera_position[0] - self.goal_state[0]), 2) +
                            np.power((self.camera_position[1] - self.goal_state[1]), 2)))
        # For debugging purposes
        self.last_distance = distance
        p.stepSimulation()
        if self.limit_fps:
            time.sleep(1. / 60.)
        cost = -1.0
        #cost = -0.5 * distance
        if distance <= 0.1:
            cost = 0.0
            done = True
        else:
            done = False

        self.step_count += 1
        if self.step_count > self.max_timesteps:
            done = True
        """if (new_pointer_pos[0] == 2.0 or new_pointer_pos[0] == -2.0) or new_pointer_pos[2] == 3.0 or new_pointer_pos[2] == 0.0:
            done = True"""

        obs = self._get_obs()
        self.current_obs = obs

        return obs, cost, done, {}

    def _get_obs(self):
        return {"observation": np.concatenate([self.camera_position, self.goal_state]),
                "achieved_goal": np.array(self.camera_position),
                "desired_goal": np.array(self.goal_state)}

    def reset(self):
        # Reset rod position
        self.rod_pos = [np.random.uniform(low=-2.0, high=2.0), 0, np.random.uniform(low=1.0, high=3.0)]
        #rod_pos = [2, 0, 3]
        p.resetBasePositionAndOrientation(self.boxId[1], self.rod_pos, [0, 0, 0, 1])
        self.goal_state = [self.rod_pos[0], self.rod_pos[2]]
        # Reset pointer position
        self.restart_position = [np.random.uniform(low=-2.0, high=2.0), 1, np.random.uniform(low=0.0, high=3.0)]
        p.resetBasePositionAndOrientation(self.boxId[0], self.restart_position, [0, 0, 0, 1])
        self.camera_position = [self.restart_position[0], self.restart_position[2]]
        # self.camera_position = [restart_position[0], restart_position[2]]
        # Reset camera position
        p.resetDebugVisualizerCamera(self.camera_distance, 0, 0, [self.restart_position[0], 0, self.restart_position[2]])
        # Reset step count
        self.step_count = 0

        return self._get_obs()

    def _distance(self, a, b):
        return np.sqrt(np.power((a[0] - b[0]), 2) + np.power((a[1] - b[1]), 2))

    def compute_reward(self, achieved_goal, desired_goal, info):
        rewards = np.zeros(shape=(len(achieved_goal,)))
        for a, d, i in zip(achieved_goal, desired_goal, range(len(achieved_goal))):
            if self._distance(a, d) <= 0.1:
                rewards[i] = 1.0
            else:
                rewards[i] = 0.0

        return rewards
