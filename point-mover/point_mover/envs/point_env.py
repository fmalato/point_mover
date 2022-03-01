import gym
import numpy as np
import pyglet
import json
import matplotlib.pyplot as plt

from gym import spaces
from copy import deepcopy


class PointMover(gym.GoalEnv):

    def __init__(self, max_speed=0.02, image_size=64, max_timesteps=500):
        self.max_speed = max_speed
        self.image_size = image_size
        self.max_timesteps = max_timesteps
        self.max_episode_length = max_timesteps
        self.step_count = 0
        self.viewer = None
        # Action space is a speed vector with (x, y) coordinates
        self.action_space = spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(2,), dtype=np.float32)
        # Observation space includes both position vector of the dot and the raw image
        # TODO: include goal encoding (DONE)
        # "rgb_image": spaces.Box(low=0.0, high=1.0, shape=(self.image_size, self.image_size, 3), dtype=np.float32),
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
            "desired_goal": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        })
        self.point_position = np.array([0.5, 0.5], dtype=np.float32)
        self.image = np.zeros(shape=(self.image_size, self.image_size, 3), dtype=np.float32)
        self.image[int(self.image_size/2), int(self.image_size/2), :] = 1.0
        # TODO: make it random (DONE)
        self.goal_state = [0.5, 0.5]
        self.discrete_x_goal = int(self.goal_state[0] * self.image_size)
        self.discrete_y_goal = int(self.goal_state[1] * self.image_size)
        self.fig = plt.imshow(self.image)
        self.obs_record_buffer = {}
        self.current_episode = None
        self.test = False

    def step(self, action):
        # TODO: Extend to move differently across different regions of the image (fisheye effect) (DONE)
        distance_from_center = (np.sqrt(np.power((self.point_position[0] - 0.5), 2) +
                                np.power((self.point_position[1] - 0.5), 2)))
        if 0 <= distance_from_center <= 0.15:
            zone = 1
        elif 0.15 < distance_from_center <= 0.30:
            zone = 2
        else:
            zone = 3
        self.point_position[0] -= action[0] * zone
        self.point_position[1] -= action[1] * zone
        self.point_position = np.clip(self.point_position, a_min=0.0, a_max=1.0)
        self.image = np.zeros(shape=(self.image_size, self.image_size, 3), dtype=np.float32)
        discrete_x = int(self.point_position[0] * self.image_size)
        discrete_y = int(self.point_position[1] * self.image_size)
        discrete_x = int(np.clip(discrete_x, 0, 63))
        discrete_y = int(np.clip(discrete_y, 0, 63))
        self.image[discrete_x, discrete_y, :] = 1.0
        self.image[self.discrete_x_goal, self.discrete_y_goal, 0] = 1.0

        # TODO: Discretize -1 for wrong time step, 0 for goal
        distance = (np.sqrt(np.power((self.point_position[0] - self.goal_state[0]), 2) +
                            np.power((self.point_position[1] - self.goal_state[1]), 2)))
        cost = -distance * 0.02
        #cost = -1.0
        if distance <= 0.01:
            cost += 2.0
            done = True
        else:
            if self.step_count < self.max_timesteps:
                done = False
            else:
                done = True

        self.step_count += 1
        obs = list(np.concatenate([self.point_position, self.goal_state], axis=0))

        return {"observation": obs, "achieved_goal": [obs[0], obs[1]], "desired_goal": self.goal_state}, cost, done, {}

    def reset(self):
        # Reset state
        self.goal_state = [np.random.uniform(low=0.0, high=1.0), np.random.uniform(low=0.0, high=1.0)]
        self.point_position = [np.random.uniform(low=0.0, high=1.0), np.random.uniform(low=0.0, high=1.0)]
        self.image = np.zeros(shape=(self.image_size, self.image_size, 3), dtype=np.float32)
        discrete_x = int(self.point_position[0] * self.image_size)
        discrete_y = int(self.point_position[1] * self.image_size)
        self.discrete_x_goal = int(self.goal_state[0] * self.image_size)
        self.discrete_y_goal = int(self.goal_state[1] * self.image_size)
        self.image[discrete_x, discrete_y, :] = 1.0
        self.image[self.discrete_x_goal, self.discrete_y_goal, 0] = 1.0
        # Reset step count
        self.step_count = 0
        obs = list(np.concatenate([self.point_position, self.goal_state], axis=0))
        # Reset trajectory buffer
        if self.test:
            if self.current_episode is None:
                self.current_episode = 0
            else:
                self.current_episode += 1
            self.obs_record_buffer[self.current_episode] = {}
            self.obs_record_buffer[self.current_episode]['obs'] = []
            self.obs_record_buffer[self.current_episode]['reward'] = []

        return {"observation": obs, "achieved_goal": [obs[0], obs[1]], "desired_goal": self.goal_state}

    def render(self, mode="human"):
        self.fig.set_data(self.image)
        plt.show(block=False)
        plt.pause(1/10)

    def compute_reward(self, achieved_goal, desired_goal, info):
        rewards = np.zeros(shape=(len(achieved_goal,)))
        for a, d, i in zip(achieved_goal, desired_goal, range(len(achieved_goal))):
            if a[0] == d[0] and a[1] == d[1]:
                rewards[i] = 0.0
            else:
                rewards[i] = -1.0

        return rewards

    def record(self, obs, done, fname):
        self.obs_record_buffer[self.current_episode]['obs'].append(obs)
        if done:
            self.obs_record_buffer[self.current_episode]['reward'].append(0.0)
            with open('{now}.json'.format(now=fname), 'a+') as f:
                f.seek(0)
                json.dump(self.obs_record_buffer, f, indent=4)
        else:
            self.obs_record_buffer[self.current_episode]['reward'].append(-1.0)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
