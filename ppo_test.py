import gym
import numpy as np
import pybullet as p
from gym.wrappers import TimeLimit
from datetime import datetime
from stable_baselines3 import DDPG, HerReplayBuffer, PPO
from stable_baselines3.common.noise import NormalActionNoise
from utils import ValuesCallback, TensorboardCallback


if __name__ == '__main__':
    max_episode_length = 1000
    online_sampling = True
    num_sampled_goals = 4
    num_test_games = 20
    buffer_size = 150000
    lr = 1e-5
    total_timesteps = 500000
    train = True
    save = True
    on_linux = True
    tb_log_name = "PPO_500k_3D_norm"
    if train:
        limit_fps = False
    else:
        limit_fps = True
    model_name = "PPO_500k_3D_2dof"
    goal_selection_strategy = 'future'
    env = gym.make('bullet_geometry_mover:GeometryMover-v0', max_timesteps=max_episode_length, on_linux=on_linux,
                   limit_fps=limit_fps, frame_skip=10)
    env = TimeLimit(env, max_episode_steps=max_episode_length)
    obs = env.reset()
    if train:
        env.test = False
        model = PPO(policy="MultiInputPolicy", env=env, verbose=1, tensorboard_log='tensorboard_logs/')
    else:
        model = PPO.load("saved_models/PPO_200k_3D_2dof.zip")
    callback = TensorboardCallback(verbose=0)
    if train:
        model.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name=tb_log_name)
    fname = datetime.now().strftime("%H_%M_%S")
    if save:
        model.save("saved_models/{name}".format(name=model_name))
    env.test = True
    for i in range(num_test_games):
        obs = env.reset()
        total_reward = 0.0
        done = False
        num_steps = 0
        print('########## Episode {i} ##########'.format(i=i+1))
        print('Initial position: {x}'.format(x=obs['observation']))
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            total_reward += rewards
            num_steps += 1
        print('Final position: {x}'.format(x=obs['observation']))
        print('Episode reward: {r} - Number of steps: {s}'.format(r=total_reward, s=num_steps))
    p.disconnect(env.env.connection)
