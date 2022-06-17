import gym
import numpy as np
import pybullet as p
from gym.wrappers import TimeLimit
from datetime import datetime
from stable_baselines3 import DDPG, HerReplayBuffer, DQN
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from utils import ValuesCallback, TensorboardCallback


if __name__ == '__main__':
    max_episode_length = 1000
    online_sampling = True
    num_sampled_goals = 4
    num_test_games = 20
    buffer_size = 500000
    lr = 1e-5
    total_timesteps = 500000
    train = True
    save = True
    on_linux = True
    tb_log_name = "500k_3D_ornstein"
    if train:
        limit_fps = False
    else:
        limit_fps = False
    model_name = "DDPG_HER_500k_3D_ornstein"
    goal_selection_strategy = 'future'
    env = gym.make('bullet_geometry_mover:GeometryMover-v0', max_timesteps=max_episode_length, on_linux=on_linux,
                   limit_fps=limit_fps, frame_skip=10)
    #env = gym.make('point_mover:point_mover-v0', max_timesteps=max_episode_length)
    env = TimeLimit(env, max_episode_steps=max_episode_length)
    obs = env.reset()
    # TODO: DDPG AC + HER
    if train:
        env.test = False
        # TODO: after first learning step, the agent stops moving meaningfully
        """
        Is it possible that there are slow movements at the beginning for some reason, 
        and then using HER reinforces choosing small movements over big leaps?
        Still, why should it move only on one direction?"""
        """
        action_noise=NormalActionNoise(mean=0, sigma=1)
        """
        model = DDPG(policy="MultiInputPolicy",
                     env=env,
                     replay_buffer_class=HerReplayBuffer,
                     buffer_size=buffer_size,
                     learning_rate=lr,
                     replay_buffer_kwargs=dict(
                         n_sampled_goal=num_sampled_goals,
                         goal_selection_strategy=goal_selection_strategy,
                         online_sampling=online_sampling,
                         max_episode_length=max_episode_length,
                     ),
                     verbose=1,
                     device="cuda",
                     learning_starts=10000,
                     batch_size=1000,
                     tensorboard_log='tensorboard_logs/',
                     action_noise=OrnsteinUhlenbeckActionNoise(mean=np.array([1.2, 1.2]), sigma=np.array([0.3, 0.3]),
                                                               theta=0.05, initial_noise=np.ndarray([2, 2])))
    else:
        model = DDPG.load('saved_models/DDPG_HER_100k_3D_2dof.zip',
                          env=env)
        """model = DDPG(policy="MultiInputPolicy",
                     env=env,
                     replay_buffer_class=HerReplayBuffer,
                     buffer_size=buffer_size,
                     learning_rate=lr,
                     replay_buffer_kwargs=dict(
                         n_sampled_goal=num_sampled_goals,
                         goal_selection_strategy=goal_selection_strategy,
                         online_sampling=online_sampling,
                         max_episode_length=max_episode_length,
                     ),
                     verbose=1)"""
    if train:
        eval_env = gym.make('bullet_geometry_mover:GeometryMover-v0', max_timesteps=max_episode_length, on_linux=True,
                            limit_fps=False, frame_skip=10)
        eval_callback = EvalCallback(eval_env=eval_env, deterministic=True, log_path='evaluation_logs/',
                                     eval_freq=20000)
        checkpoint_callback = CheckpointCallback(save_freq=20000, save_path='checkpoints/',
                                                 name_prefix='{x}_chkp'.format(x=model_name))
        model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback],
                    tb_log_name=tb_log_name)
        fname = datetime.now().strftime("%H_%M_%S")
        if save:
            model.save("saved_models/{name}".format(name=model_name))
    else:
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
                # Just for the 2D env
                #env.render()
            print('Final position: {x}'.format(x=obs['observation']))
            print('Episode reward: {r} - Number of steps: {s}'.format(r=total_reward, s=num_steps))
    p.disconnect(env.env.connection)
