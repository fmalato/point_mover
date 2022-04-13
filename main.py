import gym
from gym.wrappers import TimeLimit
from datetime import datetime
from stable_baselines3 import DDPG, PPO, HerReplayBuffer
from stable_baselines3.ppo.policies import MlpPolicy
from wrappers import PositionOnlyWrapper


if __name__ == '__main__':
    max_episode_length = 50
    online_sampling = True
    num_sampled_goals = 4
    buffer_size = 200000
    lr = 1e-4
    total_timesteps = 100000
    save = True
    model_name = "DDPG_HER_100k_mujoco"
    goal_selection_strategy = 'future'
    env = gym.make('geometry_mover:geometry_mover-v0')
    #env = PositionOnlyWrapper(env)
    env = TimeLimit(env, max_episode_steps=max_episode_length)
    obs = env.reset()
    # TODO: DDPG AC + HER
    env.test = False
    """model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)"""
    model = DDPG(policy="MultiInputPolicy",
                 env=env,
                 replay_buffer_class=HerReplayBuffer,
                 buffer_size=150000,
                 learning_rate=lr,
                 replay_buffer_kwargs=dict(
                     n_sampled_goal=num_sampled_goals,
                     goal_selection_strategy=goal_selection_strategy,
                     online_sampling=online_sampling,
                     max_episode_length=max_episode_length,
                 ),
                 verbose=1)
    model.learn(total_timesteps=total_timesteps)
    fname = datetime.now().strftime("%H_%M_%S")
    if save:
        model.save("saved_models/{name}".format(name=model_name))
    """env.test = True
    for i in range(200):
        obs = env.reset()
        total_reward = 0.0
        done = False
        num_steps = 0
        print('Episode {i}'.format(i=i+1))
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            total_reward += rewards
            num_steps += 1
            env.render()
            #env.record(obs, done, fname)
        print('Episode reward: {r} - Number of steps: {s}'.format(r=total_reward, s=num_steps))
        print('Final position: {x}'.format(x=obs))"""
