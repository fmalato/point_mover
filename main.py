import gym
from gym.wrappers import TimeLimit
from datetime import datetime
from stable_baselines3 import DDPG, PPO, HerReplayBuffer
from stable_baselines3.ppo.policies import MlpPolicy
from wrappers import PositionOnlyWrapper


if __name__ == '__main__':
    env = gym.make('point_mover:point_mover-v0')
    #env = PositionOnlyWrapper(env)
    env = TimeLimit(env, max_episode_steps=100)
    obs = env.reset()
    # TODO: DDPG AC + HER
    env.test = False
    """model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)"""
    model = DDPG(policy="MultiInputPolicy",
                 env=env,
                 replay_buffer_class=HerReplayBuffer,
                 buffer_size=50000,
                 verbose=1)
    model.learn(total_timesteps=100000)
    fname = datetime.now().strftime("%H_%M_%S")
    env.test = True
    for i in range(2):
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
            #env.render()
            #env.record(obs, done, fname)
        print('Episode reward: {r} - Number of steps: {s}'.format(r=total_reward, s=num_steps))
        print('Final position: {x}'.format(x=obs))
