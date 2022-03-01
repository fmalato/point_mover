from gym import ObservationWrapper
from gym import spaces


class PositionOnlyWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['observation']

    def observation(self, obs):
        return obs['observation']
