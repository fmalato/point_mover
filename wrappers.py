from gym import ObservationWrapper
from gym import spaces


class PositionOnlyWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['position']

    def observation(self, obs):
        return obs['position']