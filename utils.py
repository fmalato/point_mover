import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.ddpg import DDPG


class ValuesCallback(BaseCallback):

    def __init__(self, verbose):
        super().__init__(verbose=verbose)
        self.actions = np.mgrid[-0.05:0.05:0.01j, -0.05:0.05:0.01j]

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        """with open('values.csv', 'a+') as f:
            values = []
            for x in range(self.actions[1].shape[0]):
                v = []
                for y in range(self.actions[1].shape[0]):
                    v.append(self.model.policy.)"""
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass


class TensorboardCallback(BaseCallback):

    def __init__(self, verbose):
        super().__init__(verbose=verbose)
        self.values = []

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        action_x, action_y = self.training_env.envs[0].env.env.last_action
        distance_from_goal = self.training_env.envs[0].env.env.last_distance
        last_state = self.training_env.envs[0].env.env.current_obs
        state = {"observation": torch.Tensor(np.array(last_state["observation"])),
                 "achieved_goal": torch.Tensor(np.array(last_state["achieved_goal"])),
                 "desired_goal": torch.Tensor(np.array(last_state["desired_goal"]))}
        self.logger.record('action_x', action_x)
        self.logger.record('action_y', action_y)
        self.logger.record('distance_from_goal', distance_from_goal)
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass
