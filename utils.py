from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.ddpg import DDPG


class ValuesCallback(BaseCallback):

    def __init__(self, verbose):
        super().__init__(verbose=verbose)
        self.values = []

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        params = self.model.get_parameters()
        params_keys = params['policy'].keys()
        self.values.append([(x, params['policy'][x]) for x in params_keys if 'weight' in x])

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        print(self.values)
