import numpy as np


class NNet:
    def __init__(self) -> None:
        pass

    def predict(self, observation_tensor: np.ndarray):
        # return np.ones(np.prod(observation_tensor[0].shape), dtype=np.float32), 0.5
        value = (np.sum(observation_tensor[0]) - np.sum(observation_tensor[1])) / np.prod(
            observation_tensor[0].shape
        )
        # print(f"{value} value")
        return np.ones(np.prod(observation_tensor[0].shape), dtype=np.float32), value
