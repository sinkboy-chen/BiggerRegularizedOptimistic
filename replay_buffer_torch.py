"""
Simple replay buffer with GPU transfer for PyTorch BRO training.
Based on the official BRO Torch version's replay_buffer.py.
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, buffer_size: int, observation_size: int, action_size: int, device: str = 'cpu'):
        self.observations = np.empty((buffer_size, observation_size), dtype=np.float32)
        self.actions = np.empty((buffer_size, action_size), dtype=np.float32)
        self.rewards = np.empty((buffer_size,), dtype=np.float32)
        self.dones = np.empty((buffer_size,), dtype=np.float32)
        self.next_observations = np.empty((buffer_size, observation_size), dtype=np.float32)
        self.size = 0
        self.insert_index = 0
        self.buffer_size = buffer_size
        self.device = device

    def add(self, observation, next_observation, action, reward, done):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.dones[self.insert_index] = done
        self.next_observations[self.insert_index] = next_observation
        self.insert_index = (self.insert_index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample_multibatch(self, batch_size: int, num_batches: int):
        """Sample multiple batches at once, returning tensors on device.

        Returns shapes: (num_batches, batch_size, ...)
        """
        indx = np.random.randint(self.size, size=(num_batches, batch_size))
        observations = self._to_tensor(self.observations[indx])
        next_observations = self._to_tensor(self.next_observations[indx])
        actions = self._to_tensor(self.actions[indx])
        rewards = self._to_tensor(self.rewards[indx])
        dones = self._to_tensor(self.dones[indx])
        return observations, next_observations, actions, rewards, dones

    def _to_tensor(self, array: np.ndarray):
        return torch.from_numpy(array).to(self.device)
