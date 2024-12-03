# import torch
# from pandas.core.indexes.accessors import Properties
#
#
# class Memory:
#     def __init__(self, config):
#         """
#         Initialize the memory buffer with GPU support.
#
#         Args:
#             config (dict): Configuration dictionary with the following keys:
#                 - memory_size_max (int): Maximum size of the memory buffer.
#                 - device (torch.device): Device to store the memory (e.g., 'cuda' for GPU).
#         """
#         self.max_size = config['memory_size_max']
#         self.device = config['device']
#         self.ptr = 0
#         self.size = 0
#
#         # Pre-allocate memory on the GPU for storing tuples
#         self.old_states = torch.zeros((self.max_size, config['num_states']), dtype=torch.float, device=self.device)
#         self.actions = torch.zeros((self.max_size, config['num_actions']), dtype=torch.float, device=self.device)
#         self.rewards = torch.zeros((self.max_size, 1), dtype=torch.float, device=self.device)
#         self.current_states = torch.zeros((self.max_size, config['num_states']), dtype=torch.float, device=self.device)
#
#     def add_sample(self, sample):
#         """
#         Add a sample to the memory buffer.
#
#         Args:
#             sample (tuple): Tuple containing (old_state, action, reward, current_state).
#         """
#         old_state, action, reward, current_state = sample
#
#         # Store each component in the appropriate buffer
#         self.old_states[self.ptr] = torch.tensor(old_state, dtype=torch.float, device=self.device)
#         self.actions[self.ptr] = torch.tensor(action, dtype=torch.float, device=self.device)
#         self.rewards[self.ptr] = torch.tensor(reward, dtype=torch.float, device=self.device)
#         self.current_states[self.ptr] = torch.tensor(current_state, dtype=torch.float, device=self.device)
#
#         # Update the pointer and size
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
#         # print(self.size)
#
#     # def add_sample_batch(self, batch_size):
#     #     """
#     #     Sample a batch of experiences from the memory buffer.
#     #
#     #     Args:
#     #         batch_size (int): Number of samples to retrieve.
#     #
#     #     Returns:
#     #         Tuple[torch.Tensor]: Batches of old_states, actions, rewards, and current_states.
#     #     """
#     #     indices = torch.randint(0, self.size, size=(batch_size,), device=self.device)
#     #     self.size += batch_size
#     #     return (
#     #         self.old_states[indices],
#     #         self.actions[indices],
#     #         self.rewards[indices],
#     #         self.current_states[indices]
#     #     )
#     def add_sample_batch(self, batch):
#         """
#         Add a batch of samples to the memory buffer.
#
#         Args:
#             batch (list or numpy array): Batch of samples,
#                 where each sample is a tuple (old_state, action, reward, current_state).
#         """
#         batch_size = len(batch)
#         assert batch_size <= self.max_size, "Batch size exceeds the memory buffer size."
#
#         # Unpack the batch into individual components
#         old_states, actions, rewards, current_states = zip(*batch)
#
#         # Convert to tensors
#         old_states = torch.tensor(old_states, dtype=torch.float, device=self.device)
#         actions = torch.tensor(actions, dtype=torch.float, device=self.device)
#         rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
#         current_states = torch.tensor(current_states, dtype=torch.float, device=self.device)
#
#         # Determine the insertion indices
#         end_ptr = self.ptr + batch_size
#         if end_ptr <= self.max_size:
#             # No wrap-around
#             self.old_states[self.ptr:end_ptr] = old_states
#             self.actions[self.ptr:end_ptr] = actions
#             self.rewards[self.ptr:end_ptr] = rewards
#             self.current_states[self.ptr:end_ptr] = current_states
#         else:
#             # Wrap-around handling
#             first_part_size = self.max_size - self.ptr
#             self.old_states[self.ptr:self.max_size] = old_states[:first_part_size]
#             self.actions[self.ptr:self.max_size] = actions[:first_part_size]
#             self.rewards[self.ptr:self.max_size] = rewards[:first_part_size]
#             self.current_states[self.ptr:self.max_size] = current_states[:first_part_size]
#
#             wrap_size = end_ptr - self.max_size
#             self.old_states[:wrap_size] = old_states[first_part_size:]
#             self.actions[:wrap_size] = actions[first_part_size:]
#             self.rewards[:wrap_size] = rewards[first_part_size:]
#             self.current_states[:wrap_size] = current_states[first_part_size:]
#
#         # Update the pointer and size
#         self.ptr = end_ptr % self.max_size
#         self.size = min(self.size + batch_size, self.max_size)
#
#     def get_sample_batch(self, batch_size):
#         """
#         Retrieve a random batch of samples from the memory buffer.
#
#         Args:
#             batch_size (int): Number of samples to retrieve.
#
#         Returns:
#             tuple: A tuple containing batches of (old_states, actions, rewards, current_states).
#         """
#         # Ensure there are enough samples in the memory
#         batch_size = min(self.size, batch_size)
#
#         # Randomly sample indices
#         indices = torch.randint(0, self.size, (batch_size,), device=self.device)
#
#         # Retrieve samples at the sampled indices
#         batch_old_states = self.old_states[indices]
#         batch_actions = self.actions[indices]
#         batch_rewards = self.rewards[indices]
#         batch_current_states = self.current_states[indices]
#
#         return batch_old_states, batch_actions, batch_rewards, batch_current_states
#
#     def size_now(self):
#         """
#         Get the current size of the memory buffer.
#
#         Returns:
#             int: Number of samples currently in the buffer.
#         """
#         return self.size
import torch

class Memory:
    def __init__(self, config):
        """
        Initialize the memory buffer with GPU support.

        Args:
            config (dict): Configuration dictionary with the following keys:
                - memory_size_max (int): Maximum size of the memory buffer.
                - device (torch.device): Device to store the memory (e.g., 'cuda' for GPU).
                - num_states (int): Dimension of the state space.
                - num_actions (int): Number of possible actions.
        """
        self.max_size = config['memory_size_max']
        self.device = config['device']
        self.ptr = 0
        self.size = 0

        # Pre-allocate memory on the GPU for storing tuples
        self.old_states = torch.zeros((self.max_size, config['num_states']), dtype=torch.float, device=self.device)
        self.actions = torch.zeros((self.max_size, 1), dtype=torch.long, device=self.device)  # Changed to long
        self.rewards = torch.zeros((self.max_size, 1), dtype=torch.float, device=self.device)
        self.current_states = torch.zeros((self.max_size, config['num_states']), dtype=torch.float, device=self.device)

    def add_sample(self, sample):
        """
        Add a sample to the memory buffer.

        Args:
            sample (tuple): Tuple containing (old_state, action, reward, current_state).
        """
        old_state, action, reward, current_state = sample

        # Store each component in the appropriate buffer
        self.old_states[self.ptr] = torch.tensor(old_state, dtype=torch.float, device=self.device)
        self.actions[self.ptr] = torch.tensor([action], dtype=torch.long, device=self.device)  # Changed to long and added []
        self.rewards[self.ptr] = torch.tensor([reward], dtype=torch.float, device=self.device)
        self.current_states[self.ptr] = torch.tensor(current_state, dtype=torch.float, device=self.device)

        # Update the pointer and size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        # print(self.size)


    def get_sample_batch(self, batch_size):
        """
        Retrieve a random batch of samples from the memory buffer.

        Args:
            batch_size (int): Number of samples to retrieve.

        Returns:
            tuple: A tuple containing batches of (old_states, actions, rewards, current_states).
        """
        # Ensure there are enough samples in the memory
        batch_size = min(self.size, batch_size)

        # Randomly sample indices
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        # Retrieve samples at the sampled indices
        batch_old_states = self.old_states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_current_states = self.current_states[indices]

        return batch_old_states, batch_actions, batch_rewards, batch_current_states

    def size_now(self):
        """
        Get the current size of the memory buffer.

        Returns:
            int: Number of samples currently in the buffer.
        """
        # print("memory" + str(self.size))
        return self.size


if __name__ == '__main__':
    memory = Memory({'memory_size_max': 10, 'device': 'cuda', 'num_states':10,'num_actions':2})
    for i in range(10):
        memory.add_sample((torch.randn(10), torch.randn(2), 1.0, torch.randn(10)))
        memory.size_now()
    memory.size_now()