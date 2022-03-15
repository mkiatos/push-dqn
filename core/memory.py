from collections import deque
from random import Random
import numpy as np
import pickle
import os


class Transition:
    def __init__(self,
                 state=None,
                 action=None,
                 reward=None,
                 next_state=None,
                 terminal=None,
                 info=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminal = terminal
        self.info = info

    def array(self):
        return np.array([self.state, self.action, self.reward, self.next_state, self.terminal])

    def __str__(self):
        return '\n---\nTransition:' + \
               '\nstate:\n' + str(self.state) + \
               '\naction: ' + str(self.action) + \
               '\nreward: ' + str(self.reward) + \
               '\nnext_state:\n' + str(self.next_state) + \
               '\nterminal: ' + str(self.terminal) + \
               '\n---'

    def __copy__(self):
        return Transition(state=copy.copy(self.state), action=copy.copy(self.action),
                          reward=copy.copy(self.reward), next_state=copy.copy(self.next_state),
                          terminal=copy.copy(self.terminal), info=copy.copy(self.info))

    def copy(self):
        return self.__copy__()


class ReplayBuffer:
    """
    Implementation of the replay experience buffer. Creates a buffer which uses
    the deque data structure. Here you can store experience transitions (i.e.: state,
    action, next state, reward) and sample mini-batches for training.
    You can  retrieve a transition like this:
    Example of use:
    .. code-block:: python
        replay_buffer = ReplayBuffer(10)
        replay_buffer.store()
        replay_buffer.store([0, 2, 1], [1, 2], -12.9, [2, 2, 1], 0)
        # ... more storing
        transition = replay_buffer(2)
    Parameters
    ----------
    buffer_size : int
        The buffer size
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.random = Random()
        self.count = 0

    def __call__(self, index):
        """
        Returns a transition from the buffer.

        Parameters
        ----------
        index : int
            The index number of the desired transition

        Returns
        -------
        tuple
            The transition
        """
        return self.buffer[index]

    def store(self, transition):
        """
        Stores a new transition on the buffer.

        Parameters
        ----------
        transition: Transition
            A transition of {state, action, next_state, reward, terminal}
        """
        if self.count < self.buffer_size:
            self.count += 1
        else:
            self.buffer.popleft()
        self.buffer.append(transition)

    def sample_batch(self, given_batch_size):
        """
        Samples a mini-batch from the buffer.

        Parameters
        ----------
        given_batch_size : int
            The size of the mini-batch.

        Returns
        -------
        numpy.array
            The state batch
        numpy.array
            The action batch
        numpy.array
            The reward batch
        numpy.array
            The next state batch
        numpy.array
            The terminal batch
        """

        if self.count < given_batch_size:
            batch_size = self.count
        else:
            batch_size = given_batch_size

        batch = self.random.sample(self.buffer, batch_size)

        state_batch = np.array([_.state for _ in batch])
        action_batch = np.array([_.action for _ in batch])
        reward_batch = np.array([_.reward for _ in batch])
        next_state_batch = np.array([_.next_state for _ in batch])
        terminal_batch = np.array([_.terminal for _ in batch])

        return Transition(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)

    def clear(self):
        """
        Clears the buffer my removing all elements.
        """
        self.buffer.clear()
        self.count = 0

    def size(self):
        """
        Returns the current size of the buffer.
        Returns
        -------
        int
            The number of existing transitions.
        """
        return self.count

    def seed(self, random_seed):
        self.random.seed(random_seed)

    def save(self, file_path):
        b = {'buffer': self.buffer, 'buffer_size': self.buffer_size, 'count': self.count}
        pickle.dump(b, open(file_path, 'wb'))

    @classmethod
    def load(cls, file_path):
        b = pickle.load(open(file_path, 'rb'))
        self = cls(b['buffer_size'])
        self.buffer = b['buffer']
        self.count = b['count']
        return self

    def remove(self, index):
        del self.buffer[index]
        self.count -= 1