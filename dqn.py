from clt_core.util.memory import ReplayBuffer
from clt_core.core import Agent

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import math
import os
import pickle


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units):
        """
        Parameters
        ----------
        state_dim: int
            The dimensions of state.
        action_dim: int
            The dimensions of actions.
        hidden_units: list
            A list of the number of hidden units in each layer.
        """
        super(QNetwork, self).__init__()

        torch.manual_seed(0)

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(state_dim, hidden_units[0]))
        for i in range(1, len(hidden_units)):
            self.hidden_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            self.hidden_layers[i].weight.data.uniform_(-0.003, 0.003)
            self.hidden_layers[i].bias.data.uniform_(-0.003, 0.003)

        self.out = nn.Linear(hidden_units[i], action_dim)
        self.out.weight.data.uniform_(-0.003, 0.003)
        self.out.bias.data.uniform_(-0.003, 0.003)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
        return self.out(x)


class DQN(Agent):
    def __init__(self, state_dim, action_dim, params):
        super(DQN, self).__init__(name='dqn', params=params)
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        torch.manual_seed(0)

        self.network = QNetwork(state_dim, action_dim, self.params['hidden_units']).to(self.params['device'])
        self.target_network = QNetwork(state_dim, action_dim, self.params['hidden_units']).to(self.params['device'])

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.params['learning_rate'])
        self.loss = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.params['replay_buffer_size'])

        self.learn_step_counter = 0

        self.save_buffer = False

        self.info['qnet_loss'] = 0

    def predict(self, state):
        print('predict')
        s = torch.FloatTensor(state).to(self.params['device'])
        action_value = self.network(s).cpu().detach().numpy()
        return np.argmax(action_value)

    def explore(self, state):
        epsilon = self.params['epsilon_end'] + (self.params['epsilon_start'] - self.params['epsilon_end']) * \
                       math.exp(-1 * self.learn_step_counter / self.params['epsilon_decay'])
        self.info['epsilon'] = epsilon  # save for plotting
        if self.rng.uniform(0, 1) >= epsilon:
            return self.predict(state)
        print('explore')
        return self.rng.randint(0, self.action_dim)

    def q_value(self, state, action):
        s = torch.FloatTensor(state).to(self.params['device'])
        return self.network(s).cpu().detach().numpy()[action]

    def learn(self, transition):
        # Store transition to the replay buffer
        self.replay_buffer.store(transition)

        # If we have not enough samples just keep storing transitions to the
        # buffer and thus exit from learn.
        if self.replay_buffer.size() < self.params['init_replay_buffer_size']:
            return

        if not self.save_buffer:
            self.replay_buffer.save(os.path.join(self.params['log_dir'], 'replay_buffer'))
            self.save_buffer = True

        # Update target network
        new_target_params = {}
        for key in self.target_network.state_dict():
            new_target_params[key] = self.params['tau'] * self.target_network.state_dict()[key] + \
                                     (1 - self.params['tau']) * self.network.state_dict()[key]
        self.target_network.load_state_dict(new_target_params)

        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample_batch(self.params['batch_size'])
        batch.terminal = np.array(batch.terminal.reshape((batch.terminal.shape[0], 1)))
        batch.reward = np.array(batch.reward.reshape((batch.reward.shape[0], 1)))
        batch.action = np.array(batch.action.reshape((batch.action.shape[0], 1)))

        state = torch.FloatTensor(batch.state).to(self.params['device'])
        action = torch.LongTensor(batch.action.astype(int)).to(self.params['device'])
        next_state = torch.FloatTensor(batch.next_state).to(self.params['device'])
        terminal = torch.FloatTensor(batch.terminal).to(self.params['device'])
        reward = torch.FloatTensor(batch.reward).to(self.params['device'])

        if self.params['double_dqn']:
            best_action = self.network(next_state).max(1)[1]  # action selection
            q_next = self.target_network(next_state).gather(1, best_action.view(self.params['batch_size'], 1))
        else:
            q_next = self.target_network(next_state).max(1)[0].view(self.params['batch_size'], 1)

        q_target = reward + (1 - terminal) * self.params['discount'] * q_next
        q = self.network(state).gather(1, action)
        loss = self.loss(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.info['qnet_loss'] = loss.detach().cpu().numpy().copy()

    def seed(self, seed):
        super(DQN, self).seed(seed)
        self.replay_buffer.seed(seed)

    def save(self, save_dir, name):
        # Create directory
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir)

        # Save networks and log data
        torch.save({'network': self.network.state_dict(),
                    'target_network': self.target_network.state_dict()}, os.path.join(log_dir, 'model.pt'))
        log_data = {'params': self.params.copy(),
                    'learn_step_counter': self.learn_step_counter,
                    'state_dim': self.state_dim,
                    'action_dim': self.action_dim}
        pickle.dump(log_data, open(os.path.join(log_dir, 'log_data.pkl'), 'wb'))

    @classmethod
    def load(cls, log_dir):
        log_data = pickle.load(open(os.path.join(log_dir, 'log_data.pkl'), 'rb'))
        self = cls(state_dim=log_data['state_dim'],
                   action_dim=log_data['action_dim'],
                   params=log_data['params'])

        checkpoint = torch.load(os.path.join(log_dir, 'model.pt'))
        self.network.load_state_dict(checkpoint['network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        return self
