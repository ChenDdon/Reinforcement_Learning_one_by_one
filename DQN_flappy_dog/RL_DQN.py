import numpy as np
import pandas as pd
import torch
import torch.nn as nn


np.random.seed(0)
torch.manual_seed(0)

class RL_DQN(object):
    def __init__(self,
                 n_actions,
                 n_features,
                 epsilon = 0.8,
                 alpha = 0.01,
                 gamma = 0.9,
                 replace_target_iter = 1000,
                 memory_size = 2000,
                 batch_size = 32,
                 e_greedy_change = False):
        
        self.n_actions = n_actions  
        self.n_features = n_features
        self.lr = alpha
        self.reward_decay = gamma
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.e_greedy_change = e_greedy_change
        self.e_greedy = epsilon

        #record the learn_step num
        self.learn_step_counter = 0

        # initialize memory, length = n_featres + s_' n_features + a + a_
        self.memory = np.zeros((self.memory_size, n_features*2 + 2))

        # build two nets, q_value, q_target_value
        self._build_net()

    def _build_net(self):
        
        n_neural = 16

        # neural net 1
        self.q_func = nn.Sequential(nn.Linear(self.n_features, n_neural),
                                    nn.Tanh(),
                                    nn.Linear(n_neural, n_neural),
                                    nn.Tanh(),
                                    nn.Linear(n_neural, self.n_actions))
        nn.init.xavier_uniform_(self.q_func[0].weight)
        nn.init.xavier_uniform_(self.q_func[2].weight)

        # neural net 2
        self.q_target_func = self.q_func

        # optimizer, only q_value network should be trained
        self.optimizer = torch.optim.SGD(self.q_func.parameters(), lr=self.lr)

        # loss function
        self.loss_func = nn.MSELoss()

    def store_transition(self, state, action, reward, state_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((state, action, reward, state_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, state):
        '''e-greedy method'''

        if np.random.uniform() < self.e_greedy:
            # forward feed the state and get q_value for all actions
            aciton_all_value = self.q_target_func(torch.FloatTensor(state))
            action = np.expand_dims(np.argmax(aciton_all_value.cpu().data.numpy(), axis=1), axis=1)
        else:
            action = np.expand_dims([np.random.randint(0, self.n_actions)], axis=1)

        return action

    def update_q_function(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            model_parameters = self.q_func.state_dict()
            self.q_target_func.load_state_dict(model_parameters)
            print('\ntarget_params_replaced: {:d}\n'.format(self.learn_step_counter))
        
        # sample batch memory from all memory random choice
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # q_value, forward feed <state>
        q_value = self.q_func(torch.FloatTensor(batch_memory[:, :self.n_features]))

        # q_target_value, forward feed <state_> 
        q_target_value = self.q_target_func(torch.FloatTensor(batch_memory[:, -self.n_features:]))
        
        # change q_target as q_eval's action, here has little triky.
        q_target = q_value.cpu().data.numpy().copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_action_index = batch_memory[:, self.n_features].astype(int)

        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_action_index] = \
            reward + self.reward_decay * np.max(q_target_value.cpu().data.numpy(), axis=1)

        # train q_func, update parameters
        self.optimizer.zero_grad()
        loss = self.loss_func(q_value, torch.FloatTensor(q_target))
        loss.backward()
        self.optimizer.step()
        
        self.learn_step_counter += 1

        
if __name__ == '__main__':
    DQN = RL_DQN(2, 3)

