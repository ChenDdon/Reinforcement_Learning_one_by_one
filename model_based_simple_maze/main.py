"""
Introduction:
    Model based RL for MAZE
Author:
    Chend
Modify:
    2020-03-30
"""

# -------------------------- Dependence --------------------------
import os
import numpy as np
import pandas as pd
import copy
import time
import GivenModelMaze as GMM
from MazeEnv import Maze


# -------------------------- Reproducibility ---------------------
np.random.seed(12634)


# -------------------------- policy iterative --------------------
class Model_Based_RL():
    def __init__(self, 
                states_space,
                actions_space, 
                transition_function,
                reward_function,
                gamma,
                threshold):
        self.states_space  = states_space
        self.actions_space = actions_space
        self.transition_function = transition_function
        self.reward_function = reward_function
        self.discount_factor = gamma
        self.threshold = threshold

    def _initial_policy_function(self, states_space, actions_space):
        # initial policy function
        policy_function = pd.DataFrame(
            np.random.randint(0, 4, size=(len(states_space), 1)),
            index=states_space,
            dtype=int)
        return policy_function
    
    def _initial_value_function(self, states_space):
        # initial value function
        value_function = pd.DataFrame(np.zeros((len(states_space), 1)),
                                      index=states_space,
                                      dtype=np.float64)
        return value_function
    
    def _create_q_value_function(self, value_function):
        value_function_temp = np.dot(value_function, np.ones((1, len(self.actions_space))))
        q_value_function = value_function_temp + self.reward_function
        return q_value_function

    def cumulative_reward(self, state, policy_function, discount_factor):
        action = self.actions_space[policy_function.loc[state][0]]

        # feedback from the environment
        immediate_reward = self.reward_function.loc[state, action]

        # discounting
        discount_factor = self.discount_factor * discount_factor

        # assume transition probability alwarys equals to 1
        potential_states = self.transition_function.loc[state, action]
        if immediate_reward == -1:  # all traps are terminal
            return immediate_reward
        elif immediate_reward == 10:
            return immediate_reward
        elif discount_factor < 1e-5:
            return immediate_reward
        else:
            return immediate_reward + self.discount_factor * \
                self.cumulative_reward(potential_states, policy_function, discount_factor)

    def policy_iterate(self):
        value_function = self._initial_value_function(self.states_space)
        policy_function = self._initial_policy_function(self.states_space, self.actions_space)

        while True:
            # policy evaluate
            while True:
                value_function_ = copy.deepcopy(value_function)

                # updata value function for all state
                for n_state in range(len(self.states_space)):
                    state = self.states_space[n_state]

                    # update value function
                    value_function_.loc[state] = self.cumulative_reward(state, policy_function, self.discount_factor)
                if (value_function_ - value_function).abs().max()[0] < self.threshold:
                    break
                else:
                    value_function = copy.deepcopy(value_function_)

            # policy improvement
            q_value_function = self._create_q_value_function(value_function)
            policy_function_ = copy.deepcopy(policy_function)
            for n_state in range(len(self.states_space)):
                state = self.states_space[n_state]
                state_actions = q_value_function.loc[state]

                # update
                max_action_locate = np.where(state_actions.values == np.max(state_actions.values))[0]
                policy_function_.loc[state] = np.random.choice(max_action_locate)
            if (policy_function.values == policy_function_.values).all():
                break
            else:
                policy_function = copy.deepcopy(policy_function_)
        return policy_function
            

# -------------------------- application -------------------------
def main():
    gamma = 0.5
    threshold = 1e-5
    states_space = GMM.states_space
    actions_space = GMM.actions_space

    # Given model
    transition_function = GMM.transition_function
    reward_function = GMM.reward_function

    # initial model
    pie = Model_Based_RL(states_space, actions_space, transition_function,
                                reward_function, gamma, threshold)
    optimal_policy = pie.policy_iterate()

    # environment
    env = Maze()

    # initial observation
    state = env.reset()

    # simulation
    while True:
        action_ = optimal_policy.loc[str(state)].values.item()
        state_, immediate_reward, done = env.step(action_)
        action, state = action_, state_
        print(action,state)
        
        # fresh env
        env.render()
        time.sleep(2)
        if done: break


if __name__ == '__main__':
    main()
    print('Game over!')
