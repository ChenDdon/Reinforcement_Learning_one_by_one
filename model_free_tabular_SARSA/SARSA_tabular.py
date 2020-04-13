"""
Introduction:
    Model free Reinforcement for Maze
    Algorithm is tabular sarsa
Author:
    chend
Modify:
    2020-03-30
"""


# ---------------------------------------- Dependences -----------------------------------
from MazeEnv import Maze
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Tabular_SARSA():
    def __init__(self, action_space, alpha=0.01, gamma=0.9, e_greedy=0.9):
        self.action_space = action_space
        self.learning_rate = alpha
        self.discount_factor = gamma
        self.e_greedy = e_greedy

        # initial Q value
        self.q_table = pd.DataFrame(columns=self.action_space, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0]*len(self.action_space),
                index=self.q_table.columns, name=state))

    def choose_action(self, state):
        '''e-greedy method'''
        self.check_state_exist(state)

        # action selection
        if np.random.uniform() < self.e_greedy:

            # choose best action
            state_action = self.q_table.loc[state, :]

            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.action_space)

        return action

    def updata_q_table(self, state, action, reward, state_, action_):
        '''sarsa method''' 
        self.check_state_exist(state_)
        q_predict = self.q_table.loc[state, action]
        
        if state_ != 'terminal':
            q_target = reward + self.discount_factor * self.q_table.loc[state_, action_]
        else:
            # next state is terminal
            q_target = reward
        
        # update Q function
        self.q_table.loc[state, action] += self.learning_rate * (q_target - q_predict)


# ----------------------------------- simulation -------------------------------------------
def maze_simulation(max_episode, env, RL_algorithm):
    # record total rewards
    total_reward = np.zeros((max_episode, ))
    for episode in range(max_episode):
        print(f"Num.: {episode} start.")

        # initial state
        state = env.reset()

        # RL choose action based on state
        action = RL_algorithm.choose_action(str(state))
        while True:

            # fresh env
            env.render()

            # RL take action and get next state and reward
            state_, reward, done = env.step(action)

            action_ = RL_algorithm.choose_action(str(state_))

            # RL learn from this transition <s,a,r,s',a'>
            RL_algorithm.updata_q_table(str(state), action, reward, str(state_), action_)

            #  swap and action
            action = action_

            # swap state
            state = state_

            total_reward[episode] += reward

            if done:
                break

    # end of game
    env.destroy()

    return total_reward


# -------------------------------------- main ------------------------------------------
def main():
    max_episode = 100
    env = Maze()
    action_space = list(range(env.n_actions))
    # sarsa
    RL_algorithm = Tabular_SARSA(action_space, alpha=0.01, gamma=0.9, e_greedy=0.9)

    total_reward = maze_simulation(max_episode, env, RL_algorithm)
    
    # maintain the env
    env.mainloop()

    # plot total rewards
    plt.plot(total_reward, 'green', lw=1)
    plt.fill_between(np.arange(max_episode), total_reward, 
                    where=(total_reward <= 0),
                    color='green', alpha=0.3)
    plt.fill_between(np.arange(max_episode), total_reward, 
                    where=(total_reward > 0),
                    color='red', alpha=0.3)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('SARSA Reward')
    plt.savefig('MAZE_SARSA_reward.png', dpi = 300)
    plt.close()

if __name__ == "__main__":
    main()
    print('Game over')
