"""
Introduction:
    runing step of the example Maze
    algorithm is sarsa
Author:
    Chend
Modify:
    2019-11-6
"""


# Dependence ---------------------------------------------------------------------------
from FlappyDogEnv import flappy_environment
from RL_DQN import RL_DQN
import numpy as np
import torch


# ----------------------------------- simulation ---------------------------------------
def game_simulation(max_episode, env, RL_algorithm):

    # record steps for each episode
    step = 0

    for episode in range(max_episode):
        print(f"Num.: {episode} start.")

        # initial state
        state = env.reset()

        while True:
            
            # RL agent choose action based on state, float [-1, 1]
            action = RL_algorithm.choose_action(state)
            
            # RL_algorithm take action and get next state and reward
            state_, reward, done = env.step(action)

            # fresh env
            env.render()

            # store memory <s,a,r,s'> into the memary D
            RL_algorithm.store_transition(state, action, reward, state_)

            # RL_algorithm learn from transition memary D
            if (step > 200) and (step % 5 == 0):
                RL_algorithm.update_q_function()
            
            # swap state and action
            state = state_

            if env.score > 100:
                torch.save(RL_algorithm.q_func, './q_func.model')
                torch.save(RL_algorithm.q_target_func, './q_target_func.model')
                break

            # break while loop when end of this episode
            if done:
                break
        
            step += 1


# -------------------------------------- main ------------------------------------------
def main():
    max_episode = 5000
    env = flappy_environment()
    RL_algorithm = RL_DQN(
        env.n_actions,
        env.n_features,
        alpha=0.01,
        gamma=0.9,
        epsilon=1,
        replace_target_iter=200,
    )
    game_simulation(max_episode, env, RL_algorithm)


if __name__ == "__main__":
    main()
    print("End!")