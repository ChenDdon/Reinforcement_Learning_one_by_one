'''
Introduction:
    Given model and conditions for RL MAZE
Author:
    Chend
Modify:
    2020-02-20
'''


# -------------------------- Dependence ---------------------------------------------
import numpy as np
import pandas as pd


# --------------------- Create_Reward_Function --------------------------------------
def Create_Reward_Function(states_space, actions_space):
    # reward map 1, row: x, column: y
    maze_reward_map = np.array([[  0,  -1, 0.2, 0.1, 0.1,   0],
                                [0.1,  -1, 0.3,  -1,  -1,   0],
                                [0.3, 0.3, 0.4, 0.3, 0.2, 0.1],
                                [ -1, 0.4, 0.5,  -1,   0,  -1],
                                [  0,  -1, 0.6,  -1,   0,   0],
                                [  0,   0,  10,  -1,   0., -1]])
    
    # all states in maze map
    maze_map = np.array([['[{:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(i*80, j*80, (i+1)*80, (j+1)*80)
                 for i in range(6)] for j in range(6)])
    
    reward_dict = {}
    for i in states_space:
        # current state location
        location = np.where(maze_map==i)

        # up
        if location[0][0] - 1 < 0:
            up_reward = maze_reward_map[location] - 0.5  # 0.5 is punishment
        else:
            up_reward = maze_reward_map[location[0][0] - 1, location[1][0]]
        # down
        if location[0][0] + 1 > 5:
            down_reward = maze_reward_map[location] - 0.5
        else:
            down_reward = maze_reward_map[location[0][0] + 1, location[1][0]]
        # left
        if location[1][0] - 1 < 0:
            left_reward = maze_reward_map[location] - 0.5
        else:
            left_reward = maze_reward_map[location[0][0], location[1][0] - 1]
        # right
        if location[1][0] + 1 > 5:
            right_reward = maze_reward_map[location] - 0.5
        else:
            right_reward = maze_reward_map[location[0][0], location[1][0] + 1]

        reward_dict[i] = [up_reward, down_reward, left_reward, right_reward] 

    reward_function = pd.DataFrame.from_dict(reward_dict,
                                            orient='index',
                                            dtype=float,
                                            columns=actions_space)
    return reward_function


# --------------------- Create_Transition_Function ----------------------------------
def Create_Transition_Function(states_space, actions_space):
    ''' Assume all transfer probabilities to 1 in this example'''

    # Transition result is deterministic for all action
    probability = 1
    maze_map = np.array([['[{:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(i*80, j*80, (i+1)*80, (j+1)*80)
                 for i in range(6)] for j in range(6)])
    # maze_map shpae: [6, 6]
    
    transition_dict = {}
    for i in states_space:
        # state location
        location = np.where(maze_map == i)

        # up
        if location[0][0] - 1 < 0:
            up_location = [0, location[1][0]]
        else:
            up_location = [location[0][0] - 1, location[1][0]]
        # down
        if location[0][0] + 1 > 5:
            down_location = [3, location[1][0]]
        else:
            down_location = [location[0][0] + 1, location[1][0]]
        # left
        if location[1][0] - 1 < 0:
            left_location = [location[0][0], 0]
        else:
            left_location = [location[0][0], location[1][0] - 1]
        # right
        if location[1][0] + 1 > 5:
            right_location = [location[0][0], 5]
        else:
            right_location = [location[0][0], location[1][0] + 1]

        transition_dict[i] = [maze_map[up_location[0], up_location[1]],
                            maze_map[down_location[0], down_location[1]],
                            maze_map[left_location[0], left_location[1]],
                            maze_map[right_location[0], right_location[1]]]

    transition_function = pd.DataFrame.from_dict(transition_dict,
                                        orient='index',
                                        dtype=int,
                                        columns=actions_space)

    return transition_function


# ---------------------------- initial ----------------------------------------------
actions_space = ['up', 'down', 'left', 'right']
states_space = ['[{:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(i*80, j*80, (i+1)*80, (j+1)*80)
                 for i in range(6) for j in range(6)]
reward_function = Create_Reward_Function(states_space, actions_space)
transition_function = Create_Transition_Function(states_space, actions_space)
