'''
Introduction:
    MAZE environment for RF
Author:
    Chend
Modify:
    2020-02-20
'''


# -------------------------- Dependence ------------------------------
import numpy as np
import time
import sys
import tkinter as tk
# import Tkinter as tk  # for python 2.x


# -------------------------- Maze map scale --------------------------
unit = 80   # pixels for one unit
maze_height = 6  # grid height
maze_width = 6  # grid width


# -------------------------- Build maze environment ------------------
class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.title('MAZE ({:d} x {:d})'.format(maze_height, maze_height))
        self.geometry('{:d}x{:d}'.format(maze_width*unit, maze_height*unit))
        self.build_maze()

    def build_maze(self):
        # create canvas
        self.canvas = tk.Canvas(self, bg='white',
                                height=maze_height*unit,
                                width=maze_width*unit)

        # create grids
        # vertical
        for c in range(0, maze_width * (unit+1), unit):
            x0, y0, x1, y1 = c, 0, c, maze_height * unit
            self.canvas.create_line(x0, y0, x1, y1)
        # horizontal
        for r in range(0, maze_height * unit, unit):
            x0, y0, x1, y1 = 0, r, maze_width * unit, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin unit center
        origin = np.array([unit//2, unit//2])

        # create red circle
        self.rect = self.canvas.create_oval(origin[0] - unit//2,
                                            origin[1] - unit//2,
                                            origin[0] + unit//2,
                                            origin[1] + unit//2,
                                            fill='red')

        # create terminals, defalut: 1 terminal
        terminal_state = np.array([[2, 5]])
        self.all_terminals = []
        for t in range(np.shape(terminal_state)[0]):
            terminal_center = origin + terminal_state[t, :] * unit
            terminal = self.canvas.create_rectangle(terminal_center[0] - unit//2,
                                                    terminal_center[1] - unit//2,
                                                    terminal_center[0] + unit//2,
                                                    terminal_center[1] + unit//2,
                                                    fill='yellow')
            setattr(self, 'terminal{:d}'.format(t), terminal)
            self.all_terminals.append(terminal)

        # create traps
        trap_state = np.array([[0, 3], [1, 0], [1, 1], [1, 4],
                               [3, 1], [3, 3], [3, 4], [3, 5],
                               [3, 7], [4, 1], [5, 3], [5, 5]])
        self.all_traps = []
        for tp in range(np.shape(trap_state)[0]):
            trap_center = origin + trap_state[tp, :] * unit
            trap = self.canvas.create_rectangle(trap_center[0] - unit//2,
                                                trap_center[1] - unit//2,
                                                trap_center[0] + unit//2,
                                                trap_center[1] + unit//2,
                                                fill='black')
            setattr(self, 'trap{:d}'.format(tp), trap)
            self.all_traps.append(trap)

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)

        # delete original object
        self.canvas.delete(self.rect)
        origin = np.array([unit//2, unit//2])
        self.rect = self.canvas.create_oval(origin[0] - unit//2,
                                            origin[1] - unit//2,
                                            origin[0] + unit//2,
                                            origin[1] + unit//2,
                                            fill='red')
        
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        # red rectangle's state
        state = self.canvas.coords(self.rect)

        # how to move, two directions
        base_action = np.array([0, 0])
        if action == 0:   # up
            if state[1] > unit:
                base_action[1] -= unit
        elif action == 1:   # down
            if state[1] < (maze_height - 1) * unit:
                base_action[1] += unit
        elif action == 2:   # left
            if state[0] > unit:
                base_action[0] -= unit
        elif action == 3:   # right
            if state[0] < (maze_width - 1) * unit:
                base_action[0] += unit
        
        # move agent
        self.canvas.move(self.rect, base_action[0], base_action[1])  

        # next state
        state_ = self.canvas.coords(self.rect)

        # reward function
        if state_ in [self.canvas.coords(te) for te in self.all_terminals]:
            reward = 10
            done = True
            state_ = 'terminal'
        elif state_ in [self.canvas.coords(tt) for tt in self.all_traps]:
            reward = -1
            done = True
            state_ = 'terminal'
        else:
            reward = 0
            done = False

        return state_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


if __name__ == '__main__':
    env = Maze()
    env.mainloop()