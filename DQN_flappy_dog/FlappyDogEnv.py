"""
Introduction:
    flappy dog as DQN environment
Author:
    Chend
Modify:
    2020-04-18
"""


# Dependences ----------------------------------------------------------------------------
import pygame
import random
import os
import time
import neat
import pickle
import numpy as np


# Initial
pygame.font.init()
pygame.display.set_caption("Flappy Dog")
WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730  # base
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

# Load images
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha())
BG_IMG = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
DOG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird1.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())


class Dog():
    """ Dog class representing the flappy dog """
    max_rotation = 25
    dog = DOG_IMG
    rotation_velocity = 20

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt_angle = 0
        self.tick_count = 0
        self.velocity = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.dog

    def jump(self):
        self.velocity = -10  # up is negative, down is positive
        self.tick_count = 0  # like time t
        self.height = self.y  # height of object

    def move(self):
        """ make the object move """
        self.tick_count += 1

        # for downward acceleration, (x = v*t + 0.5*a*t^2, here a = 3)
        delta_y = self.velocity * (self.tick_count) + 0.5 * 3 *(self.tick_count)**2

        # terminal velocity, make sure object won't move down too fast
        if delta_y >= 16:  
            delta_y = (delta_y/abs(delta_y)) * 16

        # if move up, make the object move up a little more
        if delta_y < 0:
            delta_y -= 2

        self.y = self.y + delta_y

        if delta_y < 0 or self.y < self.height + 50:  # tilt up the object
            if self.tilt_angle < self.max_rotation:
                self.tilt_angle = self.max_rotation
        else:  # tilt down
            if self.tilt_angle > -90:
                self.tilt_angle -= self.rotation_velocity

    def draw(self, win):
        # tilt the dog
        rotated_image = pygame.transform.rotate(self.dog, self.tilt_angle)
        new_rect = rotated_image.get_rect(center=self.dog.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        """ gets the mask for the current image of the dog """
        return pygame.mask.from_surface(self.img)


class Pipe():
    GAP = 200  # distance between pipes
    VEL = 5  # velocity of pipes

    def __init__(self, x):
        self.x = x
        self.height = 0

        # where the top and bottom of the pipe is
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.passed = False
        self.set_height()

    def set_height(self):
        """ set the height of the pipe randomly, from the top of the screen """
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        """ move pipe based on velocity """
        self.x -= self.VEL

    def draw(self, win):
        """ draw both the top and bottom of the pipe """
        # draw top
        win.blit(self.PIPE_TOP, (self.x, self.top))
        # draw bottom
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))


    def collide(self, dog, win):
        """ returns if target object is colliding with the pipe """
        bird_mask = dog.get_mask()
        top_pipe_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_pipe_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - dog.x, self.top - round(dog.y))
        bottom_offset = (self.x - dog.x, self.bottom - round(dog.y))

        # bottom point, if not collide return None
        b_point = bird_mask.overlap(bottom_pipe_mask, bottom_offset)

        # top point if not collide return None
        t_point = bird_mask.overlap(top_pipe_mask,top_offset)

        if b_point or t_point:  # if collide return True
            return True

        return False


class Base():
    """ Represnts the moving floor of the game """
    BASE_VELOCITY = 5  # velocity of the base image
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y

        # initial two base images
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        """ move floor so it looks like its scrolling """
        self.x1 -= self.BASE_VELOCITY
        self.x2 -= self.BASE_VELOCITY
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        """ Draw the floor. This is two images that move together. """
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, dogs, pipes, base, score, episode_n, pipe_idx):
    """
    draws the windows for the main game loop
    :param win: pygame window surface
    :param dog: a Dog object
    :param pipes: List of pipes
    :param score: score of the game (int)
    :param episode_n: current generation
    :param pipe_idx: index of closest pipe
    :return: None
    """
    if episode_n == 0:
        episode_n = 1
    win.blit(BG_IMG, (0,0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    for dog in dogs:
        # draw lines from dog to pipe
        if DRAW_LINES:
            try:
                pygame.draw.line(win, (255, 0, 0), 
                                (dog.x+dog.img.get_width()/2, dog.y + dog.img.get_height()/2), 
                                (pipes[pipe_idx].x + pipes[pipe_idx].PIPE_TOP.get_width()/2, pipes[pipe_idx].height), 
                                5)
                pygame.draw.line(win, (255, 0, 0), 
                                (dog.x+dog.img.get_width()/2, dog.y + dog.img.get_height()/2), 
                                (pipes[pipe_idx].x + pipes[pipe_idx].PIPE_BOTTOM.get_width()/2, pipes[pipe_idx].bottom), 
                                5)
            except:
                pass

        # draw dog
        dog.draw(win)

    # draw score
    score_label = STAT_FONT.render("Score: " + str(score), 1, (0, 0, 0))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))

    # draw generations
    score_label = STAT_FONT.render("Epiode: " + str(episode_n-1), 1, (0, 0, 0))
    win.blit(score_label, (10, 10))

    # draw alive objects
    score_label = STAT_FONT.render("Alive: " + str(len(dogs)), 1, (0, 0, 0))
    win.blit(score_label, (10, 50))

    pygame.display.update()


class flappy_environment():
    def __init__(self, mode='RL'):
        self.n_actions = 2 # jump or not
        self.n_features = 3  # length of state
        self.episode_n = 0
        self.mode = mode

    def reset(self):
        ''' return initial state  '''
        self.win = WIN
        self.episode_n += 1
        self.dogs = [Dog(230, 350) for i in range(1)]
        # self.dogs.append()
        self.floor = FLOOR
        self.base = Base(FLOOR)
        self.pipes = [Pipe(700)]
        self.score = 0
        self.clock = pygame.time.Clock()

        state = np.zeros((len(self.dogs), self.n_features))
        for i, dog in enumerate(self.dogs):
            state[i] = (self.dogs[0].y,
                        # self.pipes[0].height,
                        # self.pipes[0].bottom,
                        abs(self.dogs[0].y - self.pipes[0].height), 
                        abs(self.dogs[0].y - self.pipes[0].bottom))

        return state

    def render(self):
        ''' fresh env '''
        # draw current all objects
        draw_window(self.win, self.dogs, self.pipes, self.base, 
                    self.score, self.episode_n, self.pipe_idx)

    def step(self, action):
        # action size = [len(dogs), 1]
        
        # initial immediate reward
        self.reward = np.zeros((len(self.dogs), 1))

        self.clock.tick(30)
        for event in pygame.event.get():
            if self.mode == 'keyboard':
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        action[:, 0] = 1
                    else:
                        action[:, 0] = 0
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.pipe_idx = 0
        if len(self.dogs) > 0:
            # whether to use the first or second pipe on the screen for RL input
            if len(self.pipes) > 1 and self.dogs[0].x > self.pipes[0].x + \
                self.pipes[0].PIPE_TOP.get_width():
                self.pipe_idx = 1 

        # move dog
        state_ = np.zeros((len(self.dogs), self.n_features))
        for i, dog in enumerate(self.dogs):
            # if dog.y > self.pipes[self.pipe_idx].height and \
            #     dog.y <  self.pipes[self.pipe_idx].bottom:
            #     self.reward[i, 0] += 0.1
            # else:
            #     self.reward[i, 0] -= 0.2
            self.reward += 0.1

            dog.move()
            if action[i, 0] >= 0.5:
                dog.jump()
            state_[i] = (dog.y, 
                    #   self.pipes[self.pipe_idx].height,
                    #   self.pipes[self.pipe_idx].bottom,
                      abs(dog.y - self.pipes[self.pipe_idx].height), 
                      abs(dog.y - self.pipes[self.pipe_idx].bottom))

        # move base
        self.base.move()

        # move pipe
        remove_pipe = []
        add_pipe = False
        for pipe in self.pipes:
            pipe.move()

            # check for collision
            for i, dog in enumerate(self.dogs):
                if pipe.collide(dog, self.win):
                    self.reward[i, 0] -= 20
                    self.dogs.pop(self.dogs.index(dog))

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                remove_pipe.append(pipe)
            if len(self.dogs) > 0 and not pipe.passed and pipe.x < self.dogs[0].x:
                pipe.passed = True
                add_pipe = True
        if add_pipe:
            self.score += 1
            self.reward[i, 0] += 30
            self.pipes.append(Pipe(WIN_WIDTH))

        for r in remove_pipe: # remove passed pipe
            self.pipes.remove(r)
        
        for i, dog in enumerate(self.dogs):
            if dog.y + dog.img.get_height() -10 >= self.floor or dog.y < -50:
                self.reward[i, 0] -= 15
                self.dogs.pop(self.dogs.index(dog))

        done = False if len(self.dogs) > 0 else True

        return state_, self.reward, done


if __name__ == "__main__":

    mode = ['keyboard', 'RL']
    env = flappy_environment(mode[0])
    
    for n in range(100):
        env.reset()
        step = 0
        while True:
            # action = np.array([[1]]) if step % 10 == 0 else np.array([[0]])
            action = np.array([[0]])
            s_, r, done = env.step(action)
            env.render()
            # print(s_, r, done, action)
            step += 1
            if done:
                break

