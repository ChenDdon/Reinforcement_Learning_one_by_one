"""
Introduction:
    Reinforcement Learning - K-armed bandit
Author:
    Chend(cddonfx@gmail.com)
Modify:
    2020-01-13
"""


# ----------------------------- dependences -----------------------------------
import numpy as np  # 1.15.5
import matplotlib.pyplot as plt  # 3.1.1


# ------------------------ bandit game environment ---------------------------
class Bandit(object):
    """
    Introduction:
        Build k-armed bandit game environment
    """
    def __init__(self):
        # the probability of each arm receiving a reward, default k=10
        self.distribution = [0.4, 0.2, 0.1, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3]
        self.k_arm = len(self.distribution)

    def reward_func(self, choose_k):
        """
        Introduction:
            reward function for k_th arm

        Arguments:
            choose_k [type=int] -- the index of selected arm

        Returns:
            reward [type=int(default)] -- the reward of kth arm
        """
        prob = [self.distribution[choose_k], 1-self.distribution[choose_k]]
        reward = np.random.choice([1, 0], p=prob)

        return reward


# --------------------------- how to play -------------------------------------
class Simulate_Game(object):
    def __init__(self, k_arm, reward_func, step_size, n_episode, method=None):
        self.k_arm = k_arm
        self.reward_func = reward_func
        self.n_episode = n_episode

        # step size for each episode
        self.step_size = step_size

        # choose algorithm to simulate the game
        self.method = method

        # arms indices
        self.indices = np.arange(self.k_arm)

    def reset_bandit(self):
        # initial q_value function
        self.q_estimate = np.zeros((self.k_arm, ), dtype=np.float32)

        # count selected times of each arm_k
        self.count_arm_k = np.zeros((self.k_arm, ), dtype=np.int32)

    def e_greedy_choose(self, e_greedy):
        # e-greedy algorithm
        if np.random.rand() < e_greedy:
            choose_k = np.random.choice(self.indices)
        else:
            choose_k = np.argmax(self.q_estimate)
        return choose_k

    def softmax_choose(self, tau_temperature):
        # Boltzmann distribution of all k arms
        exp_fact = self.q_estimate / tau_temperature
        if np.max(exp_fact) > 63:  # in case overflow encountered in exp
            exp_fact[np.argmax(exp_fact)] = 63
        exp_estimate = np.exp(exp_fact)
        boltzmann_distribution = exp_estimate / np.sum(exp_estimate)

        # choose k_th arm based on boltzmann distribution
        choose_k = np.random.choice(self.indices, p=boltzmann_distribution)

        return choose_k

    def simulate_loop(self, parameter):
        # how to choose an action
        if self.method == 'e_greedy':
            action_choose = self.e_greedy_choose
        elif self.method == 'softmax':
            action_choose = self.softmax_choose
        else:
            action_choose = self.e_greedy_choose

        rewards = np.zeros((self.n_episode, self.step_size), dtype=np.float32)
        for n in range(self.n_episode):

            # reset bandit and initial
            self.reset_bandit()
            cumulative_reward = 0

            # step loop
            for t in range(self.step_size):
                # choose k_th arm
                choose_k = action_choose(parameter)

                # obtain immediate reward
                immediate_value = self.reward_func(choose_k)

                # record average reward
                cumulative_reward += immediate_value
                rewards[n, t] = cumulative_reward / (t+1)

                # update q estimate function average reward
                self.count_arm_k[choose_k] += 1
                self.q_estimate[choose_k] += \
                    (immediate_value - self.q_estimate[choose_k]) / self.count_arm_k[choose_k]

        mean_reward = np.mean(rewards, axis=0)
        return mean_reward


# ------------------------- test and optimize ---------------------------------
def e_greedy_simulate_main():
    # load k_armed bandit
    bandit = Bandit()

    # simulation parameters
    k_arm = bandit.k_arm
    reward_func = bandit.reward_func
    step_size = 3000
    n_episode = 500

    # method
    algorithm = 'e_greedy'
    e_greedy = [0, 0.01, 0.1, 1]
    simulate_g = Simulate_Game(k_arm, reward_func, step_size, n_episode, method=algorithm)

    # main loop
    plt.style.use('ggplot')
    for i in range(len(e_greedy)):
        mean_reward = simulate_g.simulate_loop(e_greedy[i])

        # pictures
        plt.plot(mean_reward, '-', label='epsilon ' + str(e_greedy[i]))

    # save fig
    plt.title('$\epsilon$-greedy algorithm')
    plt.xlabel('T-steps')
    plt.ylabel('T-step Average Reward')
    # plt.ylim([0, 1])
    plt.legend()
    plt.savefig('./images/e_greedy_simulate.png')
    plt.close()


def softmax_simulate_main():
    # load k_armed bandit
    bandit = Bandit()

    # simulation parameters
    k_arm = bandit.k_arm
    reward_func = bandit.reward_func
    step_size = 3000
    n_episode = 500

    # method
    algorithm = 'softmax'
    tau = [0.01, 0.1, 0.2, 0.4, 0.5]
    simulate_g = Simulate_Game(k_arm, reward_func, step_size, n_episode, method=algorithm)

    # main loop
    plt.style.use('ggplot')
    for i in range(len(tau)):
        mean_reward = simulate_g.simulate_loop(tau[i])

        # pictures
        plt.plot(mean_reward, '-', label='epsilon ' + str(tau[i]))

    # save fig
    plt.title('softmax algorithm')
    plt.xlabel('T-steps')
    plt.ylabel('T-step Average Reward')
    # plt.ylim([0, 1])
    plt.legend()
    plt.savefig('./images/softmax_simulate.png')
    plt.close()


def optimize_parameters():
    # load k_armed bandit
    bandit = Bandit()

    # simulation parameters
    k_arm = bandit.k_arm
    reward_func = bandit.reward_func
    step_size = 3000
    n_episode = 500

    # e_greedy
    algorithm = 'e_greedy'
    e_greedy = np.arange(0.01, 1.01, 0.01)
    simulate_g = Simulate_Game(k_arm, reward_func, step_size, n_episode, method=algorithm)

    expect_reward_e_greedy = np.zeros((len(e_greedy), ), dtype=np.float32)
    for i in range(len(e_greedy)):
        expect_reward_e_greedy[i] = simulate_g.simulate_loop(e_greedy[i])[-1]

    # softmax
    algorithm = 'softmax'
    tau = np.arange(0.01, 1.01, 0.01)
    simulate_g = Simulate_Game(k_arm, reward_func, step_size, n_episode, method=algorithm)

    expect_reward_softmax = np.zeros((len(tau), ), dtype=np.float32)
    for i in range(len(tau)):
        expect_reward_softmax[i] = simulate_g.simulate_loop(tau[i])[-1]

    # show result
    plt.style.use('ggplot')
    plt.plot(e_greedy, expect_reward_e_greedy, '.-', label='$\epsilon$-greedy')
    plt.plot(tau, expect_reward_softmax, '.-', label='Softmax')
    plt.title('parameter optimization')
    plt.ylabel('Expect average reward')
    plt.xlabel('Parameters')
    plt.legend()
    plt.savefig('./images/optimize_parameters.png')
    plt.close()


if __name__ == "__main__":
    np.random.seed(1234)
    e_greedy_simulate_main()
    softmax_simulate_main()
    optimize_parameters()
