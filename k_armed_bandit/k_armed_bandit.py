"""
Introduction:
    Reinforcement Learning -- K-armed bandit
Author:
    Chend(cddonfx@gmail.com)
Modify:
    2020-03-22
"""


# ----------------------------- Dependences -----------------------------------
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- Reproducibility -------------------------------
np.random.seed(1212)


# ------------------------ Bandit game environment ----------------------------
class Bandit(object):
    """
    Introduction:
        Build k-armed bandit game environment
    """
    def __init__(self):
        # probability of each arm receiving a reward
        self.distribution = [0.4, 0.1, 0.1, 0.3, 0.4, 0.1, 0.2, 0.4, 0.2, 0.3]

        # default 10 arms, k=10
        self.k_arm = len(self.distribution)

    def reward_func(self, choose_k):
        """
        Introduction:
            reward function for k_th arm
        Arguments:
            choose_k [type=int] -- the index of selected arm
        Returns:
            reward [int(default)] -- the reward of kth arm
        """
        prob = [self.distribution[choose_k], 1 - self.distribution[choose_k]]
        reward = np.random.choice([1.5, -1], p=prob)

        return reward


# ------------------------ How to choose action ------------------------------
class Action_Choose(object):
    def __init__(self, k_arm=10, parameter=0.03):
        # arm's indices
        self.indices = np.arange(k_arm)
        
        # initial action parameter
        self.parameter = parameter

    def e_greedy_choose(self, q_estimate):
        # e-greedy algorithm

        e_greedy = self.parameter
        if np.random.rand() < e_greedy:
            choose_k = np.random.choice(self.indices)
        else:
            choose_k = np.argmax(q_estimate)
        return choose_k
    
    def softmax_choose(self, q_estimate):
        # Boltzmann distribution of all k arms

        tau_temperature = self.parameter

        # in case overflow encountered in exp(64)
        exp_factor = q_estimate / tau_temperature
        if np.max(exp_factor) > 63:
            exp_factor[np.argmax(exp_factor)] = 63

        exp_estimate = np.exp(exp_factor)
        boltzmann_distribution = exp_estimate / np.sum(exp_estimate)

        # choose k_th arm based on boltzmann distribution
        choose_k = np.random.choice(self.indices, p=boltzmann_distribution)
        return choose_k


# --------------------------- How to play -------------------------------------
class Simulate_Game(object):
    def __init__(self, k_arm, reward_func, step_size, method=None):
        self.k_arm = k_arm
        self.reward_func = reward_func

        # step size for each episode
        self.step_size = step_size

        # choose algorithm to simulate the game
        self.action_choose = self.build_method(method)

        # arms indices
        self.indices = np.arange(self.k_arm)

    def reset_bandit(self):
        # initial q_value function
        self.q_estimate = np.zeros((self.k_arm, ), dtype=np.float32)

        # count selected times of each arm_k
        self.count_arm_k = np.zeros((self.k_arm, ), dtype=np.int32)

    def build_method(self, method):
        if method == None:
            return self._default_action_method
        else:
            return method

    def _default_action_method(self, parameter=None, q_estimate=None):
        # default method to choose action in each step
        # choose action randomly
        choose_k = np.random.choice(self.indices)
        return choose_k

    def simulate_loop(self):
        # rewards = np.zeros((self.n_episode, self.step_size), dtype=np.float32)
        # for n in range(self.n_episode):

        # reset bandit and initial
        self.reset_bandit()
        cumulative_reward = 0
        cumulative_reward_all = np.zeros((self.step_size, ), dtype=float)
        average_reward = np.zeros((self.step_size, ), dtype=float)

        # step loop
        for t in range(self.step_size):
            # choose k_th arm
            choose_k = self.action_choose(self.q_estimate)

            # obtain immediate reward
            immediate_value = self.reward_func(choose_k)

            # record average reward
            cumulative_reward += immediate_value
            cumulative_reward_all[t] = cumulative_reward
            average_reward[t] = cumulative_reward / (t + 1)

            # update q estimate function average reward
            self.count_arm_k[choose_k] += 1
            self.q_estimate[choose_k] += \
                (immediate_value - self.q_estimate[choose_k]) / self.count_arm_k[choose_k]
            
        return cumulative_reward_all, average_reward


# ------------------------- Test and Optimize ---------------------------------
def e_greedy_simulate_main():
    # load k_armed bandit
    bandit = Bandit()
    k_arm = bandit.k_arm
    reward_func = bandit.reward_func

    # action chosen method
    e_greedy = 0.02
    action_choose = Action_Choose(k_arm, parameter=e_greedy)
    algorithm = action_choose.e_greedy_choose

    # simulation parameters
    step_size = 1000
    simulate_g = Simulate_Game(k_arm, reward_func, step_size, method=algorithm)

    # main loop
    n_episode = 100
    n_cumulative_reward = np.zeros((n_episode, step_size), dtype=float)
    n_mean_reward = np.zeros((n_episode, step_size), dtype=float)
    for i in range(n_episode):
        cumulative_reward, mean_reward = simulate_g.simulate_loop()
        n_cumulative_reward[i, :] = cumulative_reward
        n_mean_reward[i, :] = mean_reward

    # pictures
    plt.style.use('ggplot')
    plt.tight_layout()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    p1 = ax1.plot(np.mean(n_mean_reward, axis=0), '.', 
                  color='r', label=f'average')
    ax1.set_title(f'$\epsilon$-greedy Algorithm ($\epsilon$={action_choose.parameter})')
    ax1.set_ylabel('T-step Average Reward')
    ax1.set_xlabel('T-steps')
    ax2 = ax1.twinx()
    p2 = ax2.plot(np.mean(n_cumulative_reward, axis=0), '.', 
                  color='b', label='cumulative')
    ax2.set_ylabel('Cumulative Reward')
    
    # legend
    p_s = p1 + p2
    labs = [p.get_label() for p in p_s]
    ax2.legend(p_s, labs, loc='center right')

    # save picture
    plt.savefig('./images/e_greedy_simulate.png', dpi=300)
    plt.close()


def softmax_simulate_main():
    # load k_armed bandit
    bandit = Bandit()
    k_arm = bandit.k_arm
    reward_func = bandit.reward_func

    # action chosen method    
    tau = 0.02
    action_choose = Action_Choose(k_arm, parameter=tau)
    algorithm = action_choose.softmax_choose

    # simulation parameters
    step_size = 1000
    simulate_g = Simulate_Game(k_arm, reward_func, step_size, method=algorithm)

    # main loop
    n_episode = 100
    n_cumulative_reward = np.zeros((n_episode, step_size), dtype=float)
    n_mean_reward = np.zeros((n_episode, step_size), dtype=float)
    for i in range(n_episode):
        cumulative_reward, mean_reward = simulate_g.simulate_loop()
        n_cumulative_reward[i, :] = cumulative_reward
        n_mean_reward[i, :] = mean_reward

    # pictures
    plt.style.use('ggplot')
    plt.tight_layout()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    p1 = ax1.plot(np.mean(n_mean_reward, axis=0), '.', 
                  color='r', label=f'average')
    ax1.set_title(f'Softmax Algorithm ($\\tau$={action_choose.parameter})')
    ax1.set_ylabel('T-step Average Reward')
    ax1.set_xlabel('T-steps')
    ax2 = ax1.twinx()
    p2 = ax2.plot(np.mean(n_cumulative_reward, axis=0), '.', 
                  color='b', label='cumulative')
    ax2.set_ylabel('Cumulative Reward')
    
    # legend
    p_s = p1 + p2
    labs = [p.get_label() for p in p_s]
    ax2.legend(p_s, labs, loc='center right')

    # save picture
    plt.savefig('./images/softmax_simulate.png', dpi=300)
    plt.close()


def optimize_parameters_main():
    # load k_armed bandit
    bandit = Bandit()
    k_arm = bandit.k_arm
    reward_func = bandit.reward_func

    # action chosen method
    action_choose = Action_Choose(k_arm)
    
    # simulation parameters
    step_size = 1000

    # e_greedy
    e_greedy = np.arange(0.01, 1.01, 0.05)
    algorithm = action_choose.e_greedy_choose
    simulate_g = Simulate_Game(k_arm, reward_func, step_size, method=algorithm)

    expect_reward_e_greedy = np.zeros((len(e_greedy), ), dtype=float)
    n_episode = 100
    for i in range(len(e_greedy)):
        action_choose.parameter = e_greedy[i]
        n_episode_reward = 0
        for j in range(n_episode):
            _, mean_reward = simulate_g.simulate_loop()
            n_episode_reward += mean_reward[-1]
        expect_reward_e_greedy[i] = n_episode_reward / n_episode

    # softmax
    tau = np.arange(0.01, 1.01, 0.05)
    algorithm = action_choose.softmax_choose
    simulate_g = Simulate_Game(k_arm, reward_func, step_size, method=algorithm)

    expect_reward_softmax = np.zeros((len(tau), ), dtype=float)
    n_episode = 100
    for i in range(len(tau)):
        action_choose.parameter = tau[i]
        n_episode_reward = 0
        for j in range(n_episode):
            _, mean_reward = simulate_g.simulate_loop()
            n_episode_reward += mean_reward[-1]
        expect_reward_softmax[i] = n_episode_reward / n_episode

    # save result
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.plot(e_greedy, expect_reward_e_greedy, '.-', label='$\epsilon$-greedy')
    plt.plot(tau, expect_reward_softmax, '.-', label='Softmax')
    plt.title('parameter optimization')
    plt.ylabel('Expect average reward')
    plt.xlabel('Parameters')
    plt.legend()
    plt.savefig('./images/optimize_parameters.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    e_greedy_simulate_main()
    # softmax_simulate_main()
    # optimize_parameters_main()
