import numpy as np
from collections import defaultdict, deque
import gym
import matplotlib.pyplot as plt
import sys
from plot_utils import plot_values
import random


def epsilon_greedy(Q, state, eps, nA):
    if random.random() > eps:
        return np.argmax(Q[state])
    else:
        return random.choice(np.arange(nA))

def update_Q_expsarsa(alpha, gamma, nA, eps, Q, state, action, reward, next_state=None):
    """Returns updated Q-value for the most recent experience."""
    current = Q[state][action]         # estimate in Q-table (for current state, action pair)
    # eps_greedy
    policy_s = np.ones(nA) * eps / nA  # current policy (for next state S')
    policy_s[np.argmax(Q[next_state])] = 1 - eps + (eps / nA) # greedy action

    # 求期望
    Qsa_next = np.dot(Q[next_state], policy_s)         # get value of state at next time step
    target = reward + (gamma * Qsa_next)               # construct target
    new_value = current + (alpha * (target - current)) # get updated value
    return new_value

def expected_sarsa(env, num_episodes, alpha, gamma=1.0, plot_every=100):

    # 初始化Q
    nA = env.action_space.n
    Q = defaultdict(lambda : np.zeros(nA))

    # 查看进度
    tmp_scores = deque(maxlen=plot_every)  # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episodes)  # average scores over every plot_every episodes

    # 对每一个episode
    for i_episode in range(1, num_episodes + 1):

        # 查看进度
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        score = 0

        # 获取初始状态值
        state = env.reset()

        # 设置递减的epsilon
        eps = 0.005

        # 对该幕下的每一个状态
        while True:
            # 为当前状态选择动作
            action = epsilon_greedy(Q, state, eps, nA)

            # 获得四元组
            next_state, reward, done, info = env.step(action)

            score += reward

            # 更新Q值
            Q[state][action] = update_Q_expsarsa(alpha, gamma, nA, eps, Q, state, action, reward, next_state)

            # 更新状态
            state = next_state

            if done:
                tmp_scores.append(score)    # append score
                break

        if (i_episode % plot_every == 0):
            avg_scores.append(np.mean(tmp_scores))

    # plot performance
    plt.plot(np.linspace(0, num_episodes, len(avg_scores), endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))
    return Q

env = gym.make('CliffWalking-v0')


Q_expsarsa = expected_sarsa(env, 5000, 1)

# print the estimated optimal policy
policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4,12)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_expsarsa)

# plot the estimated optimal state-value function
plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])