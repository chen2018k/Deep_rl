import sys
import gym
import numpy as np
import random
import math
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from plot_utils import plot_values

env = gym.make('CliffWalking-v0')

# 初始化
V_opt = np.zeros((4,12))
V_opt[0][0:13] = -np.arange(3, 15)[::-1]
V_opt[1][0:13] = -np.arange(3, 15)[::-1] + 1
V_opt[2][0:13] = -np.arange(3, 15)[::-1] + 2
V_opt[3][0] = -13

# 查看设置的V的初始值
#plot_values(V_opt)


def epsilon_greedy(Q, state, nA, eps):
    """Selects epsilon-greedy action for supplied state.

    Params
    ======
        Q (dictionary): action-value function
        state (int): current state
        nA (int): number actions in the environment
        eps (float): epsilon
    """
    if random.random() > eps:  # select greedy action with probability epsilon
        return np.argmax(Q[state])
    else:  # otherwise, select an action randomly
        return random.choice(np.arange(nA))

# 理论上原理相同，为什么这个函数不可以工作
# def epsilon_greedy_test(Q,state,nA,epsilon):
#     policy_s = np.ones(nA) * epsilon / nA
#     best_a = np.argmax(Q[state])
#     Q[state] = policy_s
#     Q[state][best_a] = 1 - epsilon + (epsilon / nA)
#     action = np.random.choice(np.arange(nA), p=Q[state])
#     return action

def update_Q_sarsa(alpha, gamma, Q,state, action, reward, next_state=None, next_action=None):
    curr = Q[state][action]
    Q_next = Q[next_state][next_action] if next_state is not None else 0
    target = reward + gamma * Q_next
    new = curr + alpha * alpha * (target - curr)
    return new

def sarsa(env, num_episodes, alpha, gamma=1.0, plot_every=100):

    # 初始化Q
    nA = env.action_space.n  # number of actions
    Q = defaultdict(lambda: np.zeros(nA))  # initialize empty dictionary of arrays

    # 绘制图表
    # monitor performance
    tmp_scores = deque(maxlen=plot_every)  # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episodes)  # average scores over every plot_every episodes

    # 对每一个episode进行处理
    for i_episode in range(1, num_episodes + 1):

        #绘制图表
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # 初始化分数
        score = 0  # initialize score

        # 获得初始状态值S_0
        state = env.reset()  # start episode

        # 第二种设置逐渐衰减的eps的办法，第一种见MC/constant-alpha.py
        eps = 1.0 / i_episode  # set value of epsilon

        # 根据epsilon-greedy策略选择动作
        action = epsilon_greedy(Q, state, nA, eps)  # epsilon-greedy action selection

        # 对该episode中之后的每一个state
        while True:

            # 获得四元组
            next_state, reward, done, info = env.step(action)  # take action A, observe R, S'
            score += reward  # add reward to agent's score

            # 如果episode没有结束
            if not done:
                # 获取下一个状态要做的动作
                next_action = epsilon_greedy(Q, next_state, nA, eps)  # epsilon-greedy action

                # 已获得五元组，更新Q值
                Q[state][action] = update_Q_sarsa(alpha, gamma, Q, \
                                                  state, action, reward, next_state, next_action)
                # 更新动作、状态
                state = next_state  # S <- S'
                action = next_action  # A <- A'

            # 如果episode 已结束
            if done:
                Q[state][action] = update_Q_sarsa(alpha, gamma, Q, \
                                                  state, action, reward)
                # 记录本幕的得分
                tmp_scores.append(score)  # append score
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

# Q_sarsa = sarsa(env, 5000, .01)
#
# # print the estimated optimal policy
# policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
# print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
# print(policy_sarsa)
#
# # plot the estimated optimal state-value function
# V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
# plot_values(V_sarsa)