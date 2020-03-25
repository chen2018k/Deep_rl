from collections import defaultdict, deque
import numpy as np
import random
import gym
import sys
import matplotlib.pyplot as plt
from plot_utils import plot_values



env = gym.make('CliffWalking-v0')

def epsilon_greedy(Q, state, eps, nA):
    if random.random() > eps:
        return np.argmax(Q[state])
    else:
        return random.choice(np.arange(nA))

def update_Q_sarsamax(alpha, gamma, Q, state, action, reward, next_state=None):
    curr = Q[state][action]
    Qsa_next = np.max(Q[next_state]) if next_state is not None else 0
    target = reward + (gamma * Qsa_next)
    new_value = curr + (alpha * (target - curr))
    return new_value

def q_learning(env, num_episodes, alpha, gamma=1.0, plot_every=100):
    # 初始化Q
    nA = env.action_space.n
    Q = defaultdict(lambda : np.zeros(nA))

    #查看进度
    tmp_scores = deque(maxlen=plot_every)
    avg_scores = deque(maxlen=num_episodes)


    # 对于每一个episode
    for i_episode in range(1, num_episodes + 1):

        #查看进度
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes),end="")
            sys.stdout.flush()


        # 初始化得分
        score = 0

        # 设置递减参数epsilon
        eps = 1.0 / i_episode

        # 获得第一个状态值
        state = env.reset()

        # 对该幕之后的每一个状态
        while True:
            # 根据epsilon-greedy 选择动作
            action = epsilon_greedy(Q, state, eps, nA)
            # 获得next_state,reward
            next_state, reward, done, info = env.step(action)

            score += reward

            # 更新Q
            # 为什么不按sarsa.py里分done和not none两种情况来做?
            Q[state][action] = update_Q_sarsamax(alpha, gamma, Q, \
                 state, action, reward, next_state)
            state= next_state

            if done:
                tmp_scores.append(score)
                break

        if (i_episode % plot_every == 0):
                avg_scores.append(np.mean(tmp_scores))
    plt.plot(np.linspace(0, num_episodes, len(avg_scores), endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))

    return Q



Q_sarsamax = q_learning(env, 50000, .01)
# policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4,12))
#
# print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
# print(policy_sarsamax)
#
# # plot the estimated optimal state-value function
# plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])
