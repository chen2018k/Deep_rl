import sys
import gym
import numpy as np
from collections import defaultdict
from plot_utils import plot_blackjack_values, plot_policy

#
# for i_episode in range(3):
#     state = env.reset()
#     while True:
#         print(state)
#         action = env.action_space.sample()
#         print(action)
#         state, reward, done, info = env.step(action)
#         if done:
#             print('End game! Reward: ', reward)
#             print('You won :)\n') if reward > 0 else print('You lost :(\n')
#             break

# 创建环境
env = gym.make('Blackjack-v0')

# 生成交互的episode
def generate_episode_from_limit_stochastic(bj_env):
    episode = []
    state = bj_env.reset()
    while True:

        # 设置选择动作的概率
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]

        # 通过概率选择动作
        action = np.random.choice(np.arange(2), p=probs)

        # 与环境交互
        next_state, reward, done, info = bj_env.step(action)

        # 如果episode没有结束，就将状态动作对添加到episode中
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

# 生成三个episode
# for i in range(3):
#     print(generate_episode_from_limit_stochastic(env))

#
def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):

    # 初始化
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # 对每一个episode循环
    for i_episode in range(1, num_episodes + 1):
        # 进度查看
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # 生成episode
        episode = generate_episode(env)

        # 与环境交互，取得三元组
        states, actions, rewards = zip(*episode)

        # 计算discounting
        discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])

        # 计算回报值的和，访问次数以及Q
        # 对episode中的每一个A,S对都进行一次计算
        for i, state in enumerate(states):
            returns_sum[state][actions[i]] += sum(rewards[i:] * discounts[:-(1 + i)])
            N[state][actions[i]] += 1.0
            Q[state][actions[i]] = returns_sum[state][actions[i]] / N[state][actions[i]]

    return Q

# # obtain the action-value function
# Q = mc_prediction_q(env, 50000, generate_episode_from_limit_stochastic)
#
# # obtain the corresponding state-value function
# V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
#          for k, v in Q.items())
#
# # plot the state-value function
# plot_blackjack_values(V_to_plot)