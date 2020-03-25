import sys
import gym
import numpy as np
from collections import defaultdict

from plot_utils import plot_blackjack_values, plot_policy

env = gym.make('Blackjack-v0')

# 生成episode，通过p来传递greedy策略中对不同动作的选择概率参数
def generate_episode_from_Q(env, Q, epsilon, nA):
    """ generates an episode from following the epsilon-greedy policy """
    episode = []
    state = env.reset()
    while True:
        action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
            if state in Q else env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

# 选出最优动作，并根据epsilon为其分配更多被选择概率，其他选择概率都平分且所有选择的概率和为一
def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s

# 用最近的一组幕来更新Q
def update_Q(env, episode, Q, alpha, gamma):
    """ updates the action-value function estimate using the most recent episode """
    states, actions, rewards = zip(*episode)

    # prepare for discounting
    discounts = np.array([gamma ** i for i in range(len(rewards) + 1)]) # 为什么要在这里加一？答：x[:-0]会返回空数组
    for i, state in enumerate(states):
        old_Q = Q[state][actions[i]]
        Q[state][actions[i]] = old_Q + alpha * (sum(rewards[i:] * discounts[:-(1 + i)]) - old_Q)
    return Q


def mc_control(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
    nA = env.action_space.n

    # 初始化
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = eps_start

    # 对每个episode处理
    for i_episode in range(1, num_episodes + 1):

        # 查看进度
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # 设置epsilon的值，随幕的增加一直缩减
        epsilon = max(epsilon * eps_decay, eps_min)

        # generate an episode by following epsilon-greedy policy
        episode = generate_episode_from_Q(env, Q, epsilon, nA)

        # update the action-value function estimate using the episode
        # 其实并不需要env，这里只是根据当前幕，对该幕中每一个状态对应的Q进行更新
        Q = update_Q(env, episode, Q, alpha, gamma)

    # determine the policy corresponding to the final action-value function estimate
    policy = dict((k, np.argmax(v)) for k, v in Q.items())
    return policy, Q


# policy, Q = mc_control(env, 500, 0.02)
# V = dict((k, np.max(v)) for k, v in Q.items())
#
# # plot the state-value function
# plot_blackjack_values(V)
#
# # plot the policy
# plot_policy(policy)



