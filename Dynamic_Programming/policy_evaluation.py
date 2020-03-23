import numpy as np
import copy
from frozenlake import FrozenLakeEnv
from plot_utils import plot_values

#策略评估: （策略、MDP -> V)
def policy_evaluation(env, policy, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            Vs = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    Vs += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(V[s]-Vs))
            V[s] = Vs
        if delta < theta:
            break
    return V

#截断的策略评估:（策略、MDP -> V）
def truncated_policy_evaluation(env, policy, max_num,gamma=1):
    V = np.zeros(env.nS)
    counter = 0
    while (counter < max_num):
        for s in range(env.nS):
            Vs = 0
            for a, a_pro in enumerate(policy[s]):
                for s_pro, next_state, reward, done in env.P[s][a]:
                    Vs += a_pro * s_pro * (reward + gamma * V[next_state])
            V[s] = Vs
        counter += 1
    return V

#得出动作价值函数： （V,MDP -> Q)
def q_from_v(env, V, s, gamma=1):
    q = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    return q

def get_Q(env, V, gamma=1):
    Q = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        Q[s] = q_from_v(env, V, s)
    return Q



# #创建环境
# env = FrozenLakeEnv()
# #
# 设置一个随机策略（每个A-S pair 被选到的概率相等。））
# random_policy = np.ones([env.nS, env.nA]) / env.nA
#
#
# V1 = policy_evaluation(env, random_policy)
# V2 = truncated_policy_evaluation(env,random_policy,1000)
#
# #Q = get_Q(env,V)
# plot_values(V2)