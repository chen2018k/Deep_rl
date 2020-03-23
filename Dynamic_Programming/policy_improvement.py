import numpy as np
from policy_evaluation import q_from_v
from frozenlake import FrozenLakeEnv
from policy_evaluation import policy_evaluation


# 策略改进（env,V -> 策略（new））
def policy_improvement(env, V, gamma=1):
    policy = np.zeros([env.nS, env.nA]) / env.nA
    for s in range(env.nS):
        q = q_from_v(env, V, s, gamma)

        # OPTION 1: construct a deterministic policy
        # policy[s][np.argmax(q)] = 1
        best_a = np.argwhere(q == np.max(q)).flatten()
        policy[s] = np.sum([np.eye(env.nA)[i] for i in best_a], axis=0) / len(best_a)

        # OPTION 2: construct a stochastic policy that puts equal probability on maximizing actions
        # [np.eye(env.nA)[i] for i in best_a], axis=0
    return policy


#创建环境
# env = FrozenLakeEnv()
#
# #设置一个随机策略（每个A-S pair 被选到的概率相等。））
# random_policy = np.ones([env.nS, env.nA]) / env.nA
#
# V = policy_evaluation(env, random_policy)
#
# policy = policy_improvement(env,V)


