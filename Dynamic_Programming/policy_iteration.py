import numpy as np
from policy_evaluation import policy_evaluation
from policy_evaluation import truncated_policy_evaluation
from policy_improvement import policy_improvement
from plot_utils import plot_values
import copy
from frozenlake import FrozenLakeEnv

# 策略迭代算法
def policy_iteration(env, gamma=1, theta=1e-8):
    # 初始策略是随机策略
    policy = np.ones([env.nS, env.nA]) / env.nA
    policy_stable = False

    # 严格地查看改进后的策略与改进前的测试是否一致，如一致，则停止更新
    while True:
        V = policy_evaluation(env, policy)
        new_policy = policy_improvement(env,V,gamma)
        if (new_policy.all() == policy.all()):
            policy_stable = True
        policy = new_policy
        if (policy_stable == True):
            break

    # 此时输出的策略近似于最优策略
    return policy, V

# 截断的策略迭代算法是
def truncated_policy_iteration(env, max_num,gamma=1, theta=1e-8):
    # 初始策略是随机策略
    # policy = np.ones([env.nS, env.nA]) / env.nA
    V = np.zeros(env.nS)

    # 更新到一定步后停止更新
    while True:
        policy = policy_improvement(env,V,gamma)
        V_old = copy.copy(V)
        V = truncated_policy_evaluation(env,policy,max_num)
        if (np.max(np.abs(V_old - V))) < theta:
            break
    # 此时输出的策略近似于最优策略
    return policy, V

# env = FrozenLakeEnv()
# policy_pi, V_pi = truncated_policy_iteration(env,1000)
# #
# # # print the optimal policy
# # print("\nOptimal Policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3):")
# # print(policy_pi,"\n")
# #
# plot_values(V_pi)