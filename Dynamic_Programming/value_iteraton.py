import numpy as np
from policy_improvement import policy_improvement
from frozenlake import FrozenLakeEnv
from plot_utils import plot_values

def value_iteration(env, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delt = 0
        for s in range(env.nS):
            old_v = V[s]
            new_Q = np.zeros(env.nA)
            for a in range(env.nA):
                for s_pro,next_state,reward,done in env.P[s][a]:
                    new_Q[a] += s_pro * (reward + gamma * V[next_state])
            V[s] = new_Q[np.argmax(new_Q)]
            delt = max(delt,np.abs(old_v - V[s]))
        if delt < theta:
            break
    policy = policy_improvement(env,V,gamma)
    return policy, V

# env = FrozenLakeEnv()
# policy_pi, V_pi = value_iteration(env)
# plot_values(V_pi)
