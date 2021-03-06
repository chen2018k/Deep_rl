import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = 0.005
        self.gamma = 0.85
        self.alpha = 0.07

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        Q = self.Q
        eps = self.eps
        if random.random() > eps:
            return np.argmax(Q[state])
        else:
            return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        Q = self.Q
        gamma = self.gamma
        alpha = self.alpha
        eps = self.eps
        current = Q[state][action]
        nA = self.nA

        policy_s = np.ones(nA) * eps / nA  # current policy (for next state S')
        policy_s[np.argmax(Q[next_state])] = 1 - eps + (eps / nA)  # greedy action

        # 求期望
        Qsa_next = np.dot(Q[next_state], policy_s)  # get value of state at next time step
        target = reward + (gamma * Qsa_next)  # construct target
        Q[state][action] = current + (alpha * (target - current))  # get updated value



# import numpy as np
# import random
# from collections import defaultdict
#
# class Agent:
#
#     def __init__(self, nA=6):
#         """ Initialize agent.
#         Params
#         ======
#         - nA: number of actions available to the agent
#         """
#         self.nA = nA
#         self.Q = defaultdict(lambda: np.zeros(self.nA))
#         self.epsilon = 0.005
#         self.gamma = 0.8 # 1.0
#         self.alpha = 0.07 # 0.01
#
#     def get_probs(self,Q_s):
#         """ obtains the action probabilities corresponding to epsilon-greedy policy """
#         policy_s = np.ones(self.nA) * self.epsilon / self.nA
#         best_a = np.argmax(Q_s)
#         policy_s[best_a] = 1 - self.epsilon + (self.epsilon / self.nA)
#         return policy_s
#
#     def select_action(self, state):
#         """ Given the state, select an action.
#         Params
#         ======
#         - state: the current state of the environment
#         Returns
#         =======
#         - action: an integer, compatible with the task's action space
#         """
#         act_space = [i for i in range(0, self.nA)]
#         action = np.random.choice(np.arange(self.nA),
#                                   p = self.get_probs(self.Q[state])) if state in self.Q else random.choice(act_space)
#         # return np.random.choice(self.nA)
#         return action
#
#     def step(self, state, action, reward, next_state, done):
#         """ Update the agent's knowledge, using the most recently sampled tuple.
#         Params
#         ======
#         - state: the previous state of the environment
#         - action: the agent's previous choice of action
#         - reward: last reward received
#         - next_state: the current state of the environment
#         - done: whether the episode is complete (True or False)
#         """
#         # self.Q[state][action] += 1
#         # next_state, reward, done, _ = env.step(a_t)
#         # print(state,reward,done, prob)
#         # 36 -1 False {'prob': 1.0}
#         a_t_1 = self.select_action(next_state)
#         self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * (self.Q[next_state][a_t_1]) - self.Q[state][action])
