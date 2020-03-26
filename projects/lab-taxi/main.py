from agent import Agent
from monitor import interact
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)

