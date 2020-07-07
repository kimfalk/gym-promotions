import gym
import gym_promotions

import pytest

from agents.q_learning_agent import QLearningAgent, freeze_user
from gym_promotions.envs import PromotionsEnv
from gym_promotions.envs.promotions_env import Data


def get_test_data():
    users = [{'user_id': 1, 'age': 45, 'children': 0},
             {'user_id': 2, 'age': 35, 'children': 1}]

    promotions = [1, 2, 3]

    return Data(users, promotions)


def x_test_Q_learning_agent():

    data = get_test_data()
    env = PromotionsEnv(data)
    env.reset()
    env.render()
    agent = QLearningAgent(env)
    rewards = 0
    agent.q_learning(100)
    user1_q = agent.Q[freeze_user(data.users[0])]
    user2_q = agent.Q[freeze_user(data.users[1])]

    assert user2_q[0] < user2_q[1]
    assert user2_q[2] < user2_q[1]
    assert user1_q[1] < user1_q[0]
    assert user1_q[2] < user1_q[0]