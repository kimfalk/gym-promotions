import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class PromotionsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data=None):

        if data is None:
            data = self.get_test_data()
        self.data = data
        self.state = None
        self.reward = 0
        self.reset()

    def step(self, action):

        self.reward += self.data.clicked(self.state['user_id'], action)
        self.state = self.get_observation()
        print(f"Step: user: {self.state}")
        done = False
        return self.state, self.reward, done

    def get_observation(self):

        return np.random.choice(self.data.users)

    def reset(self):
        self.state = self.get_observation()
        self.reward = 0

    def render(self, mode='human'):
        print(f"number Users: {self.data.n_users}")
        print(f"number promotions: {self.data.n_promotions}")
        print(f"reward: {self.reward}")

    def close(self):
        pass

    def get_test_data(self):
        users = [   {'user_id': 1, 'age': '45', 'children': 0}, \
                    {'user_id': 2, 'age': '45', 'children': 0}, \
                    {'user_id': 3, 'age': '45', 'children': 1}, \
                    {'user_id': 4, 'age': '45', 'children': 0}, \
                    {'user_id': 5, 'age': '40', 'children': 1}, \
                    {'user_id': 6, 'age': '40', 'children': 0}]

        promotions = [ \
            [1, 1, 0, 0, 0, 0], \
            [0, 0, 1, 0, 1, 0], \
            [0, 0, 0, 0, 0, 0], \
            [0, 0, 0, 0, 0, 0], \
            [0, 0, 0, 0, 1, 0], \
            [0, 0, 0, 0, 0, 1] \
            ]

        return Data(users, promotions)


class Data:
    def __init__(self, users, promotions):
        self.users = users
        self.promotions = promotions

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_promotions(self):
        return len(self.promotions)

    def clicked(self, user_id, promotion_id):
        print(f"promotions {self.n_promotions}")
        print(f"user_id {user_id}users {self.n_users}")

        assert (user_id <= self.n_users)
        assert (promotion_id < self.n_promotions)
        return self.promotions[promotion_id][user_id - 1]
