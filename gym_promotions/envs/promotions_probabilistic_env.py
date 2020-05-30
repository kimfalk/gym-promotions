import random
from collections import defaultdict

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from gym_promotions.render.promotion_graph import PromotionGraph

DEBUG = False


class PromotionsProbabilisticEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data=None, max_steps=5000):
        self.max_steps = max_steps
        if data is None:
            data = self.get_test_data()
        self.reward_ratio = data.optimal_reward
        self.data = data
        self.current_step = 0
        self.total_steps = 0
        self.clicks = 0
        self.promotions_shown = defaultdict(int)
        self.promotions_clicked = defaultdict(int)
        self.seen_promotions = defaultdict(lambda: np.zeros(self.data.n_promotions))
        self.visualization = None
        self.state = None
        self.action_space = spaces.Discrete(self.data.n_promotions)
        self.observation_space = spaces.Box(low=0, high=100, shape=([2, 1]), dtype=np.int)
        self.reset()

    def _take_action(self, action):
        # send the promotion to the user
        pass

    def optimal_reward(self):
        return self.reward_ratio * self.max_steps

    def step(self, action):

        self._take_action(action)

        self.current_step += 1
        self.total_steps += 1
        clicked = self.data.clicked(self.state, action, self.total_steps)

        reward = 1. if clicked else 0.

        self.promotions_shown[action] += 1
        self.promotions_clicked[action] += reward
        # This allows the state also to contain knowledge of which promotions are seen, and whether they clicked.
        # currently the click is not stochastic so this will work
        # todo: show an average value of clicks.
        self.seen_promotions[self.state[0]['user_id']][action] = reward + 1
        self.clicks += reward

        self.state = self._next_observation()
        done = (self.current_step >= self.max_steps)
        info = ''
        if done:
            # print(f"In {self.current_step} there were {self.clicks} clicks. views: {[(promotion,count) for promotion, count in self.promotions_shown.items()]}. clicks: {[(promotion,count) for promotion, count in self.promotions_clicked.items()]}")
            print(
                f"{self.current_step},{self.clicks}, {sorted(self.promotions_shown.items())}, clicks: {sorted(self.promotions_clicked.items())}")
        return self.state, reward, done, info

    def reset(self):
        self.current_step = 0
        self.clicks = 0
        self.promotions_shown = defaultdict(int)
        self.promotions_clicked = defaultdict(int)
        self.state = self._next_observation()
        return self.state

    def render(self):
        pass

    def _next_observation(self):
        user = np.random.choice(self.data.users)
        seen_promotions = self.seen_promotions[user['user_id']]
        return user, seen_promotions

    def random_action(self):
        return np.random.choice(self.data.promotions)

    def close(self):
        pass

    def get_test_data(self):
        users = [{'user_id': 1, 'age': 30, 'children': 0},\
                 {'user_id': 2, 'age': 31, 'children': 1},\
                 {'user_id': 3, 'age': 32, 'children': 1},\
                 {'user_id': 4, 'age': 33, 'children': 0},\
                 {'user_id': 1, 'age': 20, 'children': 0},\
                 {'user_id': 2, 'age': 24, 'children': 1},\
                 {'user_id': 3, 'age': 29, 'children': 1},\
                 {'user_id': 4, 'age': 30, 'children': 0},\
                 {'user_id': 5, 'age': 41, 'children': 0},\
                 {'user_id': 6, 'age': 45, 'children': 3},\
                 {'user_id': 7, 'age': 44, 'children': 0},\
                 {'user_id': 8, 'age': 46, 'children': 1},\
                 {'user_id': 5, 'age': 50, 'children': 0},\
                 {'user_id': 6, 'age': 45, 'children': 4},\
                 {'user_id': 7, 'age': 47, 'children': 0},\
                 {'user_id': 8, 'age': 49, 'children': 2},\
 \
                 ]
        PROMOTION_MAP = {"no discount": 0,
                         "5% on x": 1,
                         "10% on y": 2,
                         "10% on x": 3,
                         "5% on y": 4,
                         "free coffee": 5,
                         "free transport": 6,
                         "buy one get one for free": 7,
                         "new arrival": 8,
                         "things for the garden": 9,
                         "things for the kitchen": 10
                         }

        return Data(users, PROMOTION_MAP)


class Data:
    def __init__(self, users, promotions):
        print(f"data: users: {len(users)}, promotions: {len(promotions)}")
        self.users = users
        self.promotions = promotions

    @property
    def optimal_reward(self):
        return 0.20

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_promotions(self):
        return len(self.promotions)

    def clicked(self, state, promotion_id, step):
        change_step = 200000

        user = state[0]
        n = random.random()
        if user['age'] > 40:
            if user['children'] > 0:
                if step < change_step:
                    promotion_probabilities = [0.2, 0.05, 0.1, 0.1, 0.2, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
                else:
                    promotion_probabilities = [0.02, 0.5, 0.01, 0.01, 0.02, 0.2, 0.02, 0.02, 0.02, 0.02, 0.02]

                return n < promotion_probabilities[promotion_id]
            elif user['children'] == 0:
                if step < change_step:
                    promotion_probabilities = [0.01, 0.05, 0.02, 0.1, 0.3, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
                else:
                    promotion_probabilities = [0.01, 0.05, 0.2, 0.1, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]

                return n < promotion_probabilities[promotion_id]
        if user['age'] <= 40:
            if user['children'] > 0:
                if step < change_step:
                    promotion_probabilities = [0.2, 0.5, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
                else:
                    promotion_probabilities = [0.02, 0.05, 0.01, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
                return n < promotion_probabilities[promotion_id]
            else:
                promotion_probabilities = [0.05, 0.2, 0.01, 0.01, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
                return n < promotion_probabilities[promotion_id]
