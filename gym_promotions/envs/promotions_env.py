from collections import defaultdict

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from gym_promotions.render.promotion_graph import PromotionGraph

DEBUG = False


class PromotionsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data=None, max_steps=5000):
        self.max_steps = max_steps
        if data is None:
            data = self.get_test_data()
        self.data = data
        self.current_step = 0
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

    def step(self, action):

        self._take_action(action)

        self.current_step += 1

        reward = self.data.clicked(self.state, action)

        self.promotions_shown[action] += 1
        self.promotions_clicked[action] += reward
        # This allows the state also to contain knowledge of which promotions are seen, and whether they clicked.
        # currently the click is not stocastic so this will work
        # todo: show an average value of clicks.
        self.seen_promotions[self.state[0]['user_id']][action] = reward + 1
        self.clicks += reward

        self.state = self._next_observation()
        done = (self.current_step >= self.max_steps)
        info = ''
        if done:
            # print(f"In {self.current_step} there were {self.clicks} clicks. views: {[(promotion,count) for promotion, count in self.promotions_shown.items()]}. clicks: {[(promotion,count) for promotion, count in self.promotions_clicked.items()]}")
            print(
                f"In {self.current_step} there were {self.clicks} clicks. views: {dict(self.promotions_shown.items())}. clicks: {dict(self.promotions_clicked)}")
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
        users = [{'user_id': 1, 'age': 30, 'children': 0}, \
                 {'user_id': 2, 'age': 30, 'children': 1}, \
                 {'user_id': 3, 'age': 30, 'children': 1}, \
                 {'user_id': 4, 'age': 30, 'children': 0}, \
                 {'user_id': 5, 'age': 45, 'children': 0}, \
                 {'user_id': 6, 'age': 45, 'children': 2}, \
                 {'user_id': 7, 'age': 45, 'children': 0}, \
                 {'user_id': 8, 'age': 45, 'children': 1}, \
                 ]

        promotions = [ \
            0,  # promotion
            1,  # 5% discount
            2,  # 10% discount
        ]

        return Data(users, promotions)


class Data:
    def __init__(self, users, promotions):
        print(f"data: users: {len(users)}, promotions: {len(promotions)}")
        self.users = users
        self.promotions = promotions

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_promotions(self):
        return len(self.promotions)

    def clicked(self, state, promotion_id):
        user = state[0]
        assert (user is not None)
        assert (promotion_id < self.n_promotions)

        if user["age"] < 30 \
                or (user["age"] > 40 and promotion_id == 0) \
                or (user["age"] <= 40 and user["children"] == 0 and promotion_id == 1) \
                or (user["age"] <= 40 and user["children"] > 0 and promotion_id == 2):

            if DEBUG:
                print(f"click: user age {user['age']} and promotion id {promotion_id}")
            return 1
        else:
            if DEBUG:
                print(f"no-click: user age {user['age']} and promotion id {promotion_id}")
            return 0
