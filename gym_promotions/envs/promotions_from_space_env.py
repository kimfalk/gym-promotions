import random
from collections import defaultdict

import numpy as np
import sklearn
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from gym_promotions.render.promotion_graph import PromotionGraph

DEBUG = False
PROMOTIONS = {'toy': 0, 'food': 1, 'bike': 2, 'car': 3, 'house': 4}


class PromotionsProbabilisticFromSpaceEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_steps=5000, user_vector_dim=10, no_users=1000, promotions=PROMOTIONS):
        self.max_steps = max_steps
        self.no_users = no_users
        self.current_step = 0
        self.total_steps = 0
        self.clicks = 0

        self.promotions_shown = defaultdict(int)
        self.promotions_clicked = defaultdict(int)
        self.promotions = promotions
        self.user_vector_dim = user_vector_dim
        self.seen_promotions = defaultdict(lambda: np.zeros(len(self.promotions)))
        self.visualization = None
        self.state = None
        self.action_space = spaces.Discrete(len(self.promotions))
        self.observation_space = spaces.Box(low=0, high=100, shape=([user_vector_dim]), dtype=np.float)


        self.promo_vecs = create_promotions(self.promotions, user_vector_dim)
        self.user_vecs = create_users(user_vector_dim, no_users)


        self.df = pd.DataFrame(data=list(self.promo_vecs.keys()) + [str(idx) for idx in range(self.no_users)],
                               columns=["id"])
        self.df['hue'] = list(self.promo_vecs.keys()) + ['user' for idx in range(self.no_users)]
        self.df['vec'] = list(self.promo_vecs.values()) + list(self.user_vecs.values())
        self.df['clicked'] = 'none'
        self.steps = 0
        for promo in promotions:
            self.df[promo] = 0.0

        self.current_user = self.get_user()
        self.reset()

    def step(self, promotion):
        """Execute one time step within the environment"""

        click = self.clicked(self.current_user[0], promotion)
        reward = 1.0 if click else 0.0
        done = False

        if self.current_step % self.max_steps == 0:
            done = True
        self.current_step += 1
        new_observation = self.get_user()

        self.current_user = new_observation

        return new_observation, reward, done, ""

    def reset(self):
        """ Reset the state of the environment to an initial state"""
        self.current_step = 0
        self.total_steps = 0
        self.clicks = 0

        self.promotions_shown = defaultdict(int)
        self.promotions_clicked = defaultdict(int)
        self.seen_promotions = defaultdict(lambda: np.zeros(self.data.n_promotions))
        self.current_user = self.get_user()

        return self.current_user

    def render(self, mode='human', close=False):
        """ Render the environment to the screen"""

        user_vecs = self.user_vecs
        promo_vecs = self.promo_vecs
        pca = create_pca(list(self.promo_vecs.values()) + list(self.user_vecs.values()))
        trans_promo = pca.transform(list(promo_vecs.values()) + list(user_vecs.values()))

        df = pd.DataFrame(data=trans_promo, columns=["x", "y"])
        df['id'] = list(promo_vecs.keys()) + [str(idx) for idx in range(self.no_users)]
        df['hue'] = list(promo_vecs.keys()) + ['user' for idx in range(self.no_users)]
        df['vec'] = list(promo_vecs.values()) + list(user_vecs.values())
        df['clicked'] = 'none'
        for promo in self.promotions:
            df[promo] = 0.0

        # sns.lmplot(x="x", y="y",
        #            data=df.sort_values('id', ascending=False),
        #            fit_reg=False,
        #            hue='hue',  # color by cluster
        #            legend=True,
        #            scatter_kws={"s": 80})  # specify the point size

        plt.figure(figsize=(20, 10))
        sns.scatterplot(x="x", y="y",
                        data=df.sort_values('id', ascending=False),
                        hue='hue',  # color by cluster
                        size='car_click_prob', sizes=(1, 200))  # specify the point size

    def click_probability(self, user, promo, max_click_prop=0.1):

        max_dist = np.linalg.norm(np.ones(self.user_vector_dim) - np.zeros(self.user_vector_dim))
        dist = self.distance(user, promo)
        # click_prob = ((max_dist-dist)/max_dist)*max_click_prop
        # click_prob = np.exp(-dist)*max_click_prop
        click_prob = np.power(10, -dist)
        # print(f"{max_dist}, {dist}, {click_prob}")
        return click_prob

    def get_user(self):
        user_id, vec = random.choice(list(self.user_vecs.items()))

        click_history = np.array(list(self.df.iloc[user_id + len(self.promotions)][self.promotions]))
        state = np.concatenate((click_history, vec), axis=0)
        self.current_user = user_id
        return user_id, state

    def clicked(self,user_id, promotion):

        user = self.user_vecs[user_id]
        promo = self.promo_vecs[promotion]
        proba = self.click_probability(user, promo)
        clicked = random.random() < proba
        if clicked:
            self.df.loc[user_id + len(self.promotions)]['clicked'] = promotion
        return clicked

    def distance(self, user, promo):
        return np.linalg.norm(user - promo)

    def distances(self, user, promos):

        distances = {}
        for k, v in promos.items():
            distances[k] = self.distance(user, v)
        return distances

    def closest_promo(self, user, promos):
        ds = self.distances(user, promos)

        return min(ds.keys(), key=(lambda k: ds[k]))


def create_pca(vecs):
    pca = PCA(n_components=2)
    pca.fit(vecs)
    return pca


def create_random_vector(dim):
    return np.random.rand(dim)+1.0


def create_promotions(promotions, dim):
    result = {}
    for promotion in promotions:
        result[promotion] = create_random_vector(dim)
    return result


def create_users(dim, no_users):
    result = {}
    for id in range(no_users):
        result[id] = create_random_vector(dim)
    return result
