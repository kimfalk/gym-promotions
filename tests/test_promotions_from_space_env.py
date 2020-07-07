import random
from collections import Counter
import tensorflow as tf

import numpy as np

from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

from gym_promotions.envs.promotions_from_space_env import PromotionsProbabilisticFromSpaceEnv

def test_init_of_env():
    env = PromotionsProbabilisticFromSpaceEnv()
    print(f"\n env obs space {env.observation_space}")
    print(f"\n env action space {env.action_space}")

    env.reset()
    user_cnt = Counter()
    promo_cnt = Counter()
    rewards = 0
    N = 1000
    for i in range(N):
        ran_promo = random.choice(list(env.promotions.keys()))
        observation, reward, done, _ = env.step(ran_promo)
        user_cnt[observation[0]] += 1
        promo_cnt[ran_promo] += 1
        rewards += reward
    print(f"\n Promotions: {promo_cnt}")
    print(f"\n No of users: {len(user_cnt.keys())}")
    print(f"\n Avg reward: {rewards/N}")


# def test_env_w_tf_agent():
    # env = PromotionsProbabilisticFromSpaceEnv()
    #
    # learning_rate = 1e-3  # @param {type:"number"}
    # print(f"shape={(env.observation_space.shape[0] + env.action_space.n,)}")
    # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    # observation_spec = tf.TensorSpec(shape=(env.observation_space.shape[0] + env.action_space.n,),
    #                                  dtype=tf.dtypes.float32)
    # action_spec = ts.tensor_spec.BoundedTensorSpec(shape=env.action_space.shape,
    #                                                maximum=len(env.promotions),
    #                                                minimum=0,
    #                                                dtype=tf.dtypes.int16)
    # q_net = q_network.QNetwork(
    #     observation_spec,
    #     action_spec)
    # # TimeStep(step_type=array([0], dtype=int32),
    # #          reward=array([0.], dtype=float32),
    # #          discount=array([1.], dtype=float32),
    # #          observation=array([[0.01579372, 0.01083774, -0.03600705, -0.02049673]],
    # #                            dtype=float32)), array([0], dtype=int32), \
    # time_step_spec = ts.time_step_spec(observation_spec)
    # train_step_counter = tf.Variable(0)
    # agent = dqn_agent.DqnAgent(time_step_spec,
    #                            action_spec,
    #                            q_net,
    #                            optimizer,
    #                            td_errors_loss_fn=common.element_wise_squared_loss,
    #                            train_step_counter=train_step_counter)
    # agent.initialize()
    # policy = agent.collect_policy
    # user = env.reset()
    # reward = 0.
    # discount = 1.
    # N = 4
    # for i in range(N):
    #     print(user[1])
    #     time_step = ts.TimeStep(step_type=tf.constant(0, dtype=tf.dtypes.int16),
    #                          reward=tf.constant(reward, dtype=tf.dtypes.float32),
    #                          discount=discount,
    #                          observation=user[1])
    #
    #     print(policy.action(time_step))
    #     observation, reward, done, _ = env.step()
