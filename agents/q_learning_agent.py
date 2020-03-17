from collections import defaultdict

import numpy as np

def freeze_user(state):
    """
    Create a touple which can be used as a key in the dictionary
    :param state:
    :return:
    """
    print(state)
    return state['age'], state['children']


class QLearningAgent():

    def __init__(self, env):
        self.env = env
        self.return_sum = defaultdict(lambda: np.zeros(env.action_space.n))
        self.N = defaultdict(lambda: np.zeros(env.data.n_promotions))
        self.Q = defaultdict(lambda: np.zeros(env.data.n_promotions))
        self.state = self.env.reset()

    def q_learning(self, max_steps = 400):
        eta = 0.625
        gamma = 0.9

        self.state = self.env.reset()
        rev_list = []
        total_reward = 0
        done = False

        for step_number in range(max_steps):
            s = self.state

            user = freeze_user(s[0])
            actions = self.Q[user]

            actions_with_noise = actions + np.random.randn(1, self.env.data.n_promotions) * (1. / ((step_number/2) + 1))
            print(f"{user}: {actions} -> {actions_with_noise}")
            action = np.argmax(actions_with_noise)
            next_state, reward, done, info = self.env.step(action)

            next_user = freeze_user(next_state[0])
            self.Q[user][action] += eta * (reward + gamma * np.max(self.Q[next_user]) - self.Q[user][action])
            total_reward += reward
            self.state = next_state

            rev_list.append(total_reward)

            if done:
                break

        print("Reward Sum on all episodes " + str(sum(rev_list) / max_steps))
        print("Final Values Q-Table")

        def sortId(user):
            return user[0]
        orderedUsers = list(self.Q.keys())
        orderedUsers.sort(key=sortId)

        [print(f"{user}: {self.Q[user]}") for user in orderedUsers]