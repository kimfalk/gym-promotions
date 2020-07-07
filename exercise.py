import gym

from agents.q_learning_agent import QLearningAgent


class User:
    def __init__(self, user_id=0, age=0, children=0):
        self.user_id = user_id
        self.age = age
        self.children = children

    def __str__(self):
        return f"(user: {self.user_id}, age: {self.age}, children: {self.children})"


def q_learning_main():
    env = gym.make('gym_promotions:promotions-v0')
    env.reset()
    env.render()
    agent = QLearningAgent(env)
    agent.q_learning(50000)


def main():
    env = gym.make('gym_promotions:promotions-v0')
    env.reset()
    env.render()
    rewards = 0
    for i in range(30):

        action = env.action_space.sample()
        user, reward, done = env.step(action)
        rewards += reward
        if i % 10 == 0:
            print(f"action: {action}, user: {user}, reward: {reward}")

    print(f"total reward: {rewards}")


if __name__ == '__main__':

    q_learning_main()


