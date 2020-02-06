import gym
import gym_promotions

env = gym.make('gym_promotions:promotions-v0')
env.reset()
env.render()

for i in range(3):
    print(env.step(1))
