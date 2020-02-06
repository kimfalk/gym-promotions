## gym-promotions

This is an implementation of an environment to test RL algorithms in showing promotions to users.

For more info on the OpenAi gyms look here: [OpenAi gym](https://github.com/openai/gym)

### Install
pip install -e gym-promotions

### Code

To run the gym, try out the following code, or have a look at the `exercise.py` file:
```python
import gym
env = gym.make('FrozenLake-v0')
env.reset()
env.render()

for i in range(3):
    print(env.step(1))
```

### To do

* Add real data -> currently it uses a small hardcoded dataset
* Add examples of solutions
* Tests?