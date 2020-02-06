## gym-promotions

This is an implementation of an environment to test RL algorithms in showing promotions to users.

For more info on the OpenAi gyms look here: [OpenAi gym](https://github.com/openai/gym)

### Install
pip install -e gym-promotions

### Code

```python
import gym
env = gym.make('FrozenLake-v0')
env.reset()
env.render()
```