from gym.envs.registration import register

register(
    id='promotions-v0',
    entry_point='gym_promotions.envs:PromotionsEnv',
)