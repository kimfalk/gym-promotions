from gym.envs.registration import register

register(
    id='promotions-v0',
    entry_point='gym_promotions.envs:PromotionsEnv',
)
register(
    id='promotions-probabilistic-v0',
    entry_point='gym_promotions.envs:PromotionsProbabilisticEnv',
)
register(
    id='promotions-probabilistic-space-v0',
    entry_point='gym_promotions.envs:PromotionsProbabilisticFromSpaceEnv',
)