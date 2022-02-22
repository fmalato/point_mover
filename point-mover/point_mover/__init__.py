from gym.envs.registration import register

register(id='point_mover-v0', entry_point='point_mover.envs:PointMover',)
register(id='point_mover-v2', entry_point='point_mover.envs:PointMover2',)
