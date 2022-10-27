import gym
import assistive_gym

import importlib


def sample_action(env, coop):
    if coop:
        return {'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}
    return env.action_space.sample()

env_name = "FeedingSawyerHuman-v1"
coop = 'Human' in env_name
module = importlib.import_module('assistive_gym.envs')
env_class = getattr(module, env_name.split('-')[0] + 'Env')
env = env_class()
env.seed(1024)

env.reset()
action = sample_action(env, coop)
env.step(action)