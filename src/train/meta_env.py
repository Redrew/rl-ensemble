import gym
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box as Continuous
import numpy as np

class MetaEnv(gym.Env):
    def __init__(self, env_config):
        self.base_env = gym.make(env_config["base_env"])
        self.policies = env_config["policies"]
        self.num_trainers = len(self.policies)

        self.is_discrete = type(self.base_env.action_space) == Discrete
        assert self.is_discrete or type(self.base_env.action_space) == Continuous

        # new action space
        action_shape = self.base_env.action_space.shape
        assert len(action_shape) <= 1
        self.action_space = self.base_env.action_space
        self.num_actions = self.action_space.n if self.is_discrete else action_shape[0]

        # new observation space
        # base_obs_space = self.base_env.observation_space
        base_obs_space = Continuous(np.array([]), np.array([]), (0,))
        base_obs_shape = base_obs_space.shape
        assert len(base_obs_shape) == 1
        assert type(base_obs_space) == Continuous

        action_features = self.num_actions * self.num_trainers
        self.num_features = base_obs_shape[0] + action_features
        low = np.concatenate((base_obs_space.low, np.zeros(action_features) \
            if self.is_discrete else np.tile(self.action_space.low, self.num_trainers)))
        high = np.concatenate((base_obs_space.high, np.ones(action_features) \
            if self.is_discrete else np.tile(self.action_space.high, self.num_trainers)))
        self.observation_space = Continuous(low, high, shape=(self.num_features,))
    
    def _expand(self, idx, size):
        oh = np.zeros(size)
        oh[idx] = 1
        return oh

    def _extend_obs(self, obs):
        actions = [policy.compute_single_action(obs)[0] for policy in self.policies]
        if self.is_discrete: actions = [self._expand(idx, self.num_actions) for idx in actions]
        # extended_obs = np.concatenate([obs] + actions)
        extended_obs = np.concatenate(actions)
        return extended_obs
    
    def reset(self):
        return self._extend_obs(self.base_env.reset())

    def step(self, action):
        obs, reward, done, info = self.base_env.step(action)
        return self._extend_obs(obs), reward, done, info

    def render(self):
        return self.base_env.render()