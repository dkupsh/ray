from copy import deepcopy

import numpy as np
import gymnasium as gym

class RankedRewardsBuffer:
    def __init__(self, buffer_max_length, percentile):
        self.buffer_max_length = buffer_max_length
        self.percentile = percentile
        self.buffer = []

    def add_reward(self, reward):
        if len(self.buffer) < self.buffer_max_length:
            self.buffer.append(reward)
        else:
            self.buffer = self.buffer[1:] + [reward]

    def normalize(self, reward):
        reward_threshold = np.percentile(self.buffer, self.percentile)
        if reward < reward_threshold:
            return -1.0
        else:
            return 1.0

    def get_state(self):
        return np.array(self.buffer)

    def set_state(self, state):
        if state is not None:
            self.buffer = list(state)


def get_r2_env_wrapper(env_creator, r2_config : dict):
    class RankedRewardsEnvWrapper(gym.Env):
        def __init__(self, env_config):
            self.env = env_creator(env_config)
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space
            max_buffer_length = r2_config.get("buffer_max_length", 100)
            percentile = r2_config.get("percentile", 70)
            self.r2_buffer = RankedRewardsBuffer(max_buffer_length, percentile)
            
            self.total_current_reward = 0

            if r2_config["initialize_buffer"]:
                self._initialize_buffer(r2_config.get("num_init_rewards"))
                
        def _initialize_buffer(self, num_init_rewards=100):
            # initialize buffer with random policy
            for _ in range(num_init_rewards):
                obs, info = self.env.reset()
                terminated = truncated = False
                while not terminated and not truncated:
                    mask = obs["action_mask"]
                    probs = mask / mask.sum()
                    action = np.random.choice(np.arange(mask.shape[0]), p=probs)
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                self.r2_buffer.add_reward(reward)

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            self.total_current_reward += reward
            
            if terminated or truncated:
                return obs, self.r2_buffer.normalize(self.total_current_reward), terminated, truncated, info
            else:
                return obs, 0, terminated, truncated, info

        def get_state(self):
            state = {
                "env_state": self.env.get_state(),
                "buffer_state": self.r2_buffer.get_state(),
                "total_current_reward": self.total_current_reward,
            }
            return deepcopy(state)

        def reset(self, *, seed=None, options=None):
            self.total_current_reward = 0
            return self.env.reset()

        def set_state(self, state):
            obs = self.env.set_state(state["env_state"])
            self.r2_buffer.set_state(state["buffer_state"])
            self.total_current_reward = state["total_current_reward"]
            return obs

    return RankedRewardsEnvWrapper
