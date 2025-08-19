import torch
import torch.nn as nn
import torch.nn.functional as F

gamma = 0.99

class Actor(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_actions)
    
    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        # dim=-1 applies softmax across the action dimension
        x = F.softmax(self.fc2(x), dim=-1)
        return x
    
class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
import gymnasium as gym
from  torch.distributions import Categorical

env = gym.make('CartPole-v1')
input_size = env.observation_space.shape[0]
num_actions = env.action_space.n

actor = Actor(input_size, num_actions)

critic = Critic(input_size)

obs, info = env.reset()

print(f"obs.dtype: {obs.dtype}")
print(f"type(obs): {type(obs)}")

obs_tensor = torch.FloatTensor(obs)
obs_tensor = obs_tensor.unsqueeze(0)
done = False
ep_return = 0
action_tensor = actor(obs_tensor)
print("softmax action tensor: ")
print(action_tensor)

# sample the action
dist = Categorical(action_tensor)
action = dist.sample().item()

print(f"type(dist.sample()): {type(action)}")

next_obs, rew, terminated, truncated, info = env.step(action)
done = terminated or truncated
next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0)
action_tensor = actor(next_obs_tensor)

value_tensor = critic(obs_tensor)
next_value_tensor = critic(next_obs_tensor)  

print(f"next_value tensor: {next_value_tensor}")
print(f"next_value tensor.detach(): {next_value_tensor.detach()}")

# calculate the TD-target
td_target = rew + gamma * next_value_tensor * (1 - done)
print(f"TD-target: {td_target}")

# sample the action
dist = Categorical(action_tensor)
action = dist.sample()

# log_prob
log_prob = dist.log_prob(action)
print(f"log_prob: {log_prob}")


obs = next_obs
obs_tensor = next_obs_tensor
# env.step(action)

