import torch
import torch.nn as nn
import torch.nn.functional as F

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

env = gym.make('CartPole-v1')
input_size = env.observation_space.shape[0]
num_actions = env.action_space.n

actor = Actor(input_size, num_actions)

critic = Critic(input_size)

obs, info = env.reset()

print(f"obs.dtype: {obs.dtype}")
print(f"type(obs): {type(obs)}")

print("torch.FloatTensor(obs)")
obs = torch.FloatTensor(obs)

print(f"obs.dtype: {obs.dtype}")
print(f"type(obs): {type(obs)}")

action = actor(obs)

print(f"type(action): {type(action)}")

# Convert tensor to Python int 
print("action = action.item()")
# action = action.item()

print("softmax tensor: ")
print(action)
# print(f"type(action): {type(action)}")

action = torch.argmax(action, dim=-1)
action = action.item()
print(f"action {action}")
print(f"type(action): {type(action)}")

# env.step(action)

