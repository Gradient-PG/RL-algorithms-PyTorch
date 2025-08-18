import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import torch.optim as optim

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
    
def train(
        env,
        actor,
        critic,
        device="cpu",
        nb_steps=10_000,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma = 0.99
):
    step_count = 0


    optim_actor = optim.Adam(actor.parameters(), lr_actor)
    optim_critic = optim.Adam(critic.parameters(), lr_critic)

    while step_count <= nb_steps:
        obs, info = env.reset()
        obs_tensor = torch.FloatTensor(obs)
        obs_tensor = obs_tensor.unsqueeze(0)
        done = False
        ep_return = 0
        action_tensor = actor(obs_tensor)
        action = torch.argmax(action_tensor, dim=-1).item()
        print("episode starts")
        while not done:
            next_obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count = step_count + 1
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0)
            action_tensor = actor(next_obs_tensor)
            action = torch.argmax(action_tensor, dim=-1).item()
            print(f"step: {step_count}, action: {action}, reward {rew}")
            obs = next_obs

env = gym.make("CartPole-v1")
input_size = env.observation_space.shape[0]
num_actions = env.action_space.n
actor = Actor(input_size, num_actions)
critic = Critic(num_actions)

train(env, actor, critic, nb_steps=100)
