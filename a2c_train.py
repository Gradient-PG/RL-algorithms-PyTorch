import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import torch.optim as optim
from torch.distributions import Categorical

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

        # sample the action
        dist = Categorical(action_tensor)
        action = dist.sample().item()

        print("episode starts")
        while not done:
            next_obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count = step_count + 1
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0)
            action_tensor = actor(next_obs_tensor)

            # sample the action
            dist = Categorical(action_tensor)
            action_sample = dist.sample()
            action = action_sample.item()

            value_tensor = critic(obs_tensor)
            next_value_tensor = critic(next_obs_tensor) 

            # calculate the TD-target
            # TD target is meant to be a fixed reference point for the loss  
            # should not be used for optimizer adjust
            # so we use .detach() 
            td_target = rew + gamma * next_value_tensor.detach() * (1 - done)

            # calcualte the advantage function
            advantage = td_target - value_tensor.detach()

            # calculate the log_prob
            # pass the index of the chosen action in form of a tensor
            log_prob = dist.log_prob(action_sample)

            # calculate the actor loss
            actor_loss = -1 * log_prob * advantage

            # calculate the critic loss
            critic_loss = F.mse_loss(td_target, value_tensor)

            # optimize the actor
            optim_actor.zero_grad()
            actor_loss.backward()
            optim_actor.step()

            # optimize the critic
            optim_critic.zero_grad()
            critic_loss.backward()
            optim_critic.step()

            print(f"step: {step_count}, action: {action}, reward {rew}, actor loss: {actor_loss.item():.2f}, critic loss: {critic_loss.item():.2f}")

            ep_return += rew
            obs = next_obs
            obs_tensor = next_obs_tensor

env = gym.make("CartPole-v1")
input_size = env.observation_space.shape[0]
num_actions = env.action_space.n
actor = Actor(input_size, num_actions)
critic = Critic(input_size)

train(env, actor, critic, nb_steps=100)
