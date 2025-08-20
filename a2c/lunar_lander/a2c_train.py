import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
    
class StatisticsRecorder():
    def __init__(self):
        self.average_rewards = []
        self.episode_rewards = []

    def update_episode_reward(self, episode_reward):
        self.episode_rewards.append(episode_reward)

    def plot_graph(self, data, x_label, y_label, title, path):
        plt.plot(data)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(path)
        plt.close()

    def evaluation_checkpoint(self):
        self.average_rewards.append(np.mean(self.episode_rewards))
        self.episode_rewards = []
        self.plot_graph(
            self.average_rewards,
            "Training Epochs",
            "Average Reward per Episode",
            "Average Reward on LunarLander",
            "Imgs/average_reward_on_lunarlander.png")
        
def train(
        env,
        actor,
        critic,
        sr,
        device="cpu",
        nb_steps=10_000,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma = 0.99,
):
    step_count = 0
    # add a progress bar
    progress_bar = tqdm(total=nb_steps)

    optim_actor = optim.Adam(actor.parameters(), lr_actor)
    optim_critic = optim.Adam(critic.parameters(), lr_critic)

    while step_count <= nb_steps:
        obs, info = env.reset()
        obs_tensor = torch.FloatTensor(obs)
        obs_tensor = obs_tensor.unsqueeze(0)
        done = False
        ep_return = 0
        action_tensor = actor(obs_tensor)


        while not done:
            # actor selects action
            action_tensor = actor(obs_tensor)
            # sample the action
            dist = Categorical(action_tensor)
            action_sample = dist.sample()
            action = action_sample.item()

            next_obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count = step_count + 1
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0)


            value_tensor = critic(obs_tensor)
            next_value_tensor = critic(next_obs_tensor) 

            # calculate the TD-target
            # TD target is meant to be a fixed reference point for the loss  
            # should not be used for optimizer adjust
            # so we use .detach() 
            td_target = rew + gamma * next_value_tensor * (1 - done)

            # calcualte the advantage function
            advantage = td_target - value_tensor

            # calculate the critic loss
            critic_loss = F.mse_loss(td_target.detach(), value_tensor)
            # optimize the critic
            optim_critic.zero_grad()
            critic_loss.backward()
            optim_critic.step()

            # calculate the log_prob
            # pass the index of the chosen action in form of a tensor
            log_prob = dist.log_prob(action_sample)

            # calculate the actor loss
            actor_loss = -1 * log_prob * advantage.detach()

            # optimize the actor
            optim_actor.zero_grad()
            actor_loss.backward()
            optim_actor.step()

            ep_return += rew
            obs = next_obs
            obs_tensor = next_obs_tensor

            # update statistics
            if(step_count % 100_000 == 0) and step_count > 0:
                sr.evaluation_checkpoint()
            progress_bar.update()
        # update statistics        
        sr.update_episode_reward(ep_return)

env = gym.make("LunarLander-v3")
input_size = env.observation_space.shape[0]
num_actions = env.action_space.n
actor = Actor(input_size, num_actions)
critic = Critic(input_size)
stat_recorder = StatisticsRecorder()

train(env, actor, critic, stat_recorder, nb_steps=2_000_000)
env.close()