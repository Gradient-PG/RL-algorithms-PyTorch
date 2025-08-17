import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.buffers import ReplayBuffer
import ale_py
import random

torch.cuda.is_available()


class DQN(nn.Module):
    def __init__(self, nb_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, nb_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)

def test_replay_buffer(rb):
    data = rb.sample(batch_size = 1)

    print(type(data.observations))

    obs = data.observations.detach().cpu().numpy().squeeze()

    frame = obs[0, :, :]

    # Display the grayscale frame
    plt.imshow(frame, cmap='gray')
    plt.title("Preprocessed Grayscale Frame")
    plt.savefig("preprocessed_obs")
    plt.close()

    next_obs = data.next_observations.detach().cpu().numpy().squeeze()

    frame = next_obs[0, :, :]

    # Display the grayscale frame
    plt.imshow(frame, cmap='gray')
    plt.title("Preprocessed Grayscale Frame")
    plt.savefig("preprocessed_next_obs")
    plt.close()
    
    print(f"action: {data.actions}")
    print(f"reward: {data.rewards}")
    print(f"done: {data.dones}")



def optimize(batch_size, discount_factor, rb, q_network, target_network, optimizer):
    data = rb.sample(batch_size)
    with torch.no_grad():
        max_q_value, _ = target_network(data.next_observations).max(dim=1)
        y = data.rewards.flatten() + discount_factor * max_q_value * (
            1 - data.dones.flatten()
        )
    current_q_value = q_network(data.observations).gather(1, data.actions).squeeze()
    loss = nn.MSELoss()(current_q_value, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def Deep_Q_Learning(
    env,
    device="cuda",
    nb_steps = 30_000_000,
    learning_rate=0.001,
    replay_memory_size=1_000_000,
    initial_exploration=1,
    final_exploration=0.01,
    exploration_steps=1_000_000,
    target_update_freq = 5,
    train_frequency = 4,
    batch_size = 32,
    discount_factor=0.99,
    replay_start_size = 80_000
):
    # Initialize replay memory D to capacity N
    rb = ReplayBuffer(
        replay_memory_size,
        env.observation_space,
        env.action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False
        
    )

    q_network = DQN(env.action_space.n).to(device)
    target_net = DQN(env.action_space.n)
    target_net.load_state_dict(q_network.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)

    step = 0
    av_rewards=[]
    rewards = []

    progress_bar = tqdm(total=nb_steps)
    while step <= nb_steps:
        done = False
        total_rewards = 0

        obs, info = env.reset()

        while not done:

            epsilon = max(
                (final_exploration - initial_exploration) / exploration_steps * step
                + initial_exploration,
                final_exploration,
            )
            if random.random() < epsilon:  # With probability ε select a random action a
                action = np.array(env.action_space.sample())
            else:  # Otherwise select a = max_a Q∗(φ(st), a; θ)
                q_values = q_network(torch.Tensor(obs).unsqueeze(0).to(device))
                action = np.array(torch.argmax(q_values, dim=1).item())

            # Execute action a in emulator and observe reward rt and image xt+1
            next_obs, reward, terminated, truncated, info = env.step(action)
            progress_bar.update(1)

            done = terminated or truncated
            total_rewards += reward

            rb.add(obs, next_obs, action, reward, done, info)
            obs = next_obs

            # update target net
            if step % target_update_freq == 0 and step > 0:
                target_net.load_state_dict(q_network.state_dict())

            if step > replay_start_size and step % train_frequency == 0:
                optimize(
                    batch_size=batch_size, 
                    discount_factor=discount_factor,
                    rb=rb,
                    q_network=q_network,
                    target_network=target_net,
                    optimizer=optimizer)

            step += 1
            if (step % 50_000 == 0) and step > 0:
                av_rewards.append(np.mean(rewards))
                rewards = []
                plt.plot(av_rewards)
                plt.title("Average Reward on Breakout")
                plt.xlabel("Training Epochs")
                plt.ylabel("Average Reward per Episode")
                plt.savefig("Imgs/average_reward_on_breakout.png")
                plt.close()
                torch.save(q_network.state_dict(), "breakout_atari_ckpt.pth")
        rewards.append(total_rewards)

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
# env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayscaleObservation(env)
env = gym.wrappers.FrameStackObservation(env, 4)
env = MaxAndSkipEnv(env, skip=4)

Deep_Q_Learning(env, device="cpu", replay_memory_size=500_000)
env.close()