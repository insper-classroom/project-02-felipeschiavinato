import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = GradScaler()

class PointerNetwork(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        
        super(PointerNetwork, self).__init__()
        self.embedding = torch.nn.Linear(input_dim, embedding_dim)
        self.encoder = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = torch.nn.LSTM(embedding_dim+1, hidden_dim, batch_first=True)
        self.pointer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, last_idx, device: torch.device):
        batch_size, seq_len, _ = x.size()
        x = self.embedding(x)
        _, (hidden, cell) = self.encoder(x)
        last_idx_one_hot = torch.zeros(batch_size, seq_len).to(device)
        last_idx_one_hot.scatter_(1, last_idx, 1)
        decoder_input = torch.cat([x, last_idx_one_hot.unsqueeze(-1)], dim=-1)  # shape: (batch_size, seq_len, embedding_dim + 1)
        decoder_out, _ = self.decoder(decoder_input, (hidden, cell))
        scores = self.pointer(decoder_out)
        return scores.squeeze(-1)
    
class Agent:
    def __init__(self, input_dim, embedding_dim, hidden_dim, lr=1e-4):
        self.network = PointerNetwork(input_dim, embedding_dim, hidden_dim).to(device)
        self.network = torch.jit.script(self.network)  # JIT compilation
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def select_action(self, state):
        batch_size, sequence_length, _ = state.size()
        start_idx = torch.zeros(batch_size, 1).long().to(device)
        log_probs = []
        actions = []
        mask = torch.zeros(batch_size, sequence_length).to(device)

        for _ in range(sequence_length):
            scores = self.network(state, start_idx, device)
            scores = scores.masked_fill(mask.bool(), float('-inf'))
            probs = F.softmax(scores, dim=-1)
            m = Categorical(probs)
            selected_action = m.sample()
            
            actions.append(selected_action)
            log_probs.append(m.log_prob(selected_action))
            mask = mask.scatter(1, selected_action.unsqueeze(1), 1.0)
            start_idx = selected_action.unsqueeze(1)

        actions = torch.stack(actions, dim=1)
        log_probs = torch.stack(log_probs, dim=1)

        return actions, log_probs

    def update(self, rewards, log_probs):
        loss = []
        for reward, log_prob in zip(rewards, log_probs):
            loss.append(-reward * log_prob)
        self.optimizer.zero_grad()
        
        loss = torch.cat(loss).sum()

        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()

# Hyperparameters
input_dim = 2  
embedding_dim = 128
hidden_dim = 256
lr = 1e-4
n_cities = 23
n_epochs = 1000000
early_stopping_patience = 1000000  # Early stopping patience
no_improve_epochs = 0  # Initialize the counter of epochs without validation improvement
warm_up_episodes = 0  # Number of episodes to use only heuristic
agent = Agent(input_dim, embedding_dim, hidden_dim, lr)
best_agent = deepcopy(agent)  # Initialize the best agent

# Training loop
rewards = []
test = [
    [26.0325, 50.5106],
    [21.5433, 39.1728],
    [-37.8497, 144.968],
    [40.3725, 49.8532],
    [25.7617, -80.1918],
    [44.3439, 11.7167],
    [43.7384, 7.4246],
    [41.57, 2.2611],
    [45.5017, -73.5673],
    [47.2197, 14.7647],
     [52.0786, -1.0169],
     [47.5839, 19.2486],
     [50.4372, 5.9714],
     [52.3886, 4.5446],
     [45.6156, 9.2811],
     [1.2914, 103.864],
     [34.8431, 136.541],
     [25.4207, 51.4700],
     [30.1328, -97.6411],
     [19.4326, -99.1332],
     [-23.5505, -46.6333],
     [36.1699, -115.1398],
     [24.4672, 54.6033]
]

distance_matrix = torch.zeros((n_cities, n_cities), device=device)

# Precompute all distances
for i in range(n_cities):
    for j in range(n_cities):
        if i != j:
            distance_matrix[i, j] = haversine_distance(torch.tensor(test[i], device=device).unsqueeze(0), 
                                                         torch.tensor(test[j], device=device).unsqueeze(0))
            print( f'distance_matrix[{i}][{j}] = {distance_matrix[i][j]}')

# We will add a scale factor for the reward
reward_scale = 1e-3
best_reward = float('-inf')  # Initialize the best reward with high value
prev_reward = float('-inf')  # Initialize the previous reward with high value

state = torch.FloatTensor(test).unsqueeze(0).to(device)
heuristic = nearest_neighbor(state).unsqueeze(0)

heuristic_weight = 1  # Weight of the heuristic reward
distance_weight = 0  # Initialize the heuristic reward

for epoch in range(n_epochs):
    with autocast():
        action, log_prob = agent.select_action(state)
        n_points = action.shape[1]
        distances = torch.empty(n_points - 1).to(device)
        reward_heuristic = 0
        if epoch < warm_up_episodes:
            score = 0
            for i in range(23):
                if action[0][i] == heuristic[0][i]:
                    score += 1
            score = score / 23
            reward_heuristic = ((score)*100)-200

        for i in range(n_points - 1):
            # Index into the distance matrix for the precomputed distance
            distances[i] = distance_matrix[action[0, i].item(), action[0, i+1].item()]

        # Fetch the distance from the last to the first point
        last_to_first = distance_matrix[action[0, -1].item(), action[0, 0].item()]

        reward_distance = -(torch.sum(distances) + last_to_first)

        # Combine the two rewards
        reward = (reward_distance*reward_scale)#*distance_weight + (reward_heuristic)*heuristic_weight

    agent.update([reward], [log_prob])
    rewards.append(reward.item())

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(agent.network.parameters(), 1.0)

    # Print and monitor the progress
    if epoch % 10 == 0:
        window_size = 10
        weights = np.repeat(1.0, window_size)/window_size
        sma = np.convolve(rewards, weights, 'valid')
        print("Moving average of last 10 rewards: ", sma[-1])
        print(f"Action: {action}")
        print(f"Epoch: {epoch+1}, Reward: {reward.item()}")
        print(f"best_reward: {best_reward}")
        print(f"no_improve_epochs: {no_improve_epochs}")
        # print(f"Heuristic: {heuristic}")

    # Early stopping condition
    if reward.item() > best_reward:
        best_reward = reward.item()
        best_agent = deepcopy(agent)  # Save the best agent
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= early_stopping_patience:
            print("Early stopping triggered. Stop training.")
            break
    heuristic_weight = heuristic_weight - 0.00001
    distance_weight = distance_weight + 0.00001

agent = best_agent  # Use the best agent

import pickle

# Save model
torch.save(best_agent.network.state_dict(), 'pointer_network_model_1.pth')

# Save rewards
with open('rewards_list_1.pkl', 'wb') as f:
    pickle.dump(rewards, f)


# Plot rewards per episode
plt.plot(rewards)
plt.title("Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
