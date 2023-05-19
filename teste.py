import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PointerNetwork(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(PointerNetwork, self).__init__()
        self.embedding = torch.nn.Linear(input_dim, embedding_dim)
        self.encoder = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.pointer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, last_idx):
        x = self.embedding(x)
        _, (hidden, cell) = self.encoder(x)
        # decoder_input = x.gather(1, last_idx.unsqueeze(-1).unsqueeze(-1).expand(x.size(0), 1, x.size(-1))).to(device)
        decoder_out, _ = self.decoder(x, (hidden, cell))
        scores = self.pointer(decoder_out)
        return scores.squeeze(-1)


class Agent:
    def __init__(self, input_dim, embedding_dim, hidden_dim, lr=1e-3):
        self.network = PointerNetwork(input_dim, embedding_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def select_action(self, state):
        batch_size, sequence_length, _ = state.size()
        start_idx = torch.zeros(batch_size, dtype=torch.long).to(device)  # start from the first city
        log_probs = []
        actions = []
        mask = torch.zeros(batch_size, sequence_length).to(device)

        count = 0
        for _ in range(sequence_length):
            
            scores = self.network(state, start_idx)
            print(f"Scores: {scores}")
            scores = scores.masked_fill(mask.bool(), float('-inf'))
            print(f"Scores: {scores}")
            probs = F.softmax(scores, dim=-1)
            print(f"Probs: {probs}")
            
            m = Categorical(probs)
            selected_action = m.sample()
            # if count == 0:
            #     selected_action = torch.zeros(batch_size, dtype=torch.long).to(device)
            #     count += 1
            # elif count == 1:
            #     selected_action = torch.ones(batch_size, dtype=torch.long).to(device)
            #     count += 1
            print(f"Selected Action: {selected_action}")
            actions.append(selected_action)
            log_probs.append(m.log_prob(selected_action))
            mask = mask.scatter(1, selected_action.unsqueeze(1), 1.0)
            start_idx = selected_action

        actions = torch.stack(actions, dim=1)
        print(actions)
        log_probs = torch.stack(log_probs, dim=1)
        # actions = torch.cat([actions, actions[:, 0:1]], dim=1)  # add first action to the end
        # log_probs = torch.cat([log_probs, log_probs[:, 0:1]], dim=1)  # add log prob of first action to the end

        return actions, log_probs

    def update(self, rewards, log_probs):
        loss = []
        for reward, log_prob in zip(rewards, log_probs):
            loss.append(-reward * log_prob)
        self.optimizer.zero_grad()
        
        loss = torch.cat(loss).sum()
        
        loss.backward()
        self.optimizer.step()

# Hyperparameters
input_dim = 2  # x, y coordinates
embedding_dim = 128
hidden_dim = 256
lr = 1e-3
n_cities = 7
n_epochs = 10000

# Initialize agent
agent = Agent(input_dim, embedding_dim, hidden_dim, lr)

# Training loop
rewards = []
test = [
    [0.0, 0.0],
    [1.0, 1.0],
    [3.0, 1.0],
    [4.0, 0.0],
    [4.0, -1.0],
    [2.0, -2.0],
]
random = np.random.uniform(size=(n_cities, input_dim))

min_reward = -1000
for epoch in range(n_epochs):
    
    state = torch.FloatTensor(test).unsqueeze(0).to(device)
    
    action, log_prob = agent.select_action(state)
    #
    distances = torch.sqrt(torch.sum((state[0, action[:, :-1]] - state[0, action[:, 1:]])**2, dim=-1))
    #print(f"Distances: {distances}")
    last_to_first = torch.sqrt(torch.sum((state[0, action[:, -1]] - state[0, action[:, 0]])**2, dim=-1))
    #print(f"Last to First: {last_to_first}")
    reward = -(torch.sum(distances) + last_to_first)

    # if reward.item() < min_reward:
    #     reward = torch.FloatTensor([-1000]).to(device)
    # else:
    #     min_reward = reward.item()

    #reward = -torch.sum(torch.sqrt(torch.sum((state[0, action[:, :-1]] - state[0, action[:, 1:]])**2, dim=-1)))

    agent.update([reward], [log_prob])
    rewards.append(reward.item())
    if(reward.item() > -11.0):
        print("-"*50)
    print(f"Action: {action}")
    print(f"Epoch: {epoch+1}, Reward: {reward.item()}")
    if(reward.item() > -11.0):
        print("-"*50)

# Plot rewards per episode
plt.plot(rewards)
plt.title("Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
