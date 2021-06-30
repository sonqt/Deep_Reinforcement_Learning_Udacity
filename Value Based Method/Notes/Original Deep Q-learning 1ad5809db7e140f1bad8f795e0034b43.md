# Original Deep Q-learning

# 1. Algorithm and Notes

![Original%20Deep%20Q-learning%201ad5809db7e140f1bad8f795e0034b43/Untitled.png](Original%20Deep%20Q-learning%201ad5809db7e140f1bad8f795e0034b43/Untitled.png)

The main idea behind this algorithm is using the nonlinear function performed by `CNN+ReLU+FFN` to estimate the Q (state-action value) function in Reinforcement Learning.

There are 2 novel points in this algorithm to take care about:

- ***Experience Replay***

    At first, the algorithm starts with initializing the memory $D$ to store $N$ transitions, which is randomized and updated every step. 

    The newly randomized action in the transition (state+action ⇒ reward+new_state) is selected with the continuingly updated policy (this can be $\epsilon$-greedy or greedy policy).

    The memory $D$ is then used as the labeled data in training the CNN network to estimate the  function, but this is only within a step, after that step, the memory would be updated.

- ***Target Network***

    The main idea behind this target network is to prevent the CNN network (which is the core of deep Q-learning) to diverge during the training session.

    This target network is used to estimate the future reward (from step $j+1$). Plus with the reward $r$ at step $j$, it give the estimate Q-value for the action chosen by the policy $(y_j)$. This  Q-value is then passed into the loss function of the CNN network, the effort to minimize the mean-squared error loss would then optimize the performance of the deep network in estimating Q-values.

    This network is updated every step. In other word, if we consider each time the parameters in CNN network update it produce a new network and then we store all those networks into a list, the current network is at index $i$, then the index of current target network is $i-1$  

# 2. Deep-Q Practice

- dqn_agent.py

    ```python
    import numpy as np
    import random
    from collections import namedtuple, deque

    from model import QNetwork

    import torch
    import torch.nn.functional as F
    import torch.optim as optim

    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 5e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class Agent():
        """Interacts with and learns from the environment."""

        def __init__(self, state_size, action_size, seed):
            """Initialize an Agent object.
            
            Params
            ======
                state_size (int): dimension of each state
                action_size (int): dimension of each action
                seed (int): random seed
            """
            self.state_size = state_size
            self.action_size = action_size
            self.seed = random.seed(seed)

            # Q-Network
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

            # Replay memory
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
            # Initialize time step (for updating every UPDATE_EVERY steps)
            self.t_step = 0
        
        def step(self, state, action, reward, next_state, done):
            # Save experience in replay memory
            self.memory.add(state, action, reward, next_state, done)
            
            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

        def act(self, state, eps=0.):
            """Returns actions for given state as per current policy.
            
            Params
            ======
                state (array_like): current state
                eps (float): epsilon, for epsilon-greedy action selection
            """
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            # Epsilon-greedy action selection
            if random.random() > eps:
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))

        def learn(self, experiences, gamma):
            """Update value parameters using given batch of experience tuples.

            Params
            ======
                experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
                gamma (float): discount factor
            """
    				states, actions, rewards, next_states, dones = experiences
            
    				# Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            # Compute Q targets for current states 
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states).gather(1, actions)

            # Compute loss
            loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

        def soft_update(self, local_model, target_model, tau):
            """Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target

            Params
            ======
                local_model (PyTorch model): weights will be copied from
                target_model (PyTorch model): weights will be copied to
                tau (float): interpolation parameter 
            """
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    ```

    - buffer

        ```python
        class ReplayBuffer:
            """Fixed-size buffer to store experience tuples."""

            def __init__(self, action_size, buffer_size, batch_size, seed):
                """Initialize a ReplayBuffer object.

                Params
                ======
                    action_size (int): dimension of each action
                    buffer_size (int): maximum size of buffer
                    batch_size (int): size of each training batch
                    seed (int): random seed
                """
                self.action_size = action_size
                self.memory = deque(maxlen=buffer_size)  
                self.batch_size = batch_size
                self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
                self.seed = random.seed(seed)
            
            def add(self, state, action, reward, next_state, done):
                """Add a new experience to memory."""
                e = self.experience(state, action, reward, next_state, done)
                self.memory.append(e)
            
            def sample(self):
                """Randomly sample a batch of experiences from memory."""
                experiences = random.sample(self.memory, k=self.batch_size)

                states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
                actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
                rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
                next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
                dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
          
                return (states, actions, rewards, next_states, dones)

            def __len__(self):
                """Return the current size of internal memory."""
                return len(self.memory)
        ```

- model.py

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class QNetwork(nn.Module):
        """Actor (Policy) Model."""

        def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
            """Initialize parameters and build model.
            Params
            ======
                state_size (int): Dimension of each state
                action_size (int): Dimension of each action
                seed (int): Random seed
            """
            super(QNetwork, self).__init__()
            self.seed = torch.manual_seed(seed)
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.fc3 = nn.Linear(fc2_units, action_size)

        def forward(self, state):
            """Build a network that maps state -> action values."""
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    ```

- run agent in notebook

    ```python
    def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """Deep Q-Learning.
        
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            state = env.reset()
            score = 0
            for t in range(max_t):
                action = agent.act(state, eps)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break 
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
                break
        return scores

    scores = dqn()

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    ```

## 2.1 Code Explanation

There are 2 main classes to take care in this implementation: `Agent` and `ReplayBuffer` 

 `ReplayBuffer` :

Hyper-parameters:

- `action_size` number of different actions the agent can take
- `memory` to store most recent `buffer_size` samples
- `batch_size` number of samples to take from the memory for training local network
- `experience`

Functions:

- `add` to add newly taken action into the memory
- `sample` to take `batch_size` samples from memory for training local network

`Agent` :

Hyper-parameters:

- `state_size` dimension of state in the environment, which is also the input of networks
- `action_size` number of different actions the agent can take
- `qnetwork_local`
- `qnetwork_target`
- `optimizer` of local network
- `memory` A `ReplayBuffer` object

Functions:

- `act` return `action` which will be passed into the `env.act(action)` to get `reward` , `next_state` , `done`  all of which will be passed into `step` to store into `memmory`
- `step` store newly generated samples and will call `learn` every $N$ times
- `learn` calculate `loss_function` optimize the local network and will call `soft_update` to update the `qnetwork_local` and `qnetwork_target`
- `soft_update`