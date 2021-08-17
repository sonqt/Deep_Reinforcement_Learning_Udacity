# Project 2: Continuous Control

***Note***: read `README.md` first to gain some intuition about this project. In this file, `Report.md` I will only focus on the **implementation** of my idea, **results** and some **ideas** to **improve** the performance of the agent.

## 1. Implementation

The core idea that help my implemented agent to successfully solve this complicated environment is [Deep Deterministic Policy Gradient]([1509.02971.pdf (arxiv.org)](https://arxiv.org/pdf/1509.02971.pdf)), a reinforcement learning algorithm first introduced by Deepmind (Lillicrap et al.) in 2015.

### Deep Deterministic Policy Gradient

According to Lillicrap et al., there are 4 core neural networks in DDPG algorithm. We have to first randomly initialize the critic network (`critic_local`) and actor network (`actor_local`). In this project I use the initialization method described in Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015), using a uniform distribution. Then, the two other networks are `critic_target` and `actor_target`.

The main idea behind this target network is to prevent the Deep Neural Network (which is the core of deep Q-learning) to diverge during the training session. This target network is used to estimate the future reward (from step ***j+1***). Plus with the reward ***r*** at step ***j***, it give the estimate Q-value for the action chosen by the policy ***(y_j)***. This Q-value is then passed into the loss function of the Deep Neural Network, the effort to minimize the mean-squared error loss would then optimize the performance of the deep network in estimating Q-values. This network is updated every step. In other word, if we consider each time the parameters in Deep Neural Network update it produce a new network and then we store all those networks into a list, the current network is at index ***i***, then the index of current target network is ***i-1***.

Besides, this algorithm also use Experience Replay: 

- At first, the algorithm starts with initializing the memory $D$ to store $N$ transitions, which is randomized and updated every step.

  The newly randomized action in the transition (state+action ⇒ reward+new_state) is selected with the continuingly updated policy (this can be $\epsilon$-greedy or greedy policy).

  The memory $D$ is then used as the labeled data in training the Deep Neural Network to estimate the function, but this is only within a step, after that step, the memory would be updated.

## 2. Hyperparameters

| Hyperparameter                                    | Value |
| ------------------------------------------------- | ----- |
| Replay buffer size                                | 1e6   |
| Batch size (number of samples to get from Replay) | 512   |
| $\gamma$ (discount factor of expected reward)     | 0.99  |
| $\tau$ (update the target networks)               | 1e-3  |
| Actor Learning rate                               | 2e-4  |
| Critic Learning rate                              | 3e-4  |
| Number of max training episodes                   | 2000  |
| Max number of timesteps per episode               | 1000  |

***<u>Note:</u>*** For more information about models' architecture, read `model.py`

## 3. Model Architecture

### 3.1 Actor

| Layer               |     # units     | Activation function |
| ------------------- | :-------------: | :-----------------: |
| Batch Normalization |        _        |          _          |
| Input layer         | state_size (33) |          _          |
| First hidden layer  |       256       |        ReLU         |
| Second hidden layer |       128       |        ReLU         |
| Output layer        | action_size (4) |        Tanh         |

### 3.2 Critic

| Layer               |        # units        | Activation function |
| ------------------- | :-------------------: | :-----------------: |
| Batch Normalization |           _           |          _          |
| Input layer         |    state_size (33)    |          _          |
| First hidden layer  |          256          |     Leaky_ReLU      |
| Concatenation       | 256+action_size (260) |          _          |
| Second hidden layer |          128          |     Leaky_ReLU      |
| Third hidden layer  |          128          |     Leaky_ReLU      |
| Output layer        |           1           |          _          |

***<u>Note:</u>*** All Leaky_ReLU function in Critic use the default `negative_slope` of Pytorch (0.01).

## 4. Result

![image-20210818000140313](..\Project\score.png)

The average reward (over 100 episodes) of the trained agent reached +30 after 190 episodes.

**<u>*Note:*</u>** This is also reported in `Continuous_Control.ipynb`.

## 5. Further Improvement

- Some recommended algorithms by Udacity teaching team: TRPO, PPO, A3C, A2C
- Some small techniques like [Batch Normalization]([1502.03167.pdf (arxiv.org)](https://arxiv.org/pdf/1502.03167.pdf)) have helped a lot in training process. I will try some other methods like [Weight Normalization]([[1602.07868\] Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks (arxiv.org)](https://arxiv.org/abs/1602.07868)) or other method of weight initialization.
- [Prioritized Replay](https://arxiv.org/pdf/1511.05952.pdf) will also work really well in solving this problem.