
**Breakout - A3C**

  

**Abstract**: The challenges of applying reinforcement learning to modern AI applications are interesting, particularly in unknown environments in which there are delayed rewards. Classic arcade games have garnered considerable interest recently as a test bed for these kinds of algorithms. _Breakout_ is an arcade game developed and published by Atari, Inc., and released on May 13, 1976. Our goal in this project is to use Reinforcement learning to train AI Agent to play Breakout game.

**Introduction:** Reinforcement learning (RL) is currently one of the most active areas in Artificial Intelligence research. It is the technique by which an agent learns how to achieve rewards r through interactions with its environment. Many real-word applications such as robotics and autonomous cars are particularly well-suited for a RL approach as the environment is unknown and the consequences of actions are uncertain. Through trial-and-error and experience over time, an RL agent learns a mapping of states s to optimal actions a and develops a policy to achieve long-term rewards. However, several challenges face the agent as it tries to learn this policy. The environment often provides delayed rewards for actions, making it difficult for the agent to learn which actions correspond to which rewards. Furthermore each action the agent takes can impact whats optimal later on, sometimes unpredictably. Even if the agent does learn a policy that allows it to achieve rewards, there is still the question of whether the policy is an optimal one. Thus, the agent must make a trade-off between exploring possibly sub-optimal actions with the hope that it may find a more optimal strategy and exploiting its existing policy.

![](https://miro.medium.com/max/1392/1*YtnGhtSAMnnHSL8PvS7t_w.png)

**Algorithm:** Asynchronous Advantage Actor-Critic (A3C)

**Asynchronous:** Unlike [DQN](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df), where a single agent represented by a single neural network interacts with a single environment, A3C utilizes multiple incarnations of the above in order to learn more efficiently. In A3C there is a global network, and multiple worker agents which each have their own set of network parameters. Each of these agents interacts with it’s own copy of the environment at the same time as the other agents are interacting with their environments. The reason this works better than having a single agent (beyond the speedup of getting more work done), is that the experience of each agent is independent of the experience of the others. In this way the overall experience available for training becomes more diverse.

**Actor-Critic:** So far this series has focused on value-iteration methods such as Q-learning, or policy-iteration methods such as Policy Gradient. Actor-Critic combines the benefits of both approaches. In the case of A3C, our network will estimate both a value function V(s) (how good a certain state is to be in) and a policy π(s) (a set of action probability outputs). These will each be separate fully-connected layers sitting at the top of the network. Critically, the agent uses the value estimate (the critic) to update the policy (the actor) more intelligently than traditional policy gradient methods.

**Advantage:** The update rule used the discounted returns from a set of experiences in order to tell the agent which of its actions were “good” and which were “bad.” The network was then updated in order to encourage and discourage actions appropriately.

Discounted Reward: R = γ(r)

The insight of using advantage estimates rather than just discounted returns is to allow the agent to determine not just how good its actions were, but how much better they turned out to be than expected. Intuitively, this allows the algorithm to focus on where the network’s predictions were lacking. If you recall from the [Dueling Q-Network architecture](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df), the advantage function is as follow:

Advantage: A = Q(s,a) - V(s)

Since we won’t be determining the Q values directly in A3C, we can use the discounted returns (R) as an estimate of Q(s,a) to allow us to generate an estimate of the advantage.

Advantage Estimate: A = R - V(s)

  

  

Model:

# AI for Breakout

# Importing the librairies

import numpy as np

import torch

import torch.nn as nn

import torch.nn. as F

# Initializing and setting the variance of a tensor of weights

def normalized_columns_initializer(weights, std=1.0):

out = torch.randn(weights.size())

out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))

return out

# Initializing the weights of the neural network in an optimal way for the learning

def weights_init(m):

classname = m.__class__.__name__

if classname.find('Conv') != -1:

weight_shape = list(m.weight.data.size())

fan_in = np.prod(weight_shape[1:4])

fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]

w_bound = np.sqrt(6. / (fan_in + fan_out))

m.weight.data.uniform_(-w_bound, w_bound)

m.bias.data.fill_(0)

elif classname.find('Linear') != -1:

weight_shape = list(m.weight.data.size())

fan_in = weight_shape[1]

fan_out = weight_shape[0]

w_bound = np.sqrt(6. / (fan_in + fan_out))

m.weight.data.uniform_(-w_bound, w_bound)

m.bias.data.fill_(0)

  

  

  

  

  

  

  

# Making the A3C brain

  

class ActorCritic(torch.nn.Module):

  

def __init__(self, num_inputs, action_space):

super(ActorCritic, self).__init__()

self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)

self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

num_outputs = action_space.n

self.critic_linear = nn.Linear(256, 1)

self.actor_linear = nn.Linear(256, num_outputs)

self.apply(weights_init)

self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)

self.actor_linear.bias.data.fill_(0)

self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0)

self.critic_linear.bias.data.fill_(0)

self.lstm.bias_ih.data.fill_(0)

self.lstm.bias_hh.data.fill_(0)

self.train()

  

def forward(self, inputs):

inputs, (hx, cx) = inputs

x = F.elu(self.conv1(inputs))

x = F.elu(self.conv2(x))

x = F.elu(self.conv3(x))

x = F.elu(self.conv4(x))

x = x.view(-1, 32 * 3 * 3)

hx, cx = self.lstm(x, (hx, cx))

x = hx

return self.critic_linear(x), self.actor_linear(x), (hx, cx)

  

**Output:**

[https://github.com/xoraus/Breakout-A3C-Algorithm/blob/master/Model%20Output%20Images/1.AgentScore541.png](https://github.com/xoraus/Breakout-A3C-Algorithm/blob/master/Model%20Output%20Images/1.AgentScore541.png)

  

  

  

  

  

  

  

  

  

  

  

  

  

  

  

  

  

**References:**

[1] [https://arxiv.org/pdf/1506.02438.pdf](https://arxiv.org/pdf/1506.02438.pdf)

[2] [https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/](https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/)

[3] [https://en.wikipedia.org/wiki/Breakout_(video_game)](https://en.wikipedia.org/wiki/Breakout_(video_game))

[4] [https://cs.stanford.edu/~rpryzant/data/rl/paper.pdf](https://cs.stanford.edu/~rpryzant/data/rl/paper.pdf)

[5] [https://www.semanticscholar.org/paper/Reinforcement-Learning-for-Atari-Breakout-Berges-Rao/b95a573d1ed948a9d423d1a0c276d220bf913d71](https://www.semanticscholar.org/paper/Reinforcement-Learning-for-Atari-Breakout-Berges-Rao/b95a573d1ed948a9d423d1a0c276d220bf913d71)

[6] [https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

[7] [https://arxiv.org/pdf/1607.05077](https://arxiv.org/pdf/1607.05077)
