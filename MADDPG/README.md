[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# Distributed Artifical Intelligence: Collaboration using MADDPG
A repository on "Distributed Artificial Intelligence" project a.a 2020/2021. Throughout the course multi-agents hold a huge of part of theory, fascinating by this concept a I used MADDPG to address the issue of Collaboration exploiting the paper "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"

![Image](https://github.com/AlessandroGulli/AI_MS_Degree/blob/main/MADDPG/images/MADDPG.png =250x)
![Image](https://github.com/AlessandroGulli/AI_MS_Degree/blob/main/MADDPG/images/MADDPG-algo.png =250x)

## Project's goal

![Trained Agent][image1]

In this project, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  In order to develop the "collaboration" aspect **the goal of each agent will be to keep the ball in play.**

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Solving the environment

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 consecutive episodes) of those **scores** is at least +0.5.

### Approach and solution

The reinforcement learning approach we use in this project is called Multi Agent Deep Deterministic Policy Gradients (MADDPG). see this [paper](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf). In this model every agent itself is modeled as a Deep Deterministic Policy Gradient (DDPG) agent (see this [paper](https://arxiv.org/pdf/1509.02971.pdf)) where, however, some information is shared between the agents.

In particular, each of the agents in this model has its own actor and critic model. The actors each receive as input the individual state (observations) of the agent and output a (two-dimensional) action. The critic model of each actor, however, receives the states and actions of all actors concatenated.

Throughout training the agents all use a common experience replay buffer (a set of stored previous 1-step experiences) and draw independent samples.
With the current set of models and hyperparameters the environment can be solved in around 3200 steps.
