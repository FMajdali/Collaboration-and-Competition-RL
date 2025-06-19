# Collaboration-and-Competition-RL
This is a solution for the Deep Reinforcement Learning nanodegree project "Collaboration and Competition" from Udacity

The code used in this solution is adopted from the Udacity repository below, and I have included the MIT license in all files that I have modified.
https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal

# Repository files:
- Tennis.ipynb: The Jupyter notebook provided by Udacity for this project, contains the initial instructions and the training loop that initializes the agents and networks and starts the learning process.
- model.py: contains the neural network architecture for both Actor and Critic
- ddpg_agent.py: contains a class named Agent, which has all the required methods to train the agent
- checkpoint_critic.pth: contains the trained weights for the critic network
- checkpoint_actor.pth: contains the trained weights for the actor network

# Python Requirements:
- Pillow>=4.2.1
- matplotlib
- numpy>=1.11.0
- jupyter
- pytest>=3.2.2
- docopt
- pyyaml
- protobuf>=3.5.2
- grpcio>=1.11.0
- torch
- pandas
- scipy
- ipykernel
- protobuf==3.20.3

# Environment:
The environment is provided by Unity with the name "Tennis", In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play. 

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. 

Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 

Specifically:
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.
The environment is considered solved when the average (over 100 episodes) of those scores is at least +0.5.

To download the Unity Tennis environment:
- Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
- Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip
- Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip
- Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip


# MDDPG:
MDDPG is a variation of DDPG or Deep Deterministic Policy Gradient for multi-agent enviroments, MDDPG is the algorithm used in this project to solve the enviroment. DDPG is a different kind of actor-critic method, and it could be seen as an approximate DQN instead of an actual actor-critic, because the critic in DDPG is used to approximate the maximizer over the Q values of the next state and not as a baseline.
One of the limitations of the DQN agent is that it is not straightforward to use in continuous action spaces
In DDPG we use two deep neural networks, we call one the actor and the other one is the critic:
-  Actor: takes the state S as input, and outputs the optimal policy deterministically, the output is the best believed action for any given state unlike the discrete case where the output is a probability distribution over all possible actions, the actor is basically learning the argmax Q(s,a)
-  Critic: takes the input state S, then it tries to calculate the optimal action-value function by using the actor's best-believed action
-  
Two interesting aspects of DDPG are:
- The use of a replay buffer
- Soft updates to the target networks:
    - In DQN you have two copies of the network weights: the regular network and the target network, where the target network gets a big update every n-step by simply copying the weights of the regular network into the target network
    - IN DDPG, you have two copies of the network weights for each network, a regular for the critic an irregular for the critic, a target for the actor, and a target for the critic, but in DDPG the target networks are updated using a soft updates strategy which is slowly blending the regular network weights with the target network weights, so every time step you mix in 0.01% of regular network weights with target network weights
 
# DDPG Chosen Hyperparameters:
- Replay buffer size: 100000 
- Minibatch size: 128
- Discount factor: 0.99
- TAU for soft update of target parameters: 0.001
- Learning rate of the actor: 0.001
- Learning rate of the critic: 0.001

# Neural Networks Architecture:
- Actor Network consists of:
    - Fully connected layer which takes the state with the size 24, and outputs 256
    - Fully connected layer which takes the ReLU of the output of the previous layer and outputs 2 logits, each representing an action
    - The output from the previous layer is passed through a tanh to ensure the output actions are between -1 and 1 

- Critc Network consists of:
    - Fully connected layer which takes the state with the size 24, and outputs 256
    - Fully connected layer which takes the Leaky ReLU of the output of the previous layer along the actions chosen by the actor with size 260 and outputs 256
    - Fully connected layer which takes the Leaky ReLU of the output of the previous layer and outputs 128
    - Fully connected layer which takes the Leaky ReLU of the output of the previous layer and outputs 1 logit which resembles the State-Value

# Solution:

The code is adopted from Udacity but with the following changes (adapted to the multi-agent nature of Tennis env):
- The same actor networkis used for both agents, the actor network receives the observation of one agent (24 elements) and outputs the deterministic two actions for that agent to take
- Created a new method for the agent class "update_memory" to update the replay memory using experiences from the 2 agents, it separates each agent experience tuple and stores it separately 
- Modified the "step" method in the agent class by removing the update memory from it
- Modified the training loop to adapt with the Unity environment, also made the updates to occur every 10 time-steps, and it updates the networks 10 times

**The environment was solved in 3671 episodes, as the graph below indicates**
![alt text](https://github.com/FMajdali/Collaboration-and-Competition-RL/blob/main/MDDPG%20Training%20Graph.jpg))


# Future Ideas:
- Another approach could be used to solve the problem like A2C, A3C, or D4PG.
- Another hyperparameter could be tested to see if there is any enhancement to the model convergence and scores
- I could not find any documentation for the Tennis environment, the idea is that it is possible to transform observations gained by agent 2 from the perspective of agent 1, the transformed observation will be fed then into the actor-critic model and the actions produced by it will be inverted to be usable by agent 2, this approach will speed up the learning as all experiences are from one perspective and will produce self-play agent
