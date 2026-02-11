# How to train an AI to learn to play a game?

To do that, we now use Reinforcement Learning. Reinforcement learning already existed in 1950. But it's in 2013 that Reinforcement Learning became well known after Deep Mind succesifully implement Deep Learning in Reinforcement Learning, thanks to that their agent was able to play all the attari game and with better results than humans. Todal will try to do the same thing withthe game Flappy Bird. 

To do it :
- What is Reinforcement Learning
    - Optimizing rewards
    - Policy research
    - Evaluating actions
- Techniques
    -  Policy gradient method
    -  Deep-Q-Networks
    -  Markov Decision Processes
- Our Project
    - Cart Pole V1 as a test
    - Lets implement it for Flappy Bird !

## What is Reinforcement Learning

### Rewards

In reinforcement learning, an agent operates in an environment and, based on its observations, learns to take beneficial actions. The goal is to maximize cumulative reward over time.

For example, consider a robot tasked with reaching a certain location. It receives a positive reward when it successfully moves toward the goal and a negative reward (penalty) when it fails to walk or moves away from the target.

Reinforcement learning is widely applied across many domains, including robotics, integrated systems, and finance.

![Applications of Reinforcement Learning](https://www.researchgate.net/profile/Ravi-Raj-39/publication/396046185/figure/fig1/AS:11431281657767637@1759325826537/The-list-of-applications-of-reinforcement-learning.png)

### Policy Search

The algorithm that an agent uses to determine its actions is called a **policy**. For instance, this policy can be a neural network that takes observations as input and outputs the action to execute.

In this repository, we focus on neural networks because they represent the intersection of reinforcement learning and deep learning, a more recent and powerful approach. There are several techniques for policy search, including brute force, genetic algorithms, and gradient-based methods.

![Policy Gradient Algorithm](https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/1_stochastic-policy-gradient-algorithm.jpeg)

Consider a robot with the ability to rotate. If the robot has a probability $p$ of turning right and a rotation angle ranging from $-r$ to $r$, we have two parameters: $p$ and $r$. The goal is to find the optimal policy and parameter values that maximize overall performance.

To maximize rewards, we can use **gradient ascent**, a method also known as **policy gradient**. This technique iteratively adjusts the policy parameters in the direction that increases expected reward.

#### Policy with Neural Networks

Our neural network must be structured as follows. We start with a vector of observations $(x\_1, x\_2, x\_3, x\_4)$ that represent different aspects of the environment. For example, if you are at a restaurant, these values could represent presentation, taste, price, and service quality. These observations form the input layer of the neural network.

The network then passes through hidden layers, where computations occur. Finally, the output layer represents the policy, the probability of taking a particular action. For a binary decision (e.g., order the meal or not), there is one output neuron. However, you could also have multiple outputs, such as 10 different meal options, where the highest probability determines the action taken.

The neural network is trained through experience. Over time, the agent learns to assign higher probabilities to actions that resulted in greater rewards. For instance, if you enjoyed a meal, the network learns to increase the probability of ordering it again in similar situations.

### Evaluating Actions

The agent only receives feedback from the environment in the form of rewards, which are typically sparse and delayed. Consider the CartPole game in OpenAI Gym, where the goal is to balance a pole on a moving cart. If the agent fails, which action was responsible for the failure? This is known as the credit assignment problem.

To resolve this issue, we evaluate actions by considering all subsequent rewards using a discount factor $\gamma$. The cumulative reward at time $t$ is calculated as the sum of all future "raw" rewards, each weighted by the discount factor raised to increasing powers.

Initially, we might write:

$$R_t = \gamma \cdot w_t + \gamma \cdot w_{t+1} + \gamma \cdot w_{t+2} + \dots$$

However, to give more importance to immediate rewards (since they are more reliable and certain), we apply increasing powers of the discount factor:

$$R_t = \gamma^0 \cdot w_t + \gamma^1 \cdot w_{t+1} + \gamma^2 \cdot w_{t+2} + \dots$$

This can be expressed more compactly as:

$$R_t = \sum_{k=0}^{n} \gamma^k \cdot w_{t+k}$$

The discount factor $\gamma$ typically ranges from 0 to 1. A value close to 1 means the agent values future rewards almost equally to immediate rewards, while a value close to 0 makes the agent prioritize immediate rewards. This approach ensures that recent decisions are held more accountable for their outcomes, helping to solve the credit assignment problem. generally the value is between 0.95 and 0.99.
for $\gamma$ = 0.95 : 0.95^13 more or less 0.5 (rewards beyond the 13 step worth for 50%)
for $\gamma$ = 0.99 : 0.99^69 more or less  0.5 (rewards beyond the 69 step worth for 50%)