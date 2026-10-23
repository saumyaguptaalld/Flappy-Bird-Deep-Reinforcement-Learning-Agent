# Flappy-Bird-Deep-Reinforcement-Learning-Agent
DQN teaches an agent to play a game by learning a function `Q(state, action)` that predicts how good each action is in a given state. Since Flappy Bird's state space is continuous (bird height, velocity, pipe distances, etc.), we can't store this in a lookup table — so a neural network approximates it instead.
