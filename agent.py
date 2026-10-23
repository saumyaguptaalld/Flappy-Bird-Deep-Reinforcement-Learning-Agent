# to do training write below in terminal
# C:\Users\Saumya\anaconda3\anaconda3\envs\nengo_env\python.exe agent.py flappybirdv0 --train

import gymnasium as gym
import flappy_bird_gymnasium

import random
import torch
import torch.nn as nn
import torch.optim as optim

from dqn import DQN
from experience_replay import ReplayMemory

import os
import argparse
import yaml

import itertools  # to run training indefinitely



# Choosing device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


RUN_DIR = "runs"
os.makedirs(RUN_DIR, exist_ok=True)

class Agent: # class has three functions : for parameter loading, run (for training), optimize (for loading batch trained data and calculating y_pred, loss and optimize)

    # Loading parameters from yaml file 
    def __init__(self,param_set):
        self.param_set = param_set
        with open("parameters.yaml","r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        # agent will have all parameter access so written outside with open ...
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None      
        self.LOG_FILE = os.path.join(RUN_DIR,f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUN_DIR,f"{self.param_set}.PT")
    # Training
    # During training, render is false, means , we donot want to see game window while training
    def run(self, is_training = True, render = False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=True)

        num_states = env.observation_space.shape[0]  # input dim
        num_actions = env.action_space.n             # output dim

        # calling the Deep Q network function defined in dqn.py
        policy_dqn = DQN(num_states, num_actions).to(device)

        # creating deque for storing experiences using ReplayMemory function from experience_delay.py
        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init  # initializing epsilon value for training
            # Creating target network
            target_dqn = DQN(num_states, num_actions).to(device)
            # Copy weight and bias values from policy network into target network
            target_dqn.load_state_dict(policy_dqn.state_dict())
            steps = 0  # we need to copy weights and biases after every 10 steps, so we will be counting step values by initializing step = 0
            # defining optimizer
            self.optimizer = optim.Adam(policy_dqn.parameters(),lr = self.alpha)
            best_reward = float("-inf")  # initialized with -inf, and will be updated  
        else: # this is for testing, where we load best model
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count(): # Training starts for each episode (indefinite training until cancelled from terminal or max reward threshold is reached)
            state, _ = env.reset()         # initial state
            state = torch.tensor(state, dtype= torch.float, device = device)  # convert state to tensor
            episode_reward = 0
            terminated = False             # Instead of done , we are directly using terminated

            while not terminated and episode_reward < self.reward_threshold:  # Checking if the player is still alive (means not terminated)
                if is_training and random.random()< epsilon:
                    action = env.action_space.sample()  # explore
                    action = torch.tensor(action, dtype=torch.long, device = device)  # convert to tensor and dtype is long here compared to others which used float
                else:
                    with torch.no_grad(): # since here no training is happening and we are exploiting thus, torch.no_grad and further argmax is used respectively
                         action = policy_dqn(state.unsqueeze(dim =0).float()).squeeze().argmax() # policy_dqn expects output 2D, so we unsqueeze and add one more dim, post that we again remove the added dim with squeeze 
                # Processing: We dont use truncated
                next_state, reward, terminated, _ , _ = env.step(action.item())
                episode_reward +=reward
                reward = torch.tensor(reward, dtype=torch.long, device = device)  
                next_state = torch.tensor(next_state, dtype=torch.long, device = device)  
                
                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps +=1  # increasing steps inorder to copy wts and biases for target network from policy network
                # updating state and episode reward
                state = next_state
                
            print(f"For episode: {episode+1}/{self.replay_memory_size} with total reward = {episode_reward}")

            if is_training:
                # we want to decay at every episode
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                # when episode_reward > best reward , we will save our models
                if episode_reward > best_reward:
                    log_msg = f"best reward = {episode_reward} for episode={episode+1}"
                    with open(self.LOG_FILE, "a") as f:  # opening file in append mode
                        f.write(log_msg + "\n")
                    # weight and bias value of episodes with best reward are saved 
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    # the best reward is now updated
                    best_reward = episode_reward
                
            if is_training and len(memory) > self.mini_batch_size:  # when the total number of memory or eperiences exceed the mini batch size that we have given for training
                # get_sample
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)
                # sync the networks
                    if steps > self.network_sync_rate:
                       target_dqn.load_state_dict(policy_dqn.state_dict())
                       steps = 0  # resetting steps to 0    
            # env.close() # commented because we are manually stopping training in terminal using ctrl+c

    # optimize function will train our DQN, get experience from experience relay, then compute target Q value using target network, calculate y pred using policy network, compute loss then backpropogate
    # instead of training each experience individually , we train our network in batch               
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # get batch of experiences
        state, action, next_state,reward, terminations  = zip(*mini_batch)
        state = torch.stack(state).float()
        action = torch.stack(action)
        next_state = torch.stack(next_state).float()
        reward = torch.stack(reward).float()
        terminations = torch.tensor(terminations).float().to(device)
        # Calculate target Q values - if terminations = True => zero
        with torch.no_grad():
            target_q = reward + (1-terminations)* self.gamma*target_dqn(next_state).max(dim = 1)[0]  # y_true
        # Calculate y_pred i.e. Q value from current policy
        current_q = policy_dqn(state).gather(dim =1, index = action.unsqueeze(dim = 1)).squeeze() # y_pred

        # loss
        loss = self.loss_fn(current_q, target_q)

        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
                    

if __name__ == "__main__":
    # parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model')
    parser.add_argument('hyperparameters',help='')
    parser.add_argument('--train',help = 'Training mode',action = 'store_true')
    args = parser.parse_args()

    dql = Agent(param_set = args.hyperparameters)

    if args.train:
        dql.run(is_training = True)
    else:
        dql.run(is_training= False, render = True)



        
