import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.distributions import Categorical
import os

"""
Contains the definition of the agent that will run in an
environment.
"""

class RandomAgent:
    def __init__(self):
        """Init a new agent.
        """

    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change, but your initial
        location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """
        pass

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        return np.random.normal(0, 10)

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        pass
    


class bBbAgent:
    def __init__(self):
        
        self.all_rewards = [] # Rewards during Agent training
        self.all_losses = [] # Losses during Agent training
        self.total_rewards = 0 # sum of rewards
        self.maximumRewardRecorded = 0 # Max reward recorded during training/testing
        self.episode_states = [] # States of the current episode
        self.episode_actions = [] # Actions of the current episode
        self.episode_rewards = [] # Rewards of the current episode
        self.iter = 1 # iteration counter 
        self.probabilities =  [] 
        
        self.xs = [] # Positions of the current episode
        self.vs = [] # Speeds of the current episode
        self.acts = [] # Acts of the current episode
        
        self.x_max = -1000 # xmax recorded during the current episode
        self.vmax = 0 # vmax during the current epsiosde
        self.ngames = 0 # games played with the agent
        
        self.log_probs = Variable(torch.Tensor()) # intialization if the probabilities during the action method
             
        print('==========================================')
        
        self.__build_network() # Build the network during the init phase
        

    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change, but your initial
        location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """
        # Reset the variables concerning one episode
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.true_rewards = []
        self.iter = 1
        self.xs = []
        self.vs = []
        self.acts = []
        self.x_max = -1000
        self.vmax = 0
        self.probabilities =  []
        self.log_probabilities = Variable(torch.Tensor(), requires_grad=False)


    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        input_ = np.array([observation[0], observation[1]]) # Input preparation for the network
        out = self.model(Variable(torch.from_numpy(input_).float(), requires_grad=False)) # computation of the output of the net
        c = Categorical(out) # probabilities
        act = c.sample() # pick one action using probabilities


        if self.log_probabilities.size() != torch.Size([0]): # completing the probabilities list
            self.log_probabilities = torch.cat([self.log_probabilities, c.log_prob(act).view(1)]) 
        else:
            self.log_probabilities = c.log_prob(act).view(1)
        
        self.acts.append(act.item() * 1 - 5) # Completing the act list
        return act.item() * 1.4 - 7 # return the act chosen and transform it into a value usable for the car

    
    def __build_network(self):
        """Create a base network"""
        
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__() # layers
                self.layer1 = nn.Linear(2, 32)
                self.layer2 = nn.Linear(32, 64)
                self.layer3 = nn.Linear(64, 128)
                self.layer4 = nn.Linear(128, 11)
                
            def forward(self, x): # activation functions
                out = F.relu(self.layer1(x))
                out = F.relu(self.layer2(out))
                out = F.relu(self.layer3(out))
                out = F.softmax(self.layer4(out), dim=-1)
                return out    
        
        self.model = Net() #instanciating the net
        
        if 'saved_model' in os.listdir(): # If a model has been trained, then load it
            self.model.load_state_dict(torch.load('saved_model'))
            print('USING SAVED MODEL')
        else:
            print('new model')
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0002) # Optimizer
        print('Parameters', list(self.model.parameters())) # learnable parameters

    

    def update_policy(self, actions, probabilities, rewards):
        ''' Updates the policy using a policy gradient method
        '''
        loss = - torch.sum(torch.mul(self.log_probabilities, Variable(torch.from_numpy(rewards).float()))) # compute the loss
        print('loss', loss.item())
        
        self.optimizer.zero_grad() # reset optimizer
        loss.backward() # gradients of the weights wrt the loss
        self.optimizer.step() # updating weights
        return loss
    
    def reward(self, observation, action, reward, stop=None):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        self.episode_states.append([observation[0], observation[1]]) # complete states of the episode
        self.episode_actions.append(action) # complete acts  of the episode
        if observation[0] > self.x_max +1 and abs(observation[1]) > self.vmax and self.iter > 10 and self.ngames < 500:
            self.episode_rewards.append(reward+20) # If we go beyond the maximum speed/position reached we give a small reward
            self.x_max = observation[0]# update maxes
            self.vmax = abs(observation[1])
        elif observation[0] > self.x_max +1 and self.iter > 10 and self.ngames < 500: # same
            self.episode_rewards.append(reward+10)
            self.x_max = observation[0]
        elif abs(observation[1]) > self.vmax and self.iter > 10 and self.ngames < 500: # same
            self.vmax = abs(observation[1])
            self.episode_rewards.append(reward+10)
        else:
            # otherwise just return the reward given by the environment
            self.episode_rewards.append(reward)
        self.true_rewards.append(reward) # rewards given by the environment
        self.xs.append(observation[0]) # update position and speed
        self.vs.append(observation[1])
        
        if stop is not None or self.iter == 400 or reward>0: # if end of the episode
            self.ngames += 1 # games count updated
            print('NGAMES', self.ngames)
            if self.ngames % 10 == 0:
                print('saving')
                torch.save(self.model.state_dict(), 'saved_model') # saving model
            print('REWARDING')
            episode_rewards_sum = np.sum(self.true_rewards) # sum the rewards obtained during the episode
            print(episode_rewards_sum)
            self.all_rewards.append(episode_rewards_sum) # updating values for the agent
            self.total_rewards = np.sum(self.all_rewards)
            self.maximumRewardRecorded = np.amax(self.all_rewards)
            
            self.episode_states_2 = np.array(self.episode_states) # transforming into numpy arrays
            self.episode_actions_2 = np.array(self.episode_actions)
            self.episode_rewards_2 = np.array(self.episode_rewards)

            action_onehot = to_categorical((self.episode_actions_2.reshape(len(self.episode_rewards_2), 1) + 10)/2, num_classes=11)
            # action are transformed into their onehot form so that they correspond to the ouput of the network
            discount_reward = compute_discounted_reward(self.episode_rewards_2)
            # compute the discounted reward (multiplication by a factor for older rewards)
            a = self.update_policy(action_onehot, np.array(self.probabilities), discount_reward) # call the update plicy method
            self.all_losses.append(a.item()) # complete the losses list

        self.iter += 1
        
        
        if self.ngames == 200:
            print(self.model.parameters())

def compute_discounted_reward(R, discount_rate=0.99):
    ''' Compute the discounted reward : R1 + gamma * R2 + gamma^2 * R3 + ...
    '''
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(R))):

        running_add = running_add * discount_rate + R[t] # multiplication by the gamma factor
        discounted_r[t] = running_add
    discounted_r -= discounted_r.mean() / discounted_r.std()
    return discounted_r

def to_categorical(array, num_classes=2):
    ''' Transforming int value into a categorical vector
    '''
    b = np.zeros((len(array), num_classes))
    for i in range(len(array)):
        b[i][array.astype(int)[i]] = 1
    return b

# Agent = RandomAgent
Agent = bBbAgent

