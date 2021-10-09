#!/usr/bin/env python
# Authors: Junior Costa de Jesus #

import rospy
import os
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Bool
from environment_stage import Env
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import deque
import copy
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2


#---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))
#---Functions to make network updates---#
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data*(1.0 - tau)+ param.data*tau)

def hard_update(target,source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

#---Ornstein-Uhlenbeck Noise for action---#

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.99, min_sigma=0.01, decay_period= 600000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_noise(self, t=0): 
        ou_state = self.evolve_state()
        decaying = float(float(t)/ self.decay_period)
        self.sigma = max(self.sigma - (self.max_sigma - self.min_sigma) * min(1.0, decaying), self.min_sigma)
        return ou_state

#---Critic--#

EPS = 0.003
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1./np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v,v)

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_size = state_dim = state_dim
        self.action_size = action_dim
        self.HIDDEN1_UNITS = 300
        self.HIDDEN2_UNITS = 600
        self.w1 = nn.Linear(self.state_size, self.HIDDEN1_UNITS)
        self.a1 = nn.Linear(self.action_size,self.HIDDEN2_UNITS)
        self.h1 = nn.Linear(self.HIDDEN1_UNITS, self.HIDDEN2_UNITS)
        self.h3 = nn.Linear(self.HIDDEN2_UNITS, self.HIDDEN2_UNITS)
        self.V = nn.Linear(self.HIDDEN2_UNITS, self.action_size)

    def forward(self, s, a):
        w1 = F.relu(self.w1(s))
        a1 = self.a1(a)
        h1 = self.h1(w1)
        h2 = h1 + a1
        h3 = F.relu(self.h3(h2))
        out = self.V(h3)
        return out

#---Actor---#

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w):
        super(Actor, self).__init__()
        self.state_dim = state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v 
        self.action_limit_w = action_limit_w 
        self.HIDDEN1_UNITS = 300
        self.HIDDEN2_UNITS = 600
       
        self.fc1 = nn.Linear(self.state_dim, self.HIDDEN1_UNITS)
        self.fc2 = nn.Linear(self.HIDDEN1_UNITS, self.HIDDEN2_UNITS)
        self.steering = nn.Linear(self.HIDDEN2_UNITS, 1)
        nn.init.normal_(self.steering.weight, 0, 1e-4)
        self.acceleration = nn.Linear(self.HIDDEN2_UNITS, 1)
        nn.init.normal_(self.acceleration.weight, 0, 1e-4)


    def forward(self, x1):
        x = F.relu(self.fc1(x1))
        x = F.relu(self.fc2(x))

        if x1.shape <= torch.Size([self.state_dim]):
            out1 = torch.sigmoid(self.acceleration(x))*self.action_limit_v
            out2 = torch.tanh(self.steering(x))*self.action_limit_w
            action = torch.cat((out1, out2))
            return action
        else:
            out1 = torch.sigmoid(self.acceleration(x))*self.action_limit_v
            out2 = torch.tanh(self.steering(x))*self.action_limit_w
            action = torch.cat((out1, out2),1)
            return action

#---Memory Buffer---#

class MemoryBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0
        
    def sample(self, count):
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)
        s_array = torch.tensor([array[0] for array in batch],dtype=torch.float32, device=device)
        a_array = torch.tensor([array[1] for array in batch],dtype=torch.float32, device=device)
        r_array = torch.tensor([array[2] for array in batch],dtype=torch.float32, device=device)
        new_s_array = torch.tensor([array[3] for array in batch],dtype=torch.float32, device=device)
        done_array = torch.tensor([array[4] for array in batch],dtype=torch.float32, device=device)
        return s_array, a_array, r_array, new_s_array, done_array
    
    def len(self):
        return self.len
    
    def add(self, s, a, r, new_s, done):
        transition = (s, a, r, new_s, done)
        self.len += 1 
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)

#---Where the train is made---#

BATCH_SIZE = 256


LEARNING_RATE_ac = 0.0001 
LEARNING_RATE_cr = 0.001
GAMMA = 0.99 #

TAU = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
class Trainer:
    
    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w, ram):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w

        self.ram = ram
        self.actor = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE_ac)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LEARNING_RATE_cr)
        self.pub_qvalue = rospy.Publisher('qvalue', Float32, queue_size=5)
        self.pub_loss = rospy.Publisher('optimizer_loss', Float32MultiArray, queue_size=5)
        self.qvalue = Float32()
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        
    def get_exploitation_action(self,state):
        state = torch.from_numpy(state).cuda()
        action = self.actor.cuda().forward(state).detach()

        return action.data
        
    def get_exploration_action(self, state):
        state = torch.from_numpy(state).cuda()
        action = self.actor.cuda().forward(state).detach()
        new_action = action.data #+ noise

        return new_action
    
    def optimizer(self):
        s_sample, a_sample, r_sample, new_s_sample, done_sample = ram.sample(BATCH_SIZE)
        
        #-------------- optimize critic
        
        a_target = self.target_actor.cuda().forward(new_s_sample).detach()
        next_value = torch.squeeze(self.target_critic.cuda().forward(new_s_sample, a_target).detach())

        y_expected = torch.zeros(next_value.shape)
        y_expected[:,0] = r_sample + (1 - done_sample)*GAMMA*next_value[:,0]
        y_expected[:,1] = r_sample + (1 - done_sample)*GAMMA*next_value[:,1]
        

        y_predicted = torch.squeeze(self.critic.cuda().forward(s_sample, a_sample))

        #-------Publisher of Vs------
        self.qvalue = y_predicted.detach()
        self.pub_qvalue.publish(torch.max(self.qvalue))
        #----------------------------
        
        loss_data = Float32MultiArray()
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected.to(device))

        loss_data.data.append(loss_critic)

        
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        
        #------------ optimize actor
        pred_a_sample = self.actor.cuda().forward(s_sample)
        loss_actor = -1*torch.sum(self.critic.cuda().forward(s_sample, pred_a_sample))

        #print(loss_actor)
        loss_data.data.append(loss_actor)
        self.pub_loss.publish(loss_data)
        
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        
        soft_update(self.target_actor, self.actor, TAU)
        soft_update(self.target_critic, self.critic, TAU)
    
    def save_models(self, episode_count,alpha,beta,gamma):
        torch.save(self.target_actor.state_dict(), dirPath +'/Models/' + world + '/' + str(episode_count)+ '_actor' +str(alpha)+'_'+str(beta)+'_'+str(gamma)+ '.pt')
        torch.save(self.target_critic.state_dict(), dirPath + '/Models/' + world + '/'+str(episode_count)+ '_critic' +str(alpha)+'_'+str(beta)+'_'+str(gamma)+ '.pt')
        print('****Models saved***')

    def save_models2(self, episode_count,alpha,beta,gamma):
        torch.save(self.target_actor.state_dict(), dirPath +'/Models/' + world + '/' + str(episode_count)+ '_actor' +str(alpha)+'_'+str(beta)+'_'+str(gamma)+ '_M.pt')
        torch.save(self.target_critic.state_dict(), dirPath + '/Models/' + world + '/'+str(episode_count)+ '_critic' +str(alpha)+'_'+str(beta)+'_'+str(gamma)+ '_M.pt')
        print('****Models saved***')
        
    def load_models(self, episode):
        self.actor.load_state_dict(torch.load(dirPath + '/Models/' + world + '/Pictures/'+str(episode)+ '_actor10_50_0.01.pt'))
        self.critic.load_state_dict(torch.load(dirPath + '/Models/' + world + '/Pictures/'+str(episode)+ '_critic10_50_0.01.pt'))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print(world, '***Models load***')

#---Mish Activation Function---#
def mish(x):
    '''
        Mish: A Self Regularized Non-Monotonic Neural Activation Function
        https://arxiv.org/abs/1908.08681v1
        implemented for PyTorch / FastAI by lessw2020
        https://github.com/lessw2020/mish
        param:
            x: output of a layer of a neural network
        return: mish activation function
    '''
    return x*(torch.tanh(F.softplus(x)))

#---Run agent---#

is_training = True
exploration_decay_rate = 0.001
MAX_EPISODES = 201
MAX_STEPS = 1000
MAX_BUFFER = 100000
rewards_all_episodes = []

step_per_episode = [0] * MAX_EPISODES 
reward_per_episode = [0] * MAX_EPISODES 

STATE_DIMENSION = 24
ACTION_DIMENSION = 2
ACTION_V_MAX = 5.5 
ACTION_W_MAX = 2. 
world = 'world_u'
if is_training:
    var_v = ACTION_V_MAX*.5
    var_w = ACTION_W_MAX*2*.5
else:
    var_v = ACTION_V_MAX*0.10
    var_w = ACTION_W_MAX*0.10

print('State Dimensions: ' + str(STATE_DIMENSION))
print('Action Dimensions: ' + str(ACTION_DIMENSION))
print('Action Max: ' + str(ACTION_V_MAX) + ' m/s and ' + str(ACTION_W_MAX) + ' rad/s')

# ddpg_override start
vehicle_control_ddpg_override = False
def update_vehicle_control_ddpg_override(data):
    global vehicle_control_ddpg_override
    vehicle_control_ddpg_override = data.data
    print("updated data", vehicle_control_ddpg_override)

if __name__ == '__main__':
    rospy.init_node('ddpg_stage')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    before_training = 1 
    rospy.Subscriber("/carla/ego_vehicle/vehicle_control_ddpg_override", Bool, update_vehicle_control_ddpg_override)
    while vehicle_control_ddpg_override == False:
        rospy.sleep(3)
        print("press v key to start ddpg!!")
    
    for a in [10]:
        for b in [50, 20]:
            for g in [0.01]:
                ram = MemoryBuffer(MAX_BUFFER)
                trainer = Trainer(STATE_DIMENSION, ACTION_DIMENSION, ACTION_V_MAX, ACTION_W_MAX, ram)
                #trainer.load_models(120)
                noise_st = OUNoise(1, mu=0.0, theta=0.6,max_sigma=0.3, min_sigma=0.1, decay_period=8000000)
                noise_vel = OUNoise(1,mu=0.5, theta=1.0, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)
                env = Env(action_dim=ACTION_DIMENSION)
                past_action = np.zeros(ACTION_DIMENSION)
                print('alpha=' , a , ' beta=' , b , ' gamma=' , g)
                start_time = rospy.Time.from_sec(time.time()).to_sec() #Woo
                env.alpha = a
                env.beta = b
                env.gamma = g
                ep_duration_list = []
                flag = 0
                
                for ep in range(MAX_EPISODES):
                    print('alpha=' , a , ' beta=' , b , ' gamma=' , g)
                    ep_start_time = rospy.Time.from_sec(time.time()).to_sec()
                    ep_angle_list = []
                    
                    done = False
                    state = env.reset()
                    check = False
                    if is_training and not ep%10 == 0 and ram.len >= before_training*MAX_STEPS:
                        print('---------------------------------')
                        print('Episode: ' + str(ep) + ' training')
                        print('---------------------------------')
                        
                    else:
                        if ram.len >= before_training*MAX_STEPS:
                            print('---------------------------------')
                            print('Episode: ' + str(ep) + ' evaluating')
                            print('---------------------------------')
                        else:
                            print('---------------------------------')
                            print('Episode: ' + str(ep) + ' adding to memory')
                            print('---------------------------------')
                            check = True
                            

                    rewards_current_episode = 0.
                    for step in range(MAX_STEPS):
                        state = np.float32(state)

                        if is_training and not ep%10 == 0 and ram.len >= before_training*MAX_STEPS: ####################################
                            action = trainer.get_exploration_action(state)
                            Nst = copy.deepcopy(noise_st.get_noise(t=step)) 
                            Nvel = copy.deepcopy(noise_vel.get_noise(t=step)) 
                            action = action.cpu().numpy()
                            Nst = (Nst*ACTION_W_MAX) # N
                            Nvel = (Nst*ACTION_V_MAX/2) # N          
                            action[0] = np.clip(action[0] + Nvel, 3, ACTION_V_MAX)
                            action[1] = np.clip(action[1] + Nst, -ACTION_W_MAX, ACTION_W_MAX)

                        else:
                            action = trainer.get_exploration_action(state)
                            action = action.cpu().numpy()
                            action[0] = np.clip(action[0], 3, ACTION_V_MAX)
                            action[1] = np.clip(action[1] , -ACTION_W_MAX, ACTION_W_MAX)
                        if not is_training:
                            action = trainer.get_exploitation_action(state)
                            action = action.cpu().numpy()
                            action[0] = np.clip(action[0] , 3, ACTION_V_MAX)
                            action[1] = np.clip(action[1] , -ACTION_W_MAX, ACTION_W_MAX)
                        next_state, reward, done = env.step(action, past_action)
                        past_action = copy.deepcopy(action)
                        rewards_current_episode += reward
                        print('M' if check else 'T','_ep ', ep, 'step :', step, 'reward : ', reward, "rewards_current_episode",rewards_current_episode)
                        next_state = np.float32(next_state)
                        if not ep%10 == 0 or not ram.len >= before_training*MAX_STEPS:
                            if reward >= 20.:
                                print('***\n-------- Maximum Reward ----------\n****')
                                for _ in range(3):
                                    ram.add(state, action, reward, next_state, done)
                            else:
                                ram.add(state, action, reward, next_state, done)
                        state = copy.deepcopy(next_state)

                        if ram.len >= before_training*MAX_STEPS and is_training and not ep%10 == 0:

                            trainer.optimizer()

                        if ep%6 == 0:
                            ep_angle_list.append(action[1] * 2.5)
            
                        if done or step == MAX_STEPS-1:
            
                            print('reward per ep: ' + str(rewards_current_episode))
                            print('*\nbreak step: ' + str(step) + '\n*')

                            print('sigma: ' + str(noise_st.sigma))
                            print('sigma: ' + str(noise_vel.sigma))

                            if not ep%10 == 0:
                                pass
                            else:
                                # if ram.len >= before_training*MAX_STEPS:
                                result = rewards_current_episode
                                pub_result.publish(result)
                            step_per_episode[ep] = step #s
                            reward_per_episode[ep] = rewards_current_episode #s
                            break

                        if ram.len >= before_training*MAX_STEPS and flag == 0:
                            flag = 1
                            trainer.save_models2(ep,a,b,g)
                            break

                    ep_finish_time = rospy.Time.from_sec(time.time()).to_sec()

                    if ep%20 == 0:
                        trainer.save_models(ep,a,b,g)

                    if ep%6 == 0:
                       plt2.plot(ep_angle_list)
                       if check :
                           plt2.title('M_Episode'+str(ep))
                       else :
                           plt2.title('T_Episode'+str(ep))
                       plt2.xlabel('Step')
                       plt2.ylabel('Angle')
                       plt2.savefig(dirPath +'/Models/' + world + '/Pictures/Angle_ep{}_a{}_b{}s_g{}.png'.format(ep,a,b,g))# change
                       plt2.clf()

                    
                    ep_duration_list.append(round(ep_finish_time - ep_start_time, 1))

                finish_time = rospy.Time.from_sec(time.time()).to_sec() #Woo
                duration = round(finish_time - start_time, 1) #Woo

# plot
                plt.subplot(3, 1, 1)
                plt.plot(step_per_episode)
                plt.title('alpha=' + str(a) + ' beta=' + str(b) + ' gamma=' + str(g) + ' Vmax=' + str(ACTION_V_MAX) + ' duration=' + str(duration) + 's' +'\nEpisode length over time')
#plt.xlabel()
#plt.xlabel('Episode')
                plt.ylabel('Step')


                plt.subplot(3, 1, 2)
                plt.plot(reward_per_episode)
                plt.title('Episode reward over time'+'\n')
                plt.xlabel('Episode')
                plt.ylabel('Total reward')

                plt.subplot(3, 1, 3)
                plt.plot(ep_duration_list)
                plt.title('Episode  time')
                plt.xlabel('Episode')
                plt.ylabel('Time')
                plt.tight_layout()
                #plt.show() #s
                plt.savefig(dirPath +'/Models/' + world + '/Pictures/a{}_b{}s_g{}.png'.format(a,b,g))# change
                print('Completed Training!! ' + 'alpha=' + str(a) + ' beta=' + str(b) + ' gamma=' + str(g) + ' Action Max: ' + str(ACTION_V_MAX) + ' m/s and ' + str(ACTION_W_MAX) + ' rad/s ' + 'duration=' + str(duration) + 's')
                plt.clf()
