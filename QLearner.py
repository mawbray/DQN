import numpy as np 
import pandas as pd
import torch
import timeit
from ANNClass import Net as QNetwork
from ReplayMemory import ReplayMemory as RollOutMemory
import math
import h5py
eps  = np.finfo(float).eps


class Agent(object):  
        
    def __init__(self, movements):
        self.movements      = movements

    def act(self, state):
        raise NotImplementedErrorqs

class QLAgent(Agent):
    """Q-Learning agent with function approximation."""

    def __init__(self, movements, controls, **kwargs):
        super(QLAgent, self).__init__(movements)

        # Definitions
        self.args           = kwargs
        self.controls       = controls
        # Agent Topology
        self.obs_size       = self.args['obs_size']
        self.inputZ         = int(self.obs_size + 3)
        self.C              = int(0)

        # code regularisation
        self.dtype          = torch.float
        self.use_cuda       = torch.cuda.is_available()
        self.device         = torch.device("cuda:0" if self.use_cuda else "cpu")
        torch.cuda.empty_cache() 


        # miscellaneous algorithm hyperparameters
        self.movements          = movements
        self.n_traj             = self.args['n_holds']                     # n trajectories in memory
        self.action_rolls       = int(self.movements * self.n_traj)         # N transitions in an memory
        self.gamma              = self.args['gamma']                        # discount future return
        self.minibatch_size     = kwargs.get('batch_size', 32)
        self.transition         = kwargs.get('Weight_Transfer', int(1e3))
 

        # defining agent topology and optimisation routine

        net_kwargs      = {'input_size': self.inputZ,                   # defining feature size
                           'hs_1': self.args['hs_1'],                   # nodes in hs1
                           'hs_2': self.args['hs_2'],                   # nodes in hs2
                           'output_size': self.args['output_size']}     # no of controls
        
        self.tau                = kwargs.get('tau', 1)
        self.model_network      = QNetwork(**net_kwargs).to(self.device)
        self.target_network     = QNetwork(**net_kwargs).to(self.device)
        self.model_network.apply(self.model_network.weights_init)         
        self.optimizer          = torch.optim.RMSprop(self.model_network.parameters(), lr=kwargs.get('learning_rate', 5e-2))                    # Stochastic Gradient Descent
        self.loss_func          = torch.nn.SmoothL1Loss().to(self.device)                                                                       # Huber loss robust regression                                                                                            # xavier weight initialisation
        self.target_network.load_state_dict(self.model_network.state_dict())
        self.current_loss       = .0                                                                                    

        # storing and standardising states
        self.memory             = RollOutMemory(self.obs_size, self.action_rolls)

        # initialising h5py file for dataset storage and collation
        self.f  = h5py.File("trajectory_transition_store.hdf5",'w')



    def hiDprojection(self, state, T= None):

        if T == None:
            X, N = state[:,0], state[:,1]
            output = torch.zeros((X.shape[0], self.inputZ), dtype = torch.float64).to(self.device)
            output[:,0], output[:,1], output[:,2], output[:,3], output[:,4] = X, N, X * N, X ** 2, N ** 2
        else: 
            X, N = state[0], state[1]
            output = torch.zeros((self.inputZ), dtype = torch.float64)
            output[0], output[1], output[2], output[3], output[4] = X, N, X * N, X ** 2, N ** 2
        
        return output
    

    def StoreEpisode(self,states, time, actions, rwd_ass, epoch, objective):
        # create datafile for each epoch

        grp     = self.f.create_group(f"trajectories_epoch_{epoch}")
        subgrp1 = grp.create_group('states')
        ds1     = subgrp1.create_dataset('state_traj', data=np.array(states))
        subgrp2 = grp.create_group('actions')
        ds2     = subgrp2.create_dataset('action_traj', data=np.array(actions))
        subgrp3 = grp.create_group('rwd_ass')
        ds3     = subgrp3.create_dataset('rwd_traj', data=np.array(rwd_ass))
        subgrp5 = grp.create_group('time')
        ds5     = subgrp5.create_dataset('time_traj', data=np.array(time))
        subgrp6 = grp.create_group('ObjFun')
        ds6     = subgrp6.create_dataset('ObjFun_traj', data=np.array(objective))

    def epsilon(self, epoch):

        TEp = self.args.get('Tepochs',1000)
        F = 0.1
        G = -np.log(0.1)*F*TEp     # =no of epochs until behave =0.1

        if epoch < G:
            behave = np.exp(-epoch/(TEp*F))
        else:
            behave = 0.1

        return behave


    def act(self, state, epoch, Valid = None):

        
        
        if Valid == None:
            epsilon = self.epsilon(epoch)
            if np.random.random() < epsilon:
                i = np.random.randint(0,len(self.controls))
            else: 
                self.model_network.eval()
                state   = self.hiDprojection(state,1)
                input   = state.reshape(1,state.shape[0])
                input   = input.clone().detach().to(self.device)
                Q       = self.model_network(input.float())
                Q       = Q.cpu().squeeze()
                i       = torch.argmax(Q)

        elif Valid != None:
            if np.random.random() < 0.0:
                i = np.random.randint(0,len(self.controls))
            else: 
                self.model_network.eval()
                state   = self.hiDprojection(state,1)
                input   = state.reshape(1,state.shape[0])
                input   = input.clone().detach().to(self.device)
                Q       = self.model_network(input.float())
                Q       = Q.cpu().squeeze()
                i       = torch.argmax(Q)

        action = self.controls[i]
        

        return action 
    
    def Observe(self, s, action, reward, s_next, is_terminal):
        self.memory.observe(s, action, reward, is_terminal)
        return 

    def update_model(self):
        self.memory.normalisedit()
        (s, action, reward, s_next, is_terminal) = self.memory.sample_minibatch(self.minibatch_size)

        s, s_next = self.hiDprojection(s), self.hiDprojection(s_next)
        # compute Q targets (max_a' Q_hat(s_next, a'))
        self.target_network.eval()
        Q_hat       = self.target_network(s_next.float())
        Q_hat       = Q_hat.cpu().squeeze()
        Q_hat_max   = torch.max(Q_hat, dim=1, keepdim=True)[0]
        Q_hat_max   = torch.reshape(Q_hat_max, (self.minibatch_size,1))
        y           = (((1.0-is_terminal.float()))* self.gamma*Q_hat_max.float() + reward.float())
        y           = torch.reshape(y, (self.minibatch_size, 1)).float()
       

        # compute Q(s, action)
        self.model_network.eval()
        Q           = self.model_network(s.float())
        Q           = Q.cpu().squeeze()
        acts        = torch.tensor(self.controls).cpu()
        Q_subset    = torch.zeros((self.minibatch_size)).cpu()
        for i in range(self.minibatch_size):
            cond        = (action[i] == acts)
            Q_subset[i] = Q[i][cond]
        Q_subset        = torch.reshape(Q_subset, (self.minibatch_size,1))
        
        # compute Huber loss
        loss            = self.loss_func(Q_subset.to(self.device) , y.to(self.device))

        self.model_network.train()
        # perform model update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
  

        self.C += 1

        if self.C % self.transition == 0:
            # target network tracks the model iterating over network parameters
            self.target_network.load_state_dict(self.model_network.state_dict())
            self.C = 0   
            print('transfer')

        return loss.item()

    
    def Learned(self):
        return self.model_network



 
    


