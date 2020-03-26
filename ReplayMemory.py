import numpy as np
import torch

class ReplayMemory(object):
    """Implements basic replay memory"""

    def __init__(self, observation_size, max_size):
        self.observation_size = observation_size
        self.num_observed = 0
        self.max_size = max_size
        self.samples = {
                 'obs'      : np.zeros((self.max_size * 1 * self.observation_size),
                                       dtype= np.float64).reshape(self.max_size, self.observation_size),
                 'action'   : np.zeros((self.max_size * 1), dtype = np.int64).reshape(self.max_size, 1),
                 'reward'   : np.zeros((self.max_size * 1), dtype = np.float64).reshape(self.max_size, 1),
                 'terminal' : np.zeros((self.max_size * 1), dtype= np.int64).reshape(self.max_size, 1),
               }
        self.dtype          = np.float64
        self.use_cuda       = torch.cuda.is_available()
        self.device         = torch.device("cuda:0" if self.use_cuda else "cpu")

    def observe(self, state, action, reward, done):
        index = self.num_observed % self.max_size

        self.samples['obs'][index, :] = state
        self.samples['action'][index, :] = action
        self.samples['reward'][index, :] = reward
        self.samples['terminal'][index, :] = done

        self.num_observed += 1

    def sample_minibatch(self, minibatch_size):
        max_index = min(self.num_observed, self.max_size) - 1
        sampled_indices = np.random.randint(max_index, size=minibatch_size)

        s      = torch.tensor((self.samples['obs'][sampled_indices, :]),dtype = torch.float64).to(self.device)
        s_next = torch.tensor((self.samples['obs'][sampled_indices+1, :]),dtype = torch.float64).to(self.device)

        a      = torch.tensor((self.samples['action'][sampled_indices].reshape(minibatch_size)),dtype = torch.float64).cpu()
        r      = torch.tensor((self.samples['reward'][sampled_indices].reshape((minibatch_size, 1))),dtype = torch.float64).cpu()
        done   = torch.tensor((self.samples['terminal'][sampled_indices].reshape((minibatch_size, 1))),dtype = torch.int64).cpu()


        return (s, a, r, s_next, done)

    def normalisedit(self):
        
        s       = self.samples['obs'][:, :]
        

        s_mean  = np.mean(s, axis = 0 , dtype = np.float64)
        s_std   = np.std(s, axis = 0 , dtype = np.float64)

        self.samples['obs'][:, :]   = (s - s_mean)/s_std



        





