import random
import numpy as np
import torch
import torch.distributed as dist

class SelectedPeers(object):
    def __init__(self, rank, size, pnum=1):
        self._rank = rank
        self._size = size
        self._pnum = pnum
        self._peers = np.zeros((size, pnum), dtype=int)
        self._out_peers = np.zeros((size, pnum), dtype=int)
        self._in_peers = np.zeros((size, pnum), dtype=int)
    
    # generate random peer pairs with adjustable peer number
    def shuffle_peers(self):
        samples = [i for i in range(self._size)]
        for j in range(self._pnum):
            tmp = []
            while len(tmp) < self._size:
                item = random.choice(samples)
                if item not in tmp:
                    tmp.append(item)
            for i in range(self._size):
                self._peers[tmp[i]][j] = tmp[self._size - i - 1]
    
    # generate neighbors pairs in a ring order with adjustable peer number
    def neighbor_peers(self):
        samples = [i for i in range(self._size)]
        for j in range(self._pnum):
            for i in range(self._size):
                self._in_peers[i][j] = samples[(i + j + 1) % self._size]
                self._out_peers[(i + j + 1) % self._size][j] = samples[i]
    
    def get_peers(self):
        return self._peers
    
    def get_onepeer(self, idx):
        return self._peers[idx]
    
    def get_in_peers(self):
        return self._in_peers
    
    def get_out_peers(self):
        return self._out_peers
    
    def get_one_inpeer(self, idx):
        return self._in_peers[idx]
    
    def get_one_outpeer(self, idx):
        return self._out_peers[idx]
    
    def get_mypeer(self):
        return self._peers[self._rank]
    
    def get_my_inpeer(self):
        return self._in_peers[self._rank]
    
    def get_my_outpeer(self):
        return self._out_peers[self._rank]

    # get list for boradcast
    def get_peers_list(self):
        tmp = []
        for p in self._peers:
            tmp.append(list(p))
        return tmp
    
    def get_in_peers_list(self):
        tmp = []
        for p in self._in_peers:
            tmp.append(list(p))
        return tmp
    
    def get_out_peers_list(self):
        tmp = []
        for p in self._out_peers:
            tmp.append(list(p))
        return tmp
