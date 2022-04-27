import torch
from torch import Tensor
from typing import List, Optional
from torch.optim.optimizer import Optimizer, required
import torch.distributed as dist

import new_com as comm
import get_peers as gpeer

class RandomOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        optim: torch.optim.Optimizer
        #averager: averagers.ModelAverager
    ):
        self.optim = optim
        self.param_groups = self.optim.param_groups
        #self.averager = averager

    @property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def step(self):
        r"""
        Performs a single optimization step (parameter update).
        """
        self.optim.step()
        size = dist.get_world_size()
        curr_rank = dist.get_rank()
        peers=[[0 for i in range(1)] for i in range(size)]
        sp = gpeer.SelectedPeers(rank=curr_rank, size=size, pnum=1)
        sp.shuffle_peers()
        peers = sp.get_peers_list()
        dist.broadcast_object_list(peers, src=0)
        selected_workers=peers[curr_rank]
        neighbors_info = {i : 0.5 for i in selected_workers}
        
        decentralized_aggregator = comm.get_aggregators(
                                            cur_rank=curr_rank,
                                            world=size,
                                            neighbors_info=neighbors_info,
                                            aggregator_type="decentralized",
                                        )
        
        #avoid dead loop
        if curr_rank not in selected_workers:
            #print ("rank {}, selected worker: {}".format(curr_rank, selected_workers[0]))
            #print(selected_workers)
            for group in self.param_groups:
                for p in group['params']:
                    p.data = decentralized_aggregator._agg(data=p.data, op='avg')
        
        #self.averager.average_parameters(params=self.param_groups)
        # calcualte the average of the parameters of the selected workers
        """
        for group in self.param_groups:
            for p in group['params']:
                device = torch.device('cpu')
                con_buf = [torch.zeros(p.data.size()).to(device) for _ in range(dist.get_world_size())] # Parameters placeholder
                dist.all_gather(con_buf, p.data)

                samples = [i for i in range(dist.get_world_size() - 1)]
                selected_worker = (random.choice(samples) + dist.get_rank() + 1) % dist.get_world_size()
                p.data = (con_buf[dist.get_rank()] + con_buf[selected_worker]) / 2
                
                selected_workers = [[0 for i in range(1)] for i in range(dist.get_world_size())]
                if dist.get_rank() == 0:
                    sp = gpeer.SelectedPeers(rank=dist.get_rank(), size=dist.get_world_size(), pnum=1)
                    sp.shuffle_peers()
                    selected_workers = sp.get_peers_list()
                dist.broadcast_object_list(selected_workers, src=0)

                send_tensor = p.data.clone().detach()
                receive_tensor = torch.zeros(p.data.size()).to(device)
                peer_rank = selected_workers[dist.get_rank()][0]
                if peer_rank != dist.get_rank():
                    req = dist.isend(send_tensor, peer_rank)
                    req.wait()
                    req = dist.irecv(receive_tensor, peer_rank)
                    req.wait()
                    p.data = torch.mean(torch.stack([send_tensor, receive_tensor]), 0)
                """
                # update selected workers
                #if dist.get_rank() in selected_workers:
                #    p.data = (con_buf[selected_workers[0]]+con_buf[selected_workers[1]])/2


    def zero_grad(self, set_to_none: bool = False):  # type: ignore[override]
        self.optim.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    
