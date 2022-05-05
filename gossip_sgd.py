import torch
from torch import Tensor
from typing import List, Optional
from torch.optim.optimizer import Optimizer, required
import torch.distributed as dist

import new_com as comm
import get_peers as gpeer
import topology

class GossipSGD(torch.optim.Optimizer):
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

        graph = topology.define_graph_topology(graph_topology="ring",
                                            world=size,
                                            rank=curr_rank,
                                            n_mpi_process=size,
                                            n_sub_process=1,
                                            comm_device="gpu",
                                            on_cuda=True)
        
        neighbors_info = graph.get_neighborhood()
        #print(neighbors_info)
        decentralized_aggregator = comm.get_aggregators(
                                            cur_rank=curr_rank,
                                            world=size,
                                            neighbors_info=neighbors_info,
                                            aggregator_type="decentralized",
                                        )
        
        for group in self.param_groups:
            for p in group['params']:
                p.data = decentralized_aggregator._agg(data=p.data, op='avg')
            #if p.grad is None:
            #    continue
            #p.grad.data = decentralized_aggregator._agg(data=p.grad.data, op='avg')
 

    def zero_grad(self, set_to_none: bool = False):  # type: ignore[override]
        self.optim.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)
