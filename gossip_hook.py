import logging
import random
import torch
import torch.distributed as dist
import get_peers as gpeer
import new_com as comm
import topology

import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default

logger = logging.getLogger(__name__)


class GossipSGDState(object):

    __slots__ = [
        "process_group",
        "subgroup",
    ]

    def __init__(
        self,
        process_group,
        subgroup,
    ):
        
        # The group used for all-reducing gradients globally.
        self.process_group = process_group
        # The group used for all-reducing gradients locally.
        self.subgroup = subgroup
        
        rank = dist.get_rank()
        size = dist.get_world_size()
        graph = topology.define_graph_topology(graph_topology="ring",
                                            world=size,
                                            rank=rank,
                                            n_mpi_process=len(process_group),
                                            n_sub_process=len(subgroup),
                                            comm_device="gpu",
                                            on_cuda=True)
        neighbors_info = graph.get_neighborhood()
        self.decentralized_aggregator = comm.get_aggregators(cur_rank=curr_rank,
                                                    world=size,
                                                    neighbors_info=neighbors_info,
                                                    aggregator_type="decentralized")
            
        #self.iter = 0


def gossip_SGD_hook(
    state: GossipSGDState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Example
        state = gossip_hook.GossipSGDState(process_group=None, subgroup=None, selected_workers=None)
        ddp_model.register_comm_hook(state, gossip_hook.gossip_SGD_hook)
    """
    global_group_to_use = (
        state.process_group if state.process_group is not None else dist.group.WORLD
    )
    torch.futures.Future[torch.Tensor]
    fut = torch.futures.Future()
    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.buffer()
    
    input_tensor = self.decentralized_aggregator._agg(data=bucket.buffer(), op='avg')
    
    fut.set_result(input_tensor)
	
    return fut
    
