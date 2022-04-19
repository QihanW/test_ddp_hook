import logging
import random
import torch
import torch.distributed as dist
import get_peers as gpeer

import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default

logger = logging.getLogger(__name__)


class GossipSGDState(object):

    __slots__ = [
        "process_group",
        "subgroup",
        "iter",
        "selected_workers"
    ]

    def __init__(
        self,
        process_group,
        subgroup,
        selected_workers,
    ):
        
        # The group used for all-reducing gradients globally.
        self.process_group = process_group
        # The group used for all-reducing gradients locally.
        self.subgroup = subgroup

        self.selected_workers = selected_workers
        size = dist.get_world_size()
        if selected_workers is None:
            peers = [[0 for i in range(1)] for i in range(size)]
            if dist.get_rank() == 0:
                sp = gpeer.SelectedPeers(rank=dist.get_rank(), size=size, pnum=1)
                sp.shuffle_peers()
                peers = sp.get_peers_list()
            dist.broadcast_object_list(peers, src=0)
            self.selected_workers = peers
            #print(self.selected_workers)
            
        self.iter = 0


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
    
    receive_tensor = input_tensor
    rank = dist.get_rank()
    # The selected workers array is 2D.
    peer_rank = state.selected_workers[rank][0]
    if peer_rank != rank:
        req = dist.isend(input_tensor, peer_rank)
        req.wait()
        req = dist.irecv(receive_tensor, peer_rank)
        req.wait()
        input_tensor = torch.mean(torch.stack([bucket.buffer(), receive_tensor]), 0)
    
    fut.set_result(input_tensor)
	
    size = dist.get_world_size()
    peers = [[0 for i in range(1)] for i in range(size)]
    if dist.get_rank() == 0:
        sp = gpeer.SelectedPeers(rank=dist.get_rank(), size=size, pnum=1)
        sp.shuffle_peers()
        peers = sp.get_peers_list()
    dist.broadcast_object_list(peers, src=0)
    state.selected_workers = peers
	
    return fut
    
