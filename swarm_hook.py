import logging
import random
import torch
import torch.distributed as dist

import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default

logger = logging.getLogger(__name__)


class SwarmSGDState(object):

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
        
        # each iteration one pair of workers are selected
        self.selected_workers = selected_workers
        # randomly select workers
        if selected_workers is None:
            self.selected_workers = update_selected_workers()
            """
            selected_worker = 0
            self.selected_workers = [i for i in range(2)]
            if dist.get_rank() == 0:
                samples = [i for i in range(dist.get_world_size())]
                self.selected_workers[0] = random.choice(samples)
                samples = [i for i in range(dist.get_world_size() - 1)]
                self.selected_workers[1] = (selected_worker + random.choice(samples) + 1) % dist.get_world_size()
            # all workers have the same selected pair by broadcast
            dist.broadcast_object_list(self.selected_workers, src=0)
            """
            #dist.barrier()
            #print("1st worker: {}, 2nd worker: {}".format(self.selected_workers[0], self.selected_workers[1]))
        
        self.iter = 0

def swarm_SGD_hook(
    state: SwarmSGDState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Example
        state = swarm_hook.SwarmSGDState(process_group=None, subgroup=None, selected_workers=None)
        ddp_model.register_comm_hook(state, swarm_hook.swarm_SGD_hook)
    """
    # Update selected workers 
    # Too many collectives (boradcast)
    # state.selected_workers = update_selected_workers()

    # Get input tensors
    torch.futures.Future[torch.Tensor]
    fut = torch.futures.Future()
    input_tensor = bucket.buffer()
    
    receive_tensor = input_tensor
    rank = dist.get_rank()
    if rank == state.selected_workers[0]:
        req = dist.isend(input_tensor, state.selected_workers[1])
        # "wait" to avoid invalid error: too many async ops
        req.wait()
        req = dist.irecv(receive_tensor, state.selected_workers[1])
        req.wait()
        input_tensor = torch.mean(torch.stack([bucket.buffer(), receive_tensor]), 0)
    if rank == state.selected_workers[1]:
        req = dist.irecv(receive_tensor, state.selected_workers[0])
        req.wait()
        req = dist.isend(input_tensor, state.selected_workers[0])
        req.wait()
        input_tensor = torch.mean(torch.stack([bucket.buffer(), receive_tensor]), 0)
    
    fut.set_result(input_tensor)
    return fut
    
def update_selected_workers():
    selected_worker = 0
    selected_workers=[i for i in range(2)]
    if dist.get_rank() == 0:
        samples = [i for i in range(dist.get_world_size())]
        selected_workers[0] = random.choice(samples)
        samples = [i for i in range(dist.get_world_size() - 1)]
        selected_workers[1] = (selected_worker + random.choice(samples) + 1) % dist.get_world_size()
    # All workers have the same selected pair by broadcast
    dist.broadcast_object_list(selected_workers, src=0)
    return selected_workers
