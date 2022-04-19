import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
#import torch.distributed.algorithms.model_averaging.averagers as averagers
import torch.nn as nn
import torch.optim as optim
#import torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook as post_localSGD

def run_worker(rank, world_size):
    #torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    #dist.init_process_group(backend="gloo", init_method='env://')
    torch.cuda.set_device(rank)
    module = nn.Linear(1, 1, bias=False).cuda()
    model = nn.parallel.DistributedDataParallel(
        module, device_ids=[rank], output_device=rank
    )

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    #averager = averagers.PeriodicModelAverager(period=4, warmup_steps=5)
    for step in range(0, 10):
        optimizer.zero_grad()
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        #averager.average_parameters(model.parameters())
        print("Step {}: loss = {}".format(step, loss.item()))
    # Cleanup.
    dist.destroy_process_group()

if __name__ == "__main__":
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12345"
   world_size = torch.cuda.device_count()
   mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
