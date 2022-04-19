import os
 
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import swarm_hook
import gossip_hook
import random
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as comhook
import torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook as post_localSGD


def run_worker(rank, world_size):
   # Initialize the distributed environment.
   torch.cuda.set_device(rank)
   dist.init_process_group("nccl", rank=rank, world_size=world_size)
 
   # Define the model architecture and wrap it up with DDP.
   model = (
       nn.Sequential(
           nn.Conv2d(3, 64, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.Conv2d(64, 2, kernel_size=3, padding=1),
       )
       .half()
       .cuda()
   )
   ddp_model = nn.parallel.DistributedDataParallel(
       module=model,
       device_ids=[rank],
   )

   samples = [i for i in range(world_size)]
   mypeer = random.choice(samples)
   if mypeer == rank:
       mypeer = (rank + 1) % world_size
   sub_group = dist.new_group([1,2])
   mypeer = random.choice(samples)
   
 
   # Define loss function and optimizer.
   loss_fn = nn.MSELoss()
   optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
   input = torch.randn(4, 3, 128, 128).half()
   labels = torch.randn(4, 2, 128, 128).half().to(rank)

   # Register a post-localSGD communication hook.
   #state = post_localSGD.PostLocalSGDState(process_group=None, subgroup=None, start_localSGD_iter=5)
   #ddp_model.register_comm_hook(state, post_localSGD.post_localSGD_hook)

   #state = swarm_hook.SwarmSGDState(process_group=None, subgroup=None, selected_workers=None)
   #ddp_model.register_comm_hook(state, swarm_hook.swarm_SGD_hook)
   
   state = gossip_hook.GossipSGDState(process_group=None, subgroup=None, selected_workers=None)
   ddp_model.register_comm_hook(state, gossip_hook.gossip_SGD_hook)


   for step in range(0, 50):
       optimizer.zero_grad()
       output = ddp_model(input)
       loss = loss_fn(output, labels).half()
       loss.backward()
       if step % 10 == 0 and rank == 0:
           print("Step {}: loss = {}".format(step, loss.item()))
       optimizer.step()
   # Cleanup.
   dist.destroy_process_group()


 
if __name__ == "__main__":
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12345"
   world_size = torch.cuda.device_count()
   mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)

