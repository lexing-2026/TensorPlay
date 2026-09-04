import os
import shutil

import tensorplay as tp
import tensorplay.distributed as dist
import tensorplay.distributed.checkpoint as dcp
import tensorplay.multiprocessing as mp
import tensorplay.nn as nn
from tensorplay.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
)
from tensorplay.distributed.device_mesh import init_device_mesh
from tensorplay.distributed.fsdp import FullyShardedDataParallel as FSDP


CHECKPOINT_DIR = f"~/{os.environ.get('LOGNAME', '')}/checkpoint"


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        tp.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        return tp.rand(8, 8, device="cuda")


def _make_stateful(model, optim):
    _patch_model_state_dict(model)
    _patch_optimizer_state_dict(model, optimizers=optim)


def _train(model, optim, train_steps=1):
    tp.manual_seed(0)
    loss = None
    for _ in range(train_steps):
        loss = model(model.get_input()).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()
    return loss


def _init_model(device, world_size):
    device_mesh = init_device_mesh(device, (world_size,))
    model = FSDP(Model().cuda(), device_mesh=device_mesh, use_orig_params=True)
    optim = tp.optim.Adam(model.parameters(), lr=0.1)
    _make_stateful(model, optim)
    return model, optim


def run(rank, world_size, device="cuda"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    tp.cuda.set_device(rank)
    model, optim = _init_model(device, world_size)
    _train(model, optim, train_steps=2)
    dcp.save(
        state_dict={"model": model, "optimizer": optim},
        checkpoint_id=CHECKPOINT_DIR,
    )
    model, optim = _init_model(device, world_size)
    dcp.load(
        state_dict={"model": model, "optimizer": optim},
        checkpoint_id=CHECKPOINT_DIR,
    )
    _train(model, optim, train_steps=2)


if __name__ == "__main__":
    world_size = tp.cuda.device_count()
    print(f"running stateful checkpoint example on {world_size} devices")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
