import os
import shutil
import traceback
from concurrent.futures import Future

import tensorplay as tp
import tensorplay.distributed as dist
import tensorplay.distributed.checkpoint as dcp
import tensorplay.multiprocessing as mp
import tensorplay.nn as nn
import tensorplay.nn.functional as F
from tensorplay.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
)
from tensorplay.distributed.device_mesh import init_device_mesh
from tensorplay.distributed.fsdp import FullyShardedDataParallel as FSDP


DEVICE = "cuda"
NUM_EPOCHS = 1000
SAVE_PERIOD = 10
FAULT_PERIOD = 25
CHECKPOINT_DIR = f"~/{os.environ.get('LOGNAME', '')}/checkpoint"


class InjectedException(Exception):
    pass


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = nn.Linear(8, 32)
        self.net2 = nn.Linear(32, 128)
        self.net3 = nn.Linear(128, 64)
        self.net4 = nn.Linear(64, 8)
        self.net5 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = F.relu(self.net4(x))
        return F.sigmoid(self.net5(x))


def _init_model(rank, world_size):
    device_mesh = init_device_mesh(DEVICE, (world_size,))
    model = Model().cuda()
    model = FSDP(model, device_mesh=device_mesh, use_orig_params=True)
    optim = tp.optim.Adam(model.parameters(), lr=0.0001)
    _patch_model_state_dict(model)
    _patch_optimizer_state_dict(model, optimizers=optim)
    return model, optim


def _print(msg):
    if dist.get_rank() == 0:
        print(msg)


def _input():
    x = tp.rand(128, 8, device="cuda")
    y = tp.zeros(128, 1, device="cuda")
    y[tp.sum(x, dim=1) >= 4] = 1.0
    return x, y


def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    tp.cuda.set_device(rank)
    model, optim = _init_model(rank, world_size)
    state_dict = {"model": model, "optim": optim}
    loss_calc = nn.BCELoss()
    future = None
    for epoch in range(NUM_EPOCHS):
        try:
            tp.manual_seed(epoch)
            x, y = _input()
            loss = loss_calc(model(x), y)
            _print(f"{epoch=} {loss=}")
            loss.backward()
            optim.step()
            optim.zero_grad()
            if epoch % SAVE_PERIOD == 0:
                if future is not None:
                    if not isinstance(future, Future):
                        raise AssertionError("future must be a Future instance")
                    future.result()
                future = dcp.async_save(state_dict, checkpoint_id=CHECKPOINT_DIR)
            if FAULT_PERIOD > 0 and epoch % FAULT_PERIOD == 0:
                raise InjectedException("fault injection")
        except InjectedException as error:
            dist.barrier()
            _print("trainer encountered an exception")
            traceback.print_tb(error.__traceback__)
            _print("reloading model from the last checkpoint")
            if future is not None:
                if not isinstance(future, Future):
                    raise AssertionError("future must be a Future instance") from None
                future.result()
            dcp.load(state_dict, checkpoint_id=CHECKPOINT_DIR)


if __name__ == "__main__":
    world_size = tp.cuda.device_count()
    print(f"running async checkpoint example on {world_size} devices")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
