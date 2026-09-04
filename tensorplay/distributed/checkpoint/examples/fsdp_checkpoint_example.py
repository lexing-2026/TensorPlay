import os
import shutil

import tensorplay as tp
import tensorplay.distributed as dist
import tensorplay.distributed.checkpoint as dist_cp
import tensorplay.multiprocessing as mp
from tensorplay.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from tensorplay.distributed.fsdp import FullyShardedDataParallel as FSDP
from tensorplay.distributed.fsdp.api import StateDictType


CHECKPOINT_DIR = f"/scratch/{os.environ.get('LOGNAME', '')}/checkpoint"


def opt_at(opt, idx):
    return list(opt.state.values())[idx]


def init_model():
    model = FSDP(tp.nn.Linear(4, 4).cuda(dist.get_rank()))
    optim = tp.optim.Adam(model.parameters(), lr=0.1)
    model(tp.rand(4, 4)).sum().backward()
    optim.step()
    return model, optim


def print_params(stage, model_1, model_2, optim_1, optim_2):
    with FSDP.summon_full_params(model_1), FSDP.summon_full_params(model_2):
        print(
            f"{stage} --- rank: {dist.get_rank()}\n"
            f"model.weight: {model_1.weight}\n"
            f"model_2.weight: {model_2.weight}\n"
            f"model.bias: {model_1.bias}\n"
            f"model_2.bias: {model_2.bias}\n"
        )
    print(
        f"{stage} --- rank: {dist.get_rank()}\n"
        f"optim exp_avg: {opt_at(optim_1, 0)['exp_avg']}\n"
        f"optim_2 exp_avg: {opt_at(optim_2, 0)['exp_avg']}\n"
        f"optim exp_avg_sq: {opt_at(optim_1, 0)['exp_avg_sq']}\n"
        f"optim_2 exp_avg_sq: {opt_at(optim_2, 0)['exp_avg_sq']}\n"
    )


def run_fsdp_checkpoint_example(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    tp.cuda.set_device(rank)
    model_1, optim_1 = init_model()
    with FSDP.state_dict_type(model_1, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model_1.state_dict(),
            "optim": FSDP.optim_state_dict(model_1, optim_1),
        }
        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
        )
    model_2, optim_2 = init_model()
    print_params("before loading", model_1, model_2, optim_1, optim_2)
    with FSDP.state_dict_type(model_2, StateDictType.SHARDED_STATE_DICT):
        state_dict = {"model": model_2.state_dict()}
        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
        )
        model_2.load_state_dict(state_dict["model"])
        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=state_dict["model"],
            optimizer_key="optim",
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
        )
        flattened_osd = FSDP.optim_state_dict_to_load(
            model_2, optim_2, optim_state["optim"]
        )
        optim_2.load_state_dict(flattened_osd)
    print_params("after loading", model_1, model_2, optim_1, optim_2)
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = tp.cuda.device_count()
    print(f"running fsdp checkpoint example on {world_size} devices")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    mp.spawn(
        run_fsdp_checkpoint_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
