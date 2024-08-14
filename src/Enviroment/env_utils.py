import logging
import os
import random
from typing import Any, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ====================================================
# seed
# ====================================================
from transformers import set_seed


def seed_everything(seed: int = 42) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def sync_across_processes(
    t: Union[torch.Tensor, np.ndarray], world_size: int, group: Any = None
) -> Union[torch.Tensor, np.ndarray]:
    """Concatenates tensors across processes.

    Args:
        t: input tensor or numpy array
        world_size: world size
        group: The process group to work on

    Returns:
        Tensor or numpy array concatenated across all processes
    """

    torch.distributed.barrier()

    if isinstance(t, torch.Tensor):
        gather_t_tensor = [torch.ones_like(t) for _ in range(world_size)]

        if t.is_cuda:
            torch.distributed.all_gather(gather_t_tensor, t)
        else:
            torch.distributed.all_gather_object(gather_t_tensor, t, group=group)

        ret = torch.cat(gather_t_tensor)
    elif isinstance(t, np.ndarray):
        gather_t_array = [np.ones_like(t) for _ in range(world_size)]
        torch.distributed.all_gather_object(gather_t_array, t, group=group)
        ret = np.concatenate(gather_t_array)  # type: ignore
    else:
        raise ValueError(f"Can't synchronize {type(t)}.")

    return ret
