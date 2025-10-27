import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# imports modules for registration
from controlcap.tasks import *
from controlcap.datasets import *
from controlcap.models import *
from controlcap.runners import *
from controlcap.common.config import Config

import lavis.tasks as tasks
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--local-rank", default=-1, type=int) # for debug
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--inference-gt-path", default=None,
                        help="(Optional) alternate COCO-style JSON used ONLY for building inference samples (bboxes/segmentations). Metrics still use original GT.")

    args = parser.parse_args()
    return args

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("task", "controlcap"))

    return runner_cls

def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)

    # If an inference GT override is provided, attach it to task and patch the dataset config
    # so that the builder will create eval datasets from the alternate ann file (used only for inference samples).
    inf_gt = getattr(cfg.args, "inference_gt_path", None)
    if inf_gt is not None:
        # attach for downstream visibility
        task.inference_gt_path = inf_gt
        try:
            import copy, logging
            # determine eval dataset name (task may have been constructed with eval_dataset_name)
            eval_name = task.eval_dataset_name or list(cfg.datasets_cfg)[0]
            # deep copy target dataset cfg before mutating
            ds_cfg = copy.deepcopy(getattr(cfg.datasets_cfg, eval_name))
            if hasattr(ds_cfg.build_info, "annotations") and hasattr(ds_cfg.build_info.annotations, "val"):
                ds_cfg.build_info.annotations.val = [inf_gt]
                setattr(cfg.datasets_cfg, eval_name, ds_cfg)
                logging.info(f"Overrode eval annotations for dataset [{eval_name}] with {inf_gt} for inference samples.")
            else:
                logging.warning(f"Could not locate build_info.annotations.val for dataset [{eval_name}] to override inference GT.")
        except Exception as e:
            import logging
            logging.warning(f"Failed to apply inference GT override ({inf_gt}): {e}")
    
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
