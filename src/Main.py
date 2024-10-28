import gc
import logging
import time
import warnings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore")

from src.Args.base_args import get_args, save_args
from src.Args.sft_args import SFTArguments  
from src.Args.dpo_args import DPOArguments
from src.Dataset.dataset import (get_train_dataloader, get_valid_dataloader,
                             load_data)
from src.Dataset.sft_dataset import LLM_Dataset
from src.Dataset.dpo_dataset import DPO_Dataset
from src.Enviroment.env import (Prepare_environment, check_disk_space,
                            wrap_model_distributed)
from src.Model.sft_model import LLM
from src.Model.dpo_model import DPO_LLM
from src.Model.tokenizer import get_tokenizer
from src.Train.lr_scheduler import get_scheduler
from src.Train.metric import get_metric
from src.Train.optimizer import get_optimizer
from src.Train.train import LLM_train
from src.Train.train_utils import calculate_steps, compile_model
from src.Utils.utils import get_logger

logger = logging.getLogger(__name__)


def main():
    # Load arguments
    exp_args, data_args, model_args, training_args, infer_args, env_args = get_args()
    
    if exp_args.task == "SFT":
        args = SFTArguments(
            exp_args, data_args, model_args, training_args, infer_args, env_args
        )
    elif exp_args.task == "DPO":
        args = DPOArguments(
            exp_args, data_args, model_args, training_args, infer_args, env_args
        )

    logger = get_logger(args)
    logger.info("🚀 Loading arguments")

    # Configure environment
    Prepare_environment(args)

    # Prepare training and validation data
    logger.info("📊 Preparing data")
    train_data, valid_data = load_data(args)

    args.tokenizer = get_tokenizer(args)
    
    if args.exp_args.task == "SFT":
        train_dataset = LLM_Dataset(conversations=train_data, args=args)
        valid_dataset = LLM_Dataset(conversations=valid_data, args=args, mode="validation")
    elif args.exp_args.task == "DPO":
        train_dataset = DPO_Dataset(conversations=train_data, args=args)
        valid_dataset = DPO_Dataset(conversations=valid_data, args=args, mode="validation")

    train_dataloader = get_train_dataloader(train_dataset, args=args)
    valid_dataloader = get_valid_dataloader(valid_dataset, args=args)

    # Calculate steps
    total_training_steps, validation_steps = calculate_steps(
        args, train_dataloader, valid_dataloader
    )

    logger.info(f"📈 Total training steps: {total_training_steps}")
    logger.info(f"🔄 Validation interval (steps): {validation_steps}")

    # Save a copy of the arguments
    save_args(args)

    if args.exp_args.task == "SFT":
        model = LLM(args)
    elif args.exp_args.task == "DPO":
        model = DPO_LLM(args)

    # Check if there is enough disk space
    check_disk_space(model, args.exp_args.output_dir, args.env_args.use_deepspeed)

    model.to(args.env_args._device)

    optimizer = get_optimizer(model=model, args=args)
    scheduler = get_scheduler(args=args, optimizer=optimizer)

    metric_func = get_metric(args)

    # If distributed training, wrap the model for distributed training
    if args.env_args._distributed:
        (
            model,
            optimizer,
            train_dataloader,
            valid_dataloader,
            scheduler,
        ) = wrap_model_distributed(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            args=args,
        )

    # Optional, compile for acceleration, pytorch > 2.0
    model = compile_model(model, args)

    # Step recording
    args.env_args._curr_step = 0
    args.env_args._curr_val_step = 0

    gc.collect()

    # Record global start time
    start_time = time.time()
    global_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger.info(f"")
    logger.info(f"🏁 Global start time: {global_start_time}")
    logger.info(f"")

    LLM_train(
        args,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        valid_dataloader,
        valid_data,
        metric_func,
    )

    global_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    end_time = time.time()  # Record the timestamp of the end time
    total_runtime = end_time - start_time  # Calculate total runtime

    logger.info(f"")
    logger.info(f"🏁 Global end time: {global_end_time}")
    logger.info(f"⌛️ Total runtime: {total_runtime:.2f} seconds")
    logger.info(f"")


if __name__ == "__main__":
    main()
