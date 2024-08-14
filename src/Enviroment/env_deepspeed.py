import json
import logging

logger = logging.getLogger(__name__)


def get_ds_config(args):
    """
    Generate DeepSpeed configuration based on provided arguments.

    Args:
        args: An object containing model, training, and environment arguments.

    Returns:
        dict: A dictionary containing the DeepSpeed configuration.
    """
    ds_config = {
        "fp16": {
            "enabled": args.model_args.backbone_dtype == "float16",
            "loss_scale_window": 100,
        },
        "bf16": {
            "enabled": args.model_args.backbone_dtype == "bfloat16",
            "loss_scale_window": 100,
        },
        "zero_force_ds_cpu_optimizer": False,
        "zero_optimization": {
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": args.env_args.deepspeed_reduce_bucket_size,
        },
        "steps_per_print": 2000,
        "train_micro_batch_size_per_gpu": args.training_args.batch_size,
        "gradient_accumulation_steps": args.training_args.grad_accumulation,
        "wall_clock_breakdown": False,
    }

    if args.env_args.deepspeed_method == "ZeRO2":
        ds_config["zero_optimization"].update(
            {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": args.env_args.deepspeed_allgather_bucket_size,
            }
        )
    elif args.env_args.deepspeed_method == "ZeRO3":
        ds_config["zero_optimization"].update(
            {
                "stage": 3,
                "stage3_prefetch_bucket_size": args.env_args.deepspeed_stage3_prefetch_bucket_size,
                "stage3_param_persistence_threshold": args.env_args.deepspeed_stage3_param_persistence_threshold,
                "stage3_gather_16bit_weights_on_model_save": True,
            }
        )

    logger.info(
        f"""
    🔧 DeepSpeed配置:
    {ds_config}
    """
    )
    return ds_config


def get_ds_config_from_file(config_file: str):
    """
    Load DeepSpeed configuration from a JSON file.

    Args:
        config_file (str): Path to the JSON configuration file.

    Returns:
        dict: A dictionary containing the DeepSpeed configuration.
    """
    with open(config_file, "r") as f:
        ds_config = json.load(f)
    return ds_config
