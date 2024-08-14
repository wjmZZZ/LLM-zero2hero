import io
import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ====================================================
# Seed Initialization
# ====================================================
from transformers import set_seed


def seed_everything(seed: int = 42) -> None:
    """
    Set the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


# ====================================================
# Logging Configuration
# ====================================================
from colorlog import ColoredFormatter


class LocalRankFilter(logging.Filter):
    def filter(self, record):
        """
        Filter log records based on the local rank of the current process.

        Args:
            record (logging.LogRecord): The log record to filter.

        Returns:
            bool: True if the local rank is 0, False otherwise.
        """
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        return local_rank == 0


def get_logger(log_file_name=None):
    """
    Configure and get a logger instance.

    Args:
        log_file_name (str, optional): The name of the log file. Defaults to None.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear existing handlers

    # Local rank filter
    local_rank_filter = LocalRankFilter()

    # Console log format with colors
    console_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)-8s%(reset)s> %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )

    # File log format without colors
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s> %(message)s", datefmt="%Y-%m-%d %H:%M"
    )

    # Console log handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(local_rank_filter)  # Add local rank filter
    logger.addHandler(console_handler)

    # File log handler
    if log_file_name is not None:
        file_handler = logging.FileHandler(filename=log_file_name)
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(local_rank_filter)  # Add local rank filter
        logger.addHandler(file_handler)

    return logger


class TqdmTologger(io.StringIO):
    """
    Output stream for TQDM that redirects output to the logger module instead of StdOut.
    """

    logger: logging.Logger = None
    level: int = None
    buf = ""

    def __init__(self, logger):
        super(TqdmTologger, self).__init__()
        self.logger = logger
        self.level = logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t [A")

    def flush(self):
        if self.buf != "":
            try:
                self.logger.log(self.level, self.buf)
            except NameError:
                pass


def flatten_dict(nested_dict):
    """
    Flatten a nested dictionary.

    Args:
        nested_dict (dict): The nested dictionary to flatten.

    Returns:
        dict: The flattened dictionary.
    """
    flat_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            for k, v in value.items():
                flat_dict[k] = v
        else:
            flat_dict[key] = value
    return flat_dict
