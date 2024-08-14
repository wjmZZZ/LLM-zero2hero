import logging
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizer

from Others.exceptions import ModelException

logger = logging.getLogger(__name__)


def get_tokenizer(args: Any) -> PreTrainedTokenizer:
    """
    Get the pre-trained tokenizer based on the configuration.

    This function initializes and configures a tokenizer using the provided configuration.
    It handles special token additions and sets up padding behavior.

    Args:
        args (Any): Configuration object containing tokenizer and environment settings.

    Returns:
        PreTrainedTokenizer: Configured pre-trained tokenizer object.

    Raises:
        ValueError: If the tokenizer initialization fails.
    """
    try:
        # Construct tokenizer initialization parameters
        kwargs = {
            "add_prefix_space": args.model_args.add_prefix_space,
            "use_fast": args.model_args.use_fast_tokenizer,
            "trust_remote_code": args.model_args.trust_remote_code,
        }

        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_args.llm_backbone, **kwargs)

        # Set padding side
        tokenizer.padding_side = getattr(
            tokenizer, "_padding_side", tokenizer.padding_side
        )

        # Add special tokens if they're missing
        _add_missing_special_tokens(tokenizer)

        return tokenizer
    except Exception as e:
        logger.error(f"Failed to initialize tokenizer: {str(e)}")
        raise ModelException("Tokenizer initialization failed") from e


def _add_missing_special_tokens(tokenizer: PreTrainedTokenizer) -> None:
    """
    Add missing special tokens to the tokenizer.

    This helper function checks for missing special tokens (EOS, BOS, PAD) and adds them if necessary.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to modify.
    """
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens(
            {"eos_token": tokenizer.eos_token or "<|endoftext|>"}
        )
        logger.warning(
            f"Added EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}"
        )

    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.warning(
            f"Added BOS token: {tokenizer.bos_token}, ID: {tokenizer.bos_token_id}"
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.unk_token
            if tokenizer.unk_token_id is not None
            else tokenizer.eos_token
        )
        logger.warning(
            f"Added PAD token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}"
        )


# TODO: Implement proper handling of stop words
