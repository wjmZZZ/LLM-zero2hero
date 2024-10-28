import logging
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DPO_Dataset(Dataset):
    """
    Direct Preference Optimization Dataset Class
    """

    def __init__(self, conversations: pd.DataFrame, args: Any, mode: str = "train"):
        """
        Initialize the dataset.

        Args:
            conversations: Input DataFrame.
            args: Configuration containing all hyperparameters.
            mode: Dataset mode, can be one of {"train", "validation"}.
        """
        self.args = args
        self.mode = mode
        self.tokenizer = args.tokenizer
        self.conversations: list[list] = conversations

        logger.info(
            f"📊 Total {mode} number of conversations: {len(self.conversations)}"
        )

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict:
        """
        Read a single sample.

        Args:
            idx: Sample index.
        Returns:
            Sample dictionary.
        """
        input_text_dict = self.conversations[idx]

        sample = dict()
        system_encoding, prompt_encodings, chosen_response_encodings, rejected_response_encodings = self.get_encodings(
            input_text_dict
        )
        # Process chosen samples
        chosen_input_ids = torch.cat(
            [
                torch.cat([prompt_encoding, response_encoding])
                for prompt_encoding, response_encoding in zip(
                    prompt_encodings, chosen_response_encodings
                )
            ]
        )
        sample.update(self.get_labels(prompt_encodings, 
                                      chosen_response_encodings, 
                                      "chosen_"))
        sample.update(
            self.pad_tokens(
                chosen_input_ids,
                attention_mask=torch.ones_like(chosen_input_ids),
                max_length=self.args.training_args.max_seq_length,
                pad_token_id=self.tokenizer.pad_token_id,
                prefix="chosen_",
            )
        )        

        # Process rejected samples
        rejected_input_ids = torch.cat(
            [
                torch.cat([prompt_encoding, response_encoding])
                for prompt_encoding, response_encoding in zip(
                    prompt_encodings, rejected_response_encodings
                )
            ]
        )
        
        sample.update(self.get_labels(prompt_encodings, 
                                      rejected_response_encodings, 
                                      "rejected_"))
        sample.update(
            self.pad_tokens(
                rejected_input_ids,
                attention_mask=torch.ones_like(rejected_input_ids),
                max_length=self.args.training_args.max_seq_length,
                pad_token_id=self.tokenizer.pad_token_id,
                prefix="rejected_",
            )
        )

        # Process prompt samples
        prompt_input_ids = torch.cat(
            [
                torch.cat([prompt_encoding, torch.empty(0)])
                for prompt_encoding in prompt_encodings
            ]
        )
        sample.update(
            self.pad_tokens(
                prompt_input_ids,
                attention_mask=torch.ones_like(prompt_input_ids),
                max_length=self.args.training_args.max_seq_length,
                pad_token_id=self.tokenizer.pad_token_id,
                prefix="prompt_",
            )
        )

        return sample

    def get_encodings(self, input_text_dict: Dict[str, List[str]]):
        """
        Get encodings for a single conversation history.
        Args:
            input_text_dict: A dictionary containing the input text for a single sample.
            Contains the keys "systems", "prompts", "responses".
            System may be an empty string.
        """
        encodings = [
            self._get_sample_encoding(system, prompt, chosen_response, rejected_response)
            for idx, (system, prompt, chosen_response, rejected_response) in enumerate(
                zip(
                    input_text_dict[f"{self.args.data_args.system_column}"],
                    input_text_dict[f"{self.args.data_args.prompt_column}"],
                    input_text_dict[f"{self.args.data_args.answer_column}"],
                    input_text_dict[f"{self.args.data_args.rejected_answer_column}"],
                )
            )
        ]

        system_encoding = encodings[0][0]
        prompt_encodings = [encoding[1] for encoding in encodings]
        chosen_response_encodings = [encoding[2] for encoding in encodings]
        rejected_response_encodings = [encoding[3] for encoding in encodings]
        # concatenate system encoding with root prompt encoding
        prompt_encodings[0] = torch.cat([system_encoding, prompt_encodings[0]])
        return (
            system_encoding,
            prompt_encodings,
            chosen_response_encodings,
            rejected_response_encodings,
        )

    def _get_sample_encoding(self, system: str, prompt: str, chosen_response: str, rejected_response: str) -> List:
        if len(system) > 0:
            system_encoding = self.encode(
                self.tokenizer,
                system,
                self.args.training_args.max_seq_length,
                truncation_side="right",
            )["input_ids"]
        else:
            system_encoding = torch.empty(0)

        prompt_encoding = self.encode(
            self.tokenizer, prompt, self.args.training_args.max_seq_length, "left"
        )["input_ids"]

        chosen_response_encoding = self.encode(
            self.tokenizer,
            chosen_response,
            self.args.training_args.max_seq_length,
            truncation_side="right",
        )["input_ids"]

        rejected_response_encoding = self.encode(
            self.tokenizer,
            rejected_response,
            self.args.training_args.max_seq_length,
            truncation_side="right",
        )["input_ids"]


        return [system_encoding, prompt_encoding, chosen_response_encoding, rejected_response_encoding]

    @staticmethod
    def encode(tokenizer, text: str, max_length: int, truncation_side: str) -> Dict:
        encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        encodings["input_ids"] = encodings["input_ids"][0]
        encodings["attention_mask"] = encodings["attention_mask"][0]
        if truncation_side == "right":
            encodings["input_ids"] = encodings["input_ids"][:max_length]
            encodings["attention_mask"] = encodings["attention_mask"][:max_length]
        else:
            encodings["input_ids"] = encodings["input_ids"][-max_length:]
            encodings["attention_mask"] = encodings["attention_mask"][-max_length:]
        return encodings

    def get_labels(self, prompt_encodings, answer_encodings, prefix: str = "") -> Dict:
        """
        Generate labels with optional prefix.

        Args:
            prompt_encodings: Encodings of prompts.
            answer_encodings: Encodings of answers.
            prefix: Prefix for the labels, e.g., 'chosen_' or 'rejected_'.

        Returns:
            A dictionary containing padded labels with the specified prefix.
        """
        # Concatenate prompt and answer encodings as labels
        labels = torch.cat(
            [
                torch.cat([prompt_encoding, answer_encoding])
                for prompt_encoding, answer_encoding in zip(
                    prompt_encodings, answer_encodings
                )
            ]
        ).clone()

        # If masking prompt labels is required
        if self.args.data_args.mask_prompt_labels:
            prompt_mask = torch.cat(
                [
                    torch.cat(
                        [
                            torch.ones_like(prompt_encoding),
                            torch.zeros_like(answer_encoding),
                        ]
                    )
                    for prompt_encoding, answer_encoding in zip(
                        prompt_encodings, answer_encodings
                    )
                ]
            ).to(torch.bool)
            labels.masked_fill_(prompt_mask, -100)  # Fill prompt part with -100, keep answer part labels

        # Truncate labels to fit maximum length
        if self.args.training_args.max_seq_length < len(labels):
            labels = labels[-self.args.training_args.max_seq_length :]

        # Initialize a labels tensor filled with -100, length of maximum length
        padded_labels = torch.full((self.args.training_args.max_seq_length,), -100)

        # Fill processed labels to the end of padded_labels, ensuring no overflow
        padded_labels[-len(labels):] = labels

        # Return labels dictionary with prefix
        return {f"{prefix}labels": padded_labels}

    @staticmethod
    def pad_tokens(
        input_ids,
        attention_mask,
        max_length,
        pad_token_id,
        direction="left",
        prefix="",
    ):
        sample = {}

        if max_length < len(input_ids):
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]

        if len(input_ids) > 0:
            if direction == "left":
                sample[f"{prefix}input_ids"] = torch.full((max_length,), pad_token_id)
                sample[f"{prefix}input_ids"][-len(input_ids) :] = input_ids
                sample[f"{prefix}attention_mask"] = torch.zeros(max_length)
                sample[f"{prefix}attention_mask"][-len(input_ids) :] = attention_mask
            else:
                sample[f"{prefix}input_ids"] = torch.full((max_length,), pad_token_id)
                sample[f"{prefix}input_ids"][: len(input_ids)] = input_ids
                sample[f"{prefix}attention_mask"] = torch.zeros(max_length)
                sample[f"{prefix}attention_mask"][: len(input_ids)] = attention_mask
        else:
            # Pad everything if empty (continued pretraining)
            sample[f"{prefix}input_ids"] = torch.full((max_length,), pad_token_id)
            sample[f"{prefix}attention_mask"] = torch.zeros(max_length)

        return sample
