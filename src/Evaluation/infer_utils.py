import types
from typing import DefaultDict, Dict, Union

import numpy as np
import torch


def batch_decode(args, output: Dict) -> Dict:
    if "predicted_answer_ids" in output.keys():
        predicted_text = []

        for ids in output["predicted_answer_ids"]:
            decoded_text = args.tokenizer.decode(ids, skip_special_tokens=True).strip()
            # print(f'ids: {ids}, decoded_text:{decoded_text}')
            predicted_text.append(decoded_text)
        output["predicted_text"] = np.array(predicted_text)
        del output["predicted_answer_ids"]
    return output


def contains_nan(output: Dict):
    return (
        sum(
            [
                1
                for key, val in output.items()
                if isinstance(val, torch.Tensor)
                and torch.isnan(val.detach().cpu()).sum() > 0
            ]
        )
        > 0
    )


def no_type_check(arg):
    """Decorator to indicate that annotations are not type hints.

    The argument must be a class or function; if it is a class, it
    applies recursively to all methods and classes defined in that class
    (but not to methods defined in its superclasses or subclasses).

    This mutates the function(s) or class(es) in place.
    """
    if isinstance(arg, type):
        arg_attrs = arg.__dict__.copy()
        for attr, val in arg.__dict__.items():
            if val in arg.__bases__ + (arg,):
                arg_attrs.pop(attr)
        for obj in arg_attrs.values():
            if isinstance(obj, types.FunctionType):
                obj.__no_type_check__ = True
            if isinstance(obj, type):
                no_type_check(obj)
    try:
        arg.__no_type_check__ = True
    except TypeError:  # built-in classes
        pass
    return arg


@no_type_check
def cat_batches(
    data: DefaultDict[str, Union[torch.Tensor, np.ndarray]]
) -> DefaultDict[str, Union[torch.Tensor, np.ndarray]]:
    """Concatenates output data from several batches

    Args:
        data: dict with keys and list of batch outputs

    Returns:
        Concatenated dict

    """

    for key, value in data.items():
        if len(value[0].shape) == 0:
            if isinstance(value[0], torch.Tensor):
                data[key] = torch.stack(value)
            else:
                data[key] = np.stack(value)
        else:
            if isinstance(value[0], torch.Tensor):
                data[key] = torch.cat(value, dim=0)
            else:
                data[key] = np.concatenate(value, axis=0)

    return data
