import csv
import random
from collections import defaultdict
from pathlib import Path
import copy
import json
import torch
from datasets import load_dataset, Dataset
import transformers
import pdb


random.seed(42)


def dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v

    return new_dict


class BigFiveForDirectOpt:
    def __init__(self):
        self.dataset = defaultdict()
        self.dataset = self.get_dataset()

    def get_dataset(self):
        edit_dataset = Dataset.from_file("./datasets/bigfive/train/data-00000-of-00001.arrow")
        edit_dict = {"instruction": [], "input": [], "output": [], "trait": []}
        for i in range(len(edit_dataset)):
            if edit_dataset[i]["train_instruction"] is None or edit_dataset[i]["train_input"] is  None or edit_dataset[i]["train_output"] is None:
                continue
            else:
                edit_dict["instruction"].append(edit_dataset[i]["train_instruction"])
                edit_dict["input"].append(edit_dataset[i]["train_input"])
                edit_dict["output"].append(edit_dataset[i]["train_output"])
                edit_dict["trait"].append(edit_dataset[i]["trait"])
        edit_dataset = Dataset.from_dict(edit_dict)
       
        return edit_dataset