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


class UnkeForDirectOpt:
    def __init__(self):
        self.dataset = defaultdict()
        self.dataset = self.get_dataset()

    def get_dataset(self):
        raw_dataset = json.load(open("../datasets/UnKE/final_data_v3.json"))
      
        edit_dict = {"question": [], "para_question": [], "answer": []}
        for i in range(len(raw_dataset)):
            for k in edit_dict:
                edit_dict[k].append(raw_dataset[i][k])
        edit_dataset = Dataset.from_dict(edit_dict) 

        return edit_dataset

    