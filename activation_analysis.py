import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer
import argparse
import random
import numpy as np
import pandas as pd
import torch
import copy
import json
import yaml
from datasets import load_dataset, Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel,AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch import Tensor
import os

from nnsight import LanguageModel
from src.dataset.bigfive import BigFiveForDirectOpt
import matplotlib.pyplot as plt
import numpy as np


Target_layer=15


def preprocess_dataset(example,tokenizer,dataset_name):
    results = {
        "input_ids": [],
        "attention_mask": [],
        "output_mask": [],
    }
    if dataset_name == "bigfive":
       
        for i in range(len(example["instruction"])):
            
            instruction= example["instruction"][i]
            input= example["input"][i]
            output= example["output"][i]
            
          
            messages = [
                {"role": "user", "content": instruction + "\n" + input},
                {"role": "assistant", "content": output}
            ]
            
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                formatted_text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True  # This adds the assistant prompt
                )
          
            tokenized = tokenizer(
                formatted_text,
                padding=False,
                truncation=True,
            )
            
            # Tokenize only the output to get its length (without special tokens)
            output_tokenized = tokenizer(
                    output,
                    padding=False,
                    truncation=True,
                    add_special_tokens=False,
                )
            output_length = len(output_tokenized.input_ids)
            
            
            total_length = len(tokenized.input_ids)
            
            # Mask: 0 for input tokens, 1 for output tokens (last output_length tokens)
            output_mask = [0] * (total_length - output_length) + [1] * output_length
            
            results["input_ids"].append(tokenized.input_ids)
            results["attention_mask"].append(tokenized.attention_mask)
            results["output_mask"].append(output_mask)

    return results




class GetHookedValue:
    def __init__(self, model_name, device='cpu', checkpoint_value=None, tokenizer=None):
       
        
        self.model_name = model_name
        self.device = device
        self.tokenizer = tokenizer
        self.layer_output = []
        
        self.model = LanguageModel(model_name, device_map=device)

    def get_activations(self, inputs, cal_type="mean"):
        activations = {}
        
        input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(inputs["attention_mask"]).unsqueeze(0).to(self.device)
        output_mask = torch.tensor(inputs["output_mask"]).unsqueeze(0).to(self.device)
        
        # Use nnsight tracing - pass inputs as a dict
        with self.model.trace({"input_ids": input_ids, "attention_mask": attention_mask}, validate=False):
            layer_input = self.model.model.layers[Target_layer].input[0].save()
        
        # Get the saved activation value (shape: [batch, seq_len, hidden_dim])
        layer_activation = layer_input.value.detach().cpu()
        
        ##mean_full: mean the full input and output
        ##last: get the last token
        ##mean_response: mean the response (only output tokens)
        if cal_type == "mean_full":
            activations['layer_' + str(Target_layer)] = torch.mean(layer_activation, dim=0)
        elif cal_type == "mean_response":
            # Filter only output tokens using the mask
            output_mask_cpu = output_mask.cpu()
            output_indices = output_mask_cpu[0].nonzero(as_tuple=True)[0]
            if len(output_indices) > 0:
                output_activations = layer_activation[output_indices, :]
                activations['layer_' + str(Target_layer)] = torch.mean(output_activations, dim=0)
            else:
                activations['layer_' + str(Target_layer)] = torch.zeros(layer_activation.shape[-1])
        elif cal_type == "last":
            activations['layer_' + str(Target_layer)] = layer_activation[-1, :]
        else:
            activations['layer_' + str(Target_layer)] = layer_activation
            
        return activations

    def inference(self, text_data, cal_type="mean"):
        for data in tqdm(text_data):
            res = self.get_activations(data, cal_type=cal_type)
            self.layer_output.append(res[f'layer_{Target_layer}'])
        
        return self.layer_output    
    


def store_activations(model_name="meta-llama/Llama-3.1-8B-Instruct", dataset_name="bigfive"):


    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if dataset_name == "bigfive":
        edit_dataset = BigFiveForDirectOpt().get_dataset()
   
    dataset=edit_dataset.select(range(1000))

    dataset=dataset.map(preprocess_dataset,batched=True,remove_columns=edit_dataset.column_names,fn_kwargs={'tokenizer': tokenizer,'dataset_name':dataset_name,})
    

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


    Hook=GetHookedValue(model_name,device=device,tokenizer=tokenizer) 

    res_inference=Hook.inference(dataset, cal_type="mean_response")
    

    stored_folder=f"activation/{dataset_name}/"
    if not os.path.exists(stored_folder):
        os.makedirs(stored_folder)

    stored_path=stored_folder+f"llama_3_8b_layer{Target_layer}_no_answer_last.pt"
    print("Length of res_inference",len(res_inference))
    torch.save(res_inference,stored_path)
    

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--dataset_name", type=str, default="bigfive")
    argparse.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    args = argparse.parse_args()
    store_activations(model_name=args.model_name, dataset_name=args.dataset_name )