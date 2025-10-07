# README

## Overview

**RILKE** (<u>R</u>epresentation <u>I</u>ntervention for <u>L</u>ifelong <u>K</u>nowledg<u>E</u> Control) is a comprehensive training and testing framework for lifelong knowledge editing. It implements an innovative approach to model editing using ReFT (Representation Fine-Tuning) with activation-based intervention selection.

## Architecture

The system consists of two main methods: Individual Training and Clustered Training in Shared Subspace. We introduce them below:

## 1. Individual Training

### 1. Data Preparation

For individual training, we first need to store the corresponding activations at the target layer using `store_activation.py`.

Example script:

```bash
python store_activation.py --model_name meta-llama/Llama-3.1-8B-Instruct --datasetname unke_v3 --datasrc "original" 
```

Here `--datasrc` denotes whether to store activations of edit data seen during training or paraphrased data used for evaluation generalization.


### 2. Training Phase 

For **RILKE**'s individual training, we use `no_batched_train.py`. 

- **Individual Intervention Creation**: For each data point in the dataset:
  - Reinitializes the ReFT module weights
  - Trains a specialized intervention on a single question-answer pair
  - Saves intervention weights for later retrieval

- **Supported Intervention Types**:
  - `Vanilla`: Standard ReFT intervention
  - `Explicit`: Enhanced ReFT with explicit regularization, as described in Section 4.1

#### Command Line Interface

```bash
python no_batched_train.py  --record True  --adv_train_method Vanilla --wandb_project unke_single_explicit_v3 --dataset unke --output_dir single_explicit_unke --rank 4  --epochs 1000  --noise_std 0.002 --drop_out 0.05 --lambda_consistency 0.001
```

#### Key Arguments

- `--num_samples`: Number of samples from dataset to process
- `--dataset`: Dataset choice (unke, unke_v3, anyedit)
- `--adv_train_method`: Training method (Vanilla, Explicit)
- `--save_weights_dir`: Directory to save intervention weights
- `--record`: Enable W&B logging
- `--rank`: Rank of intervention module
- `--drop_out`: Dropout rate for intervention module
- `--noise_std`: Noise control term $\varepsilon$, as described in Section 4.1
- `--epochs`: Number of epochs for training intervention module
- `--learning_rate`: Learning rate for training intervention module


### 3. Testing Phase

To test **RILKE**'s performance at inference, we use `test_rep.py`. It contains the following steps:

- **Activation-Based Retrieval**:
  - Loads pre-computed activation embeddings for all training samples
  - For each test query, computes cosine similarity with stored activations
  - Selects the intervention corresponding to the most similar activation

- **Dual Query Evaluation**:
  - Tests on original questions
  - Tests on paraphrased versions to measure consistency

- **Performance Metrics**: ROUGE-L recall and BERT cosine similarity


### Command Line Interface

```bash
python test_single_rep.py --dataset unke \
 --adapter_weights_dir /Stored_weights/single_unke_qwen_layer18 \
 --activation_path /activation/unke_v3/qwen_2_5_7b_layer15_no_answer_last_original.pt \
 --original_query_activation_path  /activation/unke_v3/qwen_2_5_7b_layer15_no_answer_last_original.pt \
 --rephrased_query_activation_path /activation/unke_v3/qwen_2_5_7b_layer15_no_answer_last_rephrased.pt \
 --target_layer 18 \
 --model_name Qwen/Qwen2.5-7B-Instruct \
 --rank 4 \
 --save_path explicit_unke_1000_qwen_layer15 \
 --num_samples 1000 \
```

### Key Arguments

- `--num_samples`: Number of samples from dataset to process
- `--dataset`: Dataset choice (unke, unke_v3, anyedit)
- `--activation_path`: Path to pre-computed activations for training set
- `--save_weights_dir`: Directory to saved intervention weights
- `--rephrased_query_activation_path`: Path to pre-computed activations for paraphrased queries, used for evaluation generalization
- `--rank`: Rank of trained intervention module


We also provide a uniform interface for training and then evaluating individual settings in `train_test.py`:

```bash
python train_test.py \
 --num_samples 1000 \
 --dataset unke_v3 \
 --record True  \
 --adv_train_method Explicit \
 --wandb_project unke_single_qwen \
 --output_dir single_explicit_unke --rank 4 \
 --epochs 1000  \
 --rank 4 \
 --save_weights_dir single_unke_qwen_layer18 \
 --activation_path /activation/unke_v3/qwen_2_5_7b_layer18_no_answer_last_original.pt \
 --original_query_activation_path  /activation/unke_v3/qwen_2_5_7b_layer18_no_answer_last_original.pt \
 --rephrased_query_activation_path /activation/unke_v3/qwen_2_5_7b_layer18_no_answer_last_rephrased.pt \
 --target_layer 18 \
 --model_name Qwen/Qwen2.5-7B-Instruct
```

Additionally, to test on the MMLU dataset, first run `python store_activation_mmlu.py` to store activations of the MMLU dataset, then use `python mmlu_eval.py` to generate output on MMLU.

### Example Output

We provide an example output of our method at: `result/single_explicit_unke/unke_results.jsonl`

## 2. Clustered Training

Clustered training follows the same methodology as individual training, with an additional step to cluster representations into semantically homogeneous, size-bounded groups:

### 1. Data Preparation

For clustered training, we first need to cluster the corresponding activations at the target layer using `cluster_activation.py`.

Example script:

```bash
python cluster_activation.py 
```

The clustered data should be stored like: `output/unke/unke_v3_3_hac_similarity0.9_maxsize8_clusters_no_answer_last.json`


### 2. Cluster Training & Evaluation

The training of the clustered method remains the same as individual training. We provide an example below:

```bash
python train_cluster.py  --record True  --adv_train_method Explicit --wandb_project reft_cluster_explicit_v3 --learning_rate 2e-2 --drop_out 0.01 --noise_std 0.005 --lambda_consistency 0.001 --output_dir reft_explicit_hac_cluster  --batch_size 8 --cluster_method hac  --rank 4  --save_weights_dir unke_explicit_llama_cluster
```

And its evaluation follows:


```bash
python test_cluster_rep.py --dataset unke_v3 \
 --adapter_weights_dir /Stored_weights/unke_explicit_llama_cluster \
 --activation_path /activation/unke_v3/llama_3_8b_layer15_no_answer_last_original.pt \
 --original_query_activation_path /activation/unke_v3/llama_3_8b_layer15_no_answer_last_original.pt \
 --rephrased_query_activation_path /activation/unke_v3/llama_3_8b_layer15_no_answer_last_rephrased.pt \
 --target_layer 15 \
 --model_name meta-llama/Llama-3.1-8B-Instruct \
 --rank 4 \
 --save_path cluster_unke_test \
 --num_samples 1000 \
 --cluster_indices_path /outputs/activation/unke_v3/unke_v3_3_hac_similarity0.9_maxsize8_clusters_no_answer_last.json \
```