import os
import sys

import datasets
# import torch
import transformers
import random
from datasets import load_dataset, load_from_disk
# from transformers import set_seed
# from transformers.trainer_utils import get_last_checkpoint

# from open_r1.utils.pre_datasets import get_dataset, process_sft_dataset, split_dataset, get_dataset_this_round
# from open_r1.utils.template import get_formatting_prompts_func, TEMPLATE_DICT


# seed=42
# dataset_name = '../llm_datasets/OpenR1-Math-220k'
# dataset_name = '../llm_datasets/MATH-lighteval'


# def preprocess_dataset():
#     if mode == 'preprocess':
#         dataset_config='default'
#         dataset = get_dataset(dataset_name, dataset_config)
#         dataset = process_sft_dataset(dataset_name, dataset, seed)


#         # ===== Split the dataset into clients =====
#         # fed_args = get_fed_config()
#         local_datasets = split_dataset(fed_args, training_args.seed, dataset)
#         sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
#         print('Client data number:', sample_num_list)
#     else:
#         dataset = load_from_disk(script_args.dataset_name)
#         dataset = dataset['train']
#         dataset = dataset.shard(num_shards = 5, index = 0) # 7500/5 = 1500
#         datasets.save_to_disk("../llm_datasets/mathlight0.25-iid_split-n5")

def split_for_reward_dataset(dataset_name, save_dir, split='train'):

    # Check if the input is a local file or directory
    if os.path.exists(dataset_name):
        print(f"Loading local dataset from: {dataset_name}")
        # Load local dataset
        # dataset = load_dataset(
        #     'json' if dataset_name.endswith('.jsonl') else 'csv',  # Infer format
        #     data_files=dataset_name,
        #     split=split
        # )
        seed=42

        if '46k' in dataset_name:
            dataset = load_from_disk(dataset_name)
        else:
            dataset = load_dataset(dataset_name)
            dataset = dataset['train']
    else:
        print(f"Loading remote dataset from Hugging Face Hub: {dataset_name}")
        # Change the following accordingly
        if dataset_name == 'openai/gsm8k':
            dataset = load_dataset(dataset_name, name='main', split=split)
        elif dataset_name == 'opencompass/AIME2025':
            dataset = load_dataset(dataset_name, name='AIME2025-I', split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
            
    dataset = dataset.shuffle(seed=42)
    split_result = dataset.train_test_split(test_size=1/3, seed=42)
    train_dataset = split_result['train']   # 2/3 部分
    test_dataset = split_result['test']     # 1/3 部分

    # 保存训练集和测试集
    train_dir = os.path.join(save_dir, "train")
    test_dir = os.path.join(save_dir, "test")
    print(f"Saving train dataset to: {train_dir}")
    print(f"Saving test dataset to: {test_dir}")
    train_dataset.save_to_disk(train_dir)
    test_dataset.save_to_disk(test_dir)

dataset_name='nlile/hendrycks-MATH-benchmark'
save_dir = '../llm_datasets/MATH-benchmark-reward-grpo'
split_for_reward_dataset(dataset_name, save_dir)



