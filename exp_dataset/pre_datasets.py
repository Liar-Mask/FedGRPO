import datasets
import random
from datasets import load_dataset, load_from_disk
import pandas as pd
import os



def get_dataset_new(dataset_name, split='train', system_prompt=None):

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
        from datasets import load_from_disk
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
    if '46k' in dataset_name:
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.shard(num_shards=6, index=0)
        print('dataset #zgx', dataset)
        print(f"所用训练集样本数: {len(dataset)}")
        dataset = dataset.remove_columns(['problem_type', 'question_type', \
         'source', 'uuid','is_reasoning_complete', 'generations', 'correctness_math_verify',\
         'correctness_llama', 'finish_reasons', 'correctness_count', 'messages'])  
    
    columns = dataset.column_names
    
    # Check if 'problem' is in columns (may change it accordingly):
    if 'problem' not in columns:
        for feture in columns:
            # 'problem', 'query' can be considered as 'problem' column
            # (may change or add columns accordingly)
            if feture.lower() in ['question', 'problem', 'query']:
                dataset = dataset.rename_column(feture, 'problem')
                break
        else:
            raise ValueError("No column named 'problem' in the datset!")
    
    # Check if 'solution' is in columns:
    if 'solution' not in columns:
        for feture in columns:
            # 'answer', 'response' can be considered as 'solution' column
            if feture.lower() in ['answer', 'solution', 'response']:
                dataset = dataset.rename_column(feture, 'solution')
                break
        else:
            raise ValueError("No column named 'solution' in the datset!")
    
    # if "messages" in dataset.column_names:
    #     dataset = dataset.remove_columns("messages")
    
    return dataset

def get_dataset(dataset_name, split='train', system_prompt=None):

    # Check if the input is a local file or directory
    if os.path.exists(dataset_name):
        print(f"Loading local dataset from: {dataset_name}")
        dataset = load_from_disk(dataset_name)
        if '46k' not in dataset_name and 'train' not in dataset_name and 'test' not in dataset_name:
            dataset = dataset['train']

        # seed=42
        # from datasets import load_from_disk
        # if '46k' in dataset_name:
        #     dataset = load_from_disk(dataset_name)
        # else:
        #     dataset = load_dataset(dataset_name)
        #     dataset = dataset['train']
    else:
        print(f"Loading remote dataset from Hugging Face Hub: {dataset_name}")
        # Change the following accordingly
        if dataset_name == 'openai/gsm8k':
            dataset = load_dataset(dataset_name, name='main', split=split)
        elif dataset_name == 'opencompass/AIME2025':
            dataset = load_dataset(dataset_name, name='AIME2025-I', split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
    if '46k' in dataset_name:
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.shard(num_shards=6, index=0)
        print('dataset #zgx', dataset)
        print(f"所用训练集样本数: {len(dataset)}")
        dataset = dataset.remove_columns(['problem_type', 'question_type', \
         'source', 'uuid','is_reasoning_complete', 'generations', 'correctness_math_verify',\
         'correctness_llama', 'finish_reasons', 'correctness_count', 'messages'])  
    
    columns = dataset.column_names
    print('#zgx column_names:', columns)
    import pprint
    print('#zgx Dataset example:')
    pprint.pprint(dataset[0])
    
    # Check if 'problem' is in columns (may change it accordingly):
    if 'problem' not in columns:
        for feture in columns:
            # 'problem', 'query' can be considered as 'problem' column
            # (may change or add columns accordingly)
            if feture.lower() in ['question', 'problem', 'query']:
                dataset = dataset.rename_column(feture, 'problem')
                break
        else:
            raise ValueError("No column named 'problem' in the datset!")
    
    # Check if 'solution' is in columns:
    if 'solution' not in columns:
        for feture in columns:
            # 'answer', 'response' can be considered as 'solution' column
            if feture.lower() in ['answer', 'solution', 'response']:
                dataset = dataset.rename_column(feture, 'solution')
                break
        else:
            raise ValueError("No column named 'solution' in the datset!")
    return dataset

def get_dataset_old(dataset_name, name, local_data_dir=None):

    if dataset_name in ["gsm8k"]:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train", name="main")
    elif dataset_name in ["nlile/hendrycks-MATH-benchmark"]:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train")
    elif dataset_name in ["MATH-lighteval"]:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train", name=name)
    elif dataset_name in ["OpenR1-Math-220k"]:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train", name=name)
    elif dataset_name in ["lighteval/MATH"]:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train", name="all")
    elif dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train_sft")
    else:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        # dataset = load_dataset(dataset_name, split="train")
        dataset = load_from_disk(dataset_name)
        dataset = dataset['train']
        dataset = dataset.shuffle(seed=42)
        
        if 'Math' in dataset_name and 'Open' in dataset_name:
            # For OpenR1-Math datasets, we need to shuffle and shard the dataset
            sub_dataset = dataset.shard(num_shards=10, index=0, contiguous=True)
            dataset = sub_dataset

    if 'problem' not in columns:
        for feture in columns:
            # 'problem', 'query' can be considered as 'problem' column
            # (may change or add columns accordingly)
            if feture.lower() in ['question', 'problem', 'query']:
                dataset = dataset.rename_column(feture, 'problem')
                break
        else:
            raise ValueError("No column named 'problem' in the datset!")
    
    # Check if 'solution' is in columns:
    if 'solution' not in columns:
        for feture in columns:
            # 'answer', 'response' can be considered as 'solution' column
            if feture.lower() in ['answer', 'solution', 'response']:
                dataset = dataset.rename_column(feture, 'solution')
                break
        else:
            raise ValueError("No column named 'solution' in the datset!")
    return dataset


def get_dataset_fedgrpo(dataset_name, local_data_dir=None):
    # load dataset for FedGRPO
    if dataset_name == "DigitalLearningGmbH/MATH-lighteval":
        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.remove_columns(['type','level'])
    elif dataset_name == "OpenR1-Math-220k":
        if local_data_dir is not None:
            dataset_name = local_data_dir + dataset_name 
            dataset = load_from_disk(dataset_name)
            dataset = dataset['train']
        else:
            dataset = load_dataset(dataset_name, split="train")

        dataset = dataset.remove_columns(['source', 'uuid','is_reasoning_complete', 'generations', 'correctness_math_verify',\
         'correctness_llama', 'finish_reasons', 'correctness_count', 'messages']) 
        dataset = dataset.shuffle(seed=42)
        sub_dataset = dataset.shard(num_shards=10, index=0, contiguous=True)
        dataset = sub_dataset

    return dataset

def get_grpo_reward_dataset(dataset_name, chosen_types, client_id, split='train'):

    # Check if the input is a local file or directory
    if os.path.exists(dataset_name):
        print(f"Loading local dataset from: {dataset_name}")
        dataset = load_from_disk(dataset_name)
        # Load local dataset
        # dataset = load_dataset(
        #     'json' if dataset_name.endswith('.jsonl') else 'csv',  # Infer format
        #     data_files=dataset_name,
        #     split=split
        # )

        # if '46k' in dataset_name:
        #     dataset = load_from_disk(dataset_name)
        # else:
        #     dataset = load_dataset(dataset_name)
        #     dataset = dataset['train']
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
    if '46k' in dataset_name:
        dataset = dataset.shard(num_shards=6, index=0)
        print('dataset #zgx', dataset)
        print(f"所用训练集样本数: {len(dataset)}")
        dataset = dataset.remove_columns(['problem_type', 'question_type', \
         'source', 'uuid','is_reasoning_complete', 'generations', 'correctness_math_verify',\
         'correctness_llama', 'finish_reasons', 'correctness_count', 'messages'])  
    
    columns = dataset.column_names
    
    # Check if 'problem' is in columns (may change it accordingly):
    if 'problem' not in columns:
        for feture in columns:
            # 'problem', 'query' can be considered as 'problem' column
            # (may change or add columns accordingly)
            if feture.lower() in ['question', 'problem', 'query']:
                dataset = dataset.rename_column(feture, 'problem')
                break
        else:
            raise ValueError("No column named 'problem' in the datset!")
    
    # Check if 'solution' is in columns:
    if 'solution' not in columns:
        for feture in columns:
            # 'answer', 'response' can be considered as 'solution' column
            if feture.lower() in ['answer', 'solution', 'response']:
                dataset = dataset.rename_column(feture, 'solution')
                break
        else:
            raise ValueError("No column named 'solution' in the datset!")
        
    # 根据problem_type筛选样本，划分为两份
    for type_title_ in ['subject', 'problem_type']:
        if type_title_ in columns:
            type_title = type_title_

    # 根据传入的 chosen_type 过滤数据集
    chosen_types = [t.replace('_', ' ') for t in chosen_types]
    print(f"Filtering for subject(s): {chosen_types}")
    dataset = dataset.filter(lambda x: x[type_title] in chosen_types)
    print(f'Data Num for {chosen_types}: {len(dataset)}')

    # 使用shard函数将dataset连续地划分为2个相等的部分
    sub_dataset = dataset.shard(num_shards=2, index=client_id)
    print(f"Select {client_id}-th dataset size: {len(sub_dataset)}")
    if len(sub_dataset) >= 5:
        print(f"Sub_dataset sample 5 'problem': {sub_dataset[4]['problem']}")

    return sub_dataset


def process_sft_dataset(dataset_name, dataset, seed, dataset_sample=None):
    if dataset_name in ["lucasmccabe-lmi/CodeAlpaca-20k", "yahma/alpaca-cleaned", "FinGPT/fingpt-sentiment-train"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
    elif dataset_name in ["WizardLM/WizardLM_evol_instruct_70k"]:
        dataset = dataset.rename_column("output", "response")
    elif dataset_name in ["tatsu-lab/alpaca", "vicgalle/alpaca-gpt4", "gbharti/finance-alpaca"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output', 'text'], desc=f"Preprocessing {dataset_name} for unified format.")
    elif dataset_name in ["TIGER-Lab/MathInstruct"]:
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=['instruction'])
        dataset = datasets.Dataset.from_pandas(df)
        dataset = dataset.rename_column("output", "response")
        dataset = dataset.remove_columns(['source'])
    elif dataset_name in ["lighteval/MATH"] or 'MATH' in dataset_name:
        # dataset = dataset.rename_column("solution", "response")
        # dataset = dataset.rename_column("problem", "instruction")
        # dataset = dataset.remove_columns(['level', 'type'])
        dataset = dataset
    elif dataset_name in ['gsm8k']:
        dataset = dataset.rename_column("question", "instruction")
        dataset = dataset.rename_column("answer", "response")
    elif dataset_name in ['medalpaca/medical_meadow_medical_flashcards']:       # TODO: 'lavita/ChatDoctor-HealthCareMagic-100k'. not sure whether to discard the instruction.
        dataset = dataset.remove_columns(['instruction'])
        dataset = dataset.rename_column("input", "instruction")
        dataset = dataset.rename_column("output", "response")
    elif 'OpenR1-Math' in dataset_name or 'open_math' in dataset_name: 
        dataset = dataset.rename_column("solution", "response")
        dataset = dataset.rename_column("problem", "instruction")
        dataset = dataset.remove_columns(['answer', 'problem_type', 'question_type', \
         'source', 'uuid','is_reasoning_complete', 'generations', 'correctness_math_verify',\
         'correctness_llama', 'finish_reasons', 'correctness_count', 'messages'])   
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    dataset = dataset.shuffle(seed=seed)
    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    return dataset


def split_dataset(fed_args, seed, dataset):
    dataset = dataset.shuffle(seed=seed)        # Shuffle the dataset
    local_datasets = []
    if fed_args.split_strategy == "iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    # if fed_args.split_strategy == "non-iid": 
    return local_datasets

def get_dataset_this_round(dataset, round, script_args, multi_cof):
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * multi_cof #* script_args.max_steps
    num2sample = min(num2sample, len(dataset))
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)

    return dataset_this_round