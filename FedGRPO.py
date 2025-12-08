import logging
import os
import sys
from dataclasses import dataclass, field
import random
import datasets
import torch
import transformers
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from typing import Optional

from configs import GRPOConfig
from rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
    model_reward,
    boxed_format_reward,  # Use the modified format_reward function
)
from utils import get_tokenizer
from utils.callbacks import get_callbacks
from utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from exp_dataset.pre_datasets import get_dataset, get_dataset_fedgrpo
from utils.trainer_1117 import FedGRPOTrainer

# os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )
    dataset_ratio: float = field(
        default=0.0,
        metadata={"help": "dataset ration for fedgrpo"},
    )
    max_num_train_samples: int = field(
        default=-1,
        metadata={"help": "Chose certain samples for fast check"},
    )

    num_clients: int = field(
        default=10,
        metadata={"help": "The number of federated learning clients."},
    )
    num_group: int = field(
        default=2,
        metadata={"help": "The number of selected groups from clients."},
    )


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)
        
        # wandb.init(project='fedgrpo')

    # dataset = get_dataset_fedgrpo(script_args.dataset_name, local_data_dir = '../llm_datasets')
    # dataset = get_dataset_fedgrpo(script_args.dataset_name, local_data_dir = '../llm_datasets')
    dataset = get_dataset(script_args.dataset_name)
    # dataset = get_dataset_classification('facebook/anli', split='train_r1')

    def assign_client_ids(example, num_clients, num_group):
        """为每个样本分配num_group个不同的client_id"""
        example["client_ids"] = random.sample(range(num_clients), num_group)
        return example

    dataset = dataset.map(
        assign_client_ids, 
        fn_kwargs={"num_clients": script_args.num_clients, "num_group": script_args.num_group}
    )
    
    import pprint
    pprint.pprint(dataset[0])
    # Note: Use a small dataset for fast check. Should be commented in production!
    # Make sure it's called after data preprocessing
    if script_args.max_num_train_samples is not None and script_args.max_num_train_samples > 0:
        random.seed(training_args.seed)
        num_samples = min(script_args.max_num_train_samples, len(dataset))
        sample_ids = random.sample(range(len(dataset)), num_samples)
        dataset = dataset.select(sample_ids) 
      
    

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    # -----------------------------------
    # # Add special tokens when necessary?
    # print("-"*100)
    # special_tokens = ["<think>", "</think>", "<answer>", "</answer>"]
    # for token in special_tokens:
    #     token_id = tokenizer.convert_tokens_to_ids(token)
    #     # If token is not included in vocabulary, add it
    #     if token_id == tokenizer.unk_token_id:
    #         tokenizer.add_tokens([token])
    #         logger.warning(f"'{token}' is not in the vocabulary, add it now.")
    #     else:
    #         logger.info(f"'{token}' is already in the vocabulary.")
    # tokenizer.save_pretrained('output_models/fedgrpo_2507/tokenizer_0710')
    # print('tokenizer saved successfully!')
    # print("-"*100)

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        "model_reward": model_reward,
        "boxed_format": boxed_format_reward,  # Use the modified format_reward function
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Format into conversation
    def make_conversation(example):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        # print('keys:', example.keys())

        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    eval_dataset = None
    # eval_dataset_name='nlile/hendrycks-MATH-benchmark'
    # eval_dataset_name='HuggingFaceH4/MATH-500'
    eval_dataset_name = None
    if eval_dataset_name:
        eval_dataset = get_dataset(
            eval_dataset_name, split='test'
        )
        eval_dataset=eval_dataset.map(make_conversation)

    if "messages" in dataset.column_names:
            dataset = dataset.remove_columns("messages")
    print('dataset[0]')
    pprint.pprint(dataset[0])
    # for split in dataset:
    #     if "messages" in dataset[split].column_names:
    #         dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs


    # # 修改model的embedding size以适应新增的tokens
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    # model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # print(f"Updated model token embedding: {model.model.embed_tokens}")
    # if model.config.vocab_size is not None:
    #     assert len(tokenizer) <= model.config.vocab_size, "Mismatch: model vocab_size < tokenizer vocab_size"

    #############################
    # Initialize the GRPO trainer
    #############################
    from peft import LoraConfig, TaskType
    # peft_config2 = LoraConfig(
    # r=8,
    # lora_alpha=16,
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    # lora_dropout=0.05,
    # bias="none",
    # task_type="CAUSAL_LM"
    # )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,  # 训练模式
        r=64,  # Lora 秩
        lora_alpha=16,  # Lora alaph
        lora_dropout=0.1,  # Dropout 比例
    )
    # print('training_args.dataset_text_field:', training_args.dataset_text_field)
    if '14B' in model_args.model_name_or_path:
        trainer = FedGRPOTrainer(
            model=model_args.model_name_or_path,
            reward_funcs_name=script_args.reward_funcs,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset, #[script_args.dataset_train_split],
            # eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            # peft_config=get_peft_config(model_args),
            peft_config = peft_config,
            callbacks=get_callbacks(training_args, model_args),
            processing_class=tokenizer,
            num_clients=script_args.num_clients,  # FL参数
            num_group=script_args.num_group,      # FL参数
        )
    else:
        trainer = FedGRPOTrainer(
            model=model_args.model_name_or_path,
            reward_funcs_name=script_args.reward_funcs,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset, #[script_args.dataset_train_split],
            eval_dataset=eval_dataset,
            # eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            # peft_config=get_peft_config(model_args),
            # peft_config = peft_config,
            callbacks=get_callbacks(training_args, model_args),
            processing_class=tokenizer,
            num_clients=script_args.num_clients,  # FL参数
            num_group=script_args.num_group,      # FL参数
        ) 
        # trainer = FedGRPOTrainer_RewardModel(
        #     model=model_args.model_name_or_path,
        #     reward_funcs=reward_funcs,
        #     args=training_args,
        #     train_dataset=dataset, #[script_args.dataset_train_split],
        #     # eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        #     # peft_config=get_peft_config(model_args),
        #     # peft_config = peft_config,
        #     callbacks=get_callbacks(training_args, model_args),
        #     processing_class=tokenizer,
        # )         

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    # training_args.resume_from_checkpoint = 'output_models/fedgrpo/Qwen2.5-3B-Instruct-mathlight-fedgrpo-len-0620/checkpoint-234'
    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # elif last_checkpoint is not None:
    #     checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset) #len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
        # 0710  Save tokenizer for the added special tokens (<think>, </think>, <answer>, </answer>)
        # tokenizer.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()
    #     metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # #############
    # # push to hub
    # #############
    # if training_args.push_to_hub:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
