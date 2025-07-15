import os
import asyncio
import warnings
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel
from accelerate.utils import gather_object, broadcast_object_list, gather
from trl import GRPOTrainer
from trl import maybe_apply_chat_template, is_conversational,  apply_chat_template
from utils.profiling import profiling_context
from utils.import_utils import unwrap_model_for_generation
from utils.vllm_query import vllm_evaluate
from utils.model_utils import extract_scores
from contextlib import nullcontext

from rewards import extract_answer, compare_answer



from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

class GRPOTrainerWithRWModel(GRPOTrainer):
    """
    A variant of GRPOTrainer that uses a language model as reward model.
    Inherits from GRPOTrainer and overrides only the reward calculation logic.
    """

    def _generate_and_score_completions(self, inputs):
        # Keep most of the original method's functionality
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Use parent class method to generate completions
        completions_result = super()._generate_and_score_completions(inputs)

        # Extract completions
        completions_text = self.processing_class.batch_decode(
            completions_result["completion_ids"], 
            skip_special_tokens=True
        )
        
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Override reward calculation logic for language model rewards
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                # Language model reward calculation
                reward_system_prompt = """As a mathematical problem-solving evaluator, your output MUST strictly follow this structure:

                    ### Analysis Report
                    ```scorecard
                    Evaluate the sloution from the below three dimesions, and the score point output must use double-layer nested tags:
                    <evaluation_steps>
                    1. Step Completeness: 
                    - Analysis: (Check solution steps completeness, such as whether contain the answer, and give your score 0 or 1 below);
                    - <step_completeness><value>0 or 1</value></step_completeness>
                    2. Step-by-Step Verification: 
                    - Analysis: (Check whether the solution is thought step-by-step, and give your score 0 or 1 below);
                    - <step_by_step><value>0 or 1</value></step_by_step>
                    3. Answer Correctness: 
                    - Analysis: (Verify whether the answer is correct through your own computation, and give your score 0 or 1 below).
                    - <answer_correctness><value>0 or 1</value></answer_correctness>
                    </evaluation_steps> 
                    """

                reward_prompt = [[
                    {"role": "system", "content": reward_system_prompt},
                    {"role": "user", "content": f"As a mathematical problem-solving evaluator, your output MUST strictly follow the structure required by system. The problem and solution is as follows:\nProblem: {per_prompt[1]['content']}\nSolution to be evaluated as required: {per_completion[0]['content']}"
                    }
                ] for per_prompt, per_completion in zip(prompts, completions)]

                # Process prompts in batch
                batch_texts = [
                    reward_processing_class.apply_chat_template(
                        messages, 
                        tokenize=False,
                        add_generation_prompt=True
                    ) for messages in reward_prompt
                ]

                # Get reward scores using vLLM
                reward_texts = asyncio.run(
                    vllm_evaluate("http://localhost:8002/generate", batch_texts)
                )

                # Extract numerical scores
                output_reward_func = extract_scores(reward_texts)
                
                # Handle None values
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                
            else:
                # For non-language model rewards, use parent implementation
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather rewards across processes
        rewards_per_func = gather(rewards_per_func)
        
        # Calculate advantages using parent class logic
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.args.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Get local slice
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Update metrics
        completions_result["advantages"] = advantages
        self._update_metrics(rewards_per_func, rewards, mean_grouped_rewards, std_grouped_rewards, 
                           prompts_text, completions_text, prompt_mask)

        return completions_result

    def _update_metrics(self, rewards_per_func, rewards, mean_grouped_rewards, std_grouped_rewards,
                       prompts_text, completions_text, attention_mask):
        """Helper method to update training metrics"""
        mode = "eval" if self.control.should_evaluate else "train"

        if mode == "train":
            self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Get reward function names
        reward_func_names = []
        for reward_func in self.reward_funcs:
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            reward_func_names.append(reward_func_name)

        # Update reward metrics
        for i, reward_func_name in enumerate(reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = self.nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)

        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # Log completions if needed
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            self._log_completions(prompts_text, completions_text, rewards, reward_func_names, rewards_per_func)


from typing import Union, Any
from transformers import Trainer
class FedGRPOTrainer(GRPOTrainer):   
    """
    A variant of GRPOTrainer that supports federated learning.
    Inherits from GRPOTrainer and overrides the reward calculation logic.
    """

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        # if len(inputs) <5:
        # print('len(inputs):',len(inputs))
        # print('inputs:',inputs)
        # print('inputs[0]:',inputs[0])
        # print(inputs[0]["prompt"])
        # inputs = inputs.to(device)
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None
                sampling_params = SamplingParams(
                    n=1,  # vLLM on each GPU generates only 1 in colocate mode
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                )

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text = prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ):
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # # Compute grouped-wise rewards
        # mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        # std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        # is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # # Normalize the rewards to compute the advantages
        # mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        # std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        # advantages = rewards - mean_grouped_rewards
        # if self.scale_rewards:
        #     advantages = advantages / (std_grouped_rewards + 1e-4)


        # # Split rewards into two equal parts
        # split_point = int(len(rewards) // 2)
        # rewards_first_half = rewards[:split_point]
        # rewards_second_half = rewards[split_point:]

        # # Compute grouped-wise rewards for each half
        # mean_grouped_rewards_first = rewards_first_half.view(-1, split_point).mean(dim=1)
        # std_grouped_rewards_first = rewards_first_half.view(-1, split_point).std(dim=1)
        # is_std_zero_first = torch.isclose(std_grouped_rewards_first, torch.zeros_like(std_grouped_rewards_first))
        # mean_grouped_rewards_second = rewards_second_half.view(-1, split_point).mean(dim=1)
        # std_grouped_rewards_second = rewards_second_half.view(-1, split_point).std(dim=1)
        # is_std_zero_second = torch.isclose(std_grouped_rewards_second, torch.zeros_like(std_grouped_rewards_second))

        # # Repeat for each generation in respective halves
        # mean_grouped_rewards_first = mean_grouped_rewards_first.repeat_interleave(split_point, dim=0)
        # std_grouped_rewards_first = std_grouped_rewards_first.repeat_interleave(split_point, dim=0)
        # mean_grouped_rewards_second = mean_grouped_rewards_second.repeat_interleave(split_point, dim=0)
        # std_grouped_rewards_second = std_grouped_rewards_second.repeat_interleave(split_point, dim=0)

        # # Calculate advantages for each half
        # advantages_first = rewards_first_half - mean_grouped_rewards_first
        # advantages_second = rewards_second_half - mean_grouped_rewards_second

        # # Scale advantages if required
        # if self.scale_rewards:
        #     advantages_first = advantages_first / (std_grouped_rewards_first + 1e-4)
        #     advantages_second = advantages_second / (std_grouped_rewards_second + 1e-4)

        # # Combine the two halves of advantages
        # advantages = torch.cat([advantages_first, advantages_second], dim=0)

        # First, handle the edge case where splitting is not possible or meaningful.
        # If num_generations is 1, the advantage is always 0. The original logic works but is overkill.
        # This makes it explicit.
        if self.num_generations <= 1:
            advantages = torch.zeros_like(rewards)
            # For logging purposes, we can create dummy variables
            mean_grouped_rewards = rewards.clone()
            std_grouped_rewards = torch.zeros_like(rewards)
            is_std_zero = torch.ones_like(rewards, dtype=torch.bool)
        else:
            # 1. Reshape rewards to group them by prompt. This is the correct first step.
            # Shape: (num_prompts, num_generations)
            num_prompts = rewards.shape[0] // self.num_generations
            grouped_rewards = rewards.view(num_prompts, self.num_generations)

            # 2. Define the split point and split along the generation dimension (dim=1).
            # This handles both even and odd num_generations correctly.
            split_point_gen = self.num_generations // 2
            
            rewards_subgroup1 = grouped_rewards[:, :split_point_gen] # Shape: (num_prompts, split_point_gen)
            rewards_subgroup2 = grouped_rewards[:, split_point_gen:] # Shape: (num_prompts, num_generations - split_point_gen)

            # 3. Calculate advantage for each subgroup separately.
            # Using .unsqueeze(1) is crucial for correct broadcasting.
            
            # -- Subgroup 1 --
            mean1 = rewards_subgroup1.mean(dim=1, keepdim=True) # Shape: (num_prompts, 1)
            std1 = rewards_subgroup1.std(dim=1, keepdim=True)   # Shape: (num_prompts, 1)
            advantages1 = rewards_subgroup1 - mean1
            if self.scale_rewards:
                advantages1 = advantages1 / (std1 + 1e-4)

            # -- Subgroup 2 --
            mean2 = rewards_subgroup2.mean(dim=1, keepdim=True) # Shape: (num_prompts, 1)
            std2 = rewards_subgroup2.std(dim=1, keepdim=True)   # Shape: (num_prompts, 1)
            advantages2 = rewards_subgroup2 - mean2
            if self.scale_rewards:
                advantages2 = advantages2 / (std2 + 1e-4)

            # 4. Concatenate the advantages back together along the generation dimension.
            grouped_advantages = torch.cat([advantages1, advantages2], dim=1) # Shape: (num_prompts, num_generations)

            # 5. Flatten the result to match the original rewards tensor's shape.
            advantages = grouped_advantages.flatten() # Shape: (total_batch_size,)

            # For logging purposes, you might want to reconstruct the metrics.
            # We can create them by repeating the subgroup means/stds.
            mean_grouped_rewards_flat = torch.cat([
                mean1.repeat(1, split_point_gen),
                mean2.repeat(1, self.num_generations - split_point_gen)
            ], dim=1).flatten()
            std_grouped_rewards_flat = torch.cat([
                std1.repeat(1, split_point_gen),
                std2.repeat(1, self.num_generations - split_point_gen)
            ], dim=1).flatten()
            # It's better to log subgroup stats separately, but if you need to match the old variable names:
            mean_grouped_rewards = mean_grouped_rewards_flat
            std_grouped_rewards = std_grouped_rewards_flat
            is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        import pandas as pd
        if self.accelerator.is_main_process:
            df = pd.DataFrame(rewards_per_func.cpu().numpy(), columns=self.reward_func_names)
            df['advantage'] = all_process_advantages.cpu().numpy()
            print(df.head(10))  # 只打印前10行

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)

        # self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        # self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        # self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())
        
        is_std_zero1 = torch.isclose(std1, torch.zeros_like(std1))
        is_std_zero2 = torch.isclose(std2, torch.zeros_like(std2))
        self._metrics[mode]["reward1"].append(mean1.mean().item())
        self._metrics[mode]["reward1_std"].append(std1.mean().item())
        self._metrics[mode]["frac_reward1_zero_std"].append(is_std_zero1.float().mean().item())
        self._metrics[mode]["reward2"].append(mean2.mean().item())
        self._metrics[mode]["reward2_std"].append(std2.mean().item())
        self._metrics[mode]["frac_reward2_zero_std"].append(is_std_zero2.float().mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text)) # 注释以减少庞大的输出
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
        }
    
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

import numpy as np
from typing import Optional
def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: Optional[int] = None,
) -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`list[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.
        pad_to_multiple_of (`int`, *optional*, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Apply pad_to_multiple_of to the first (sequence) dimension
    if pad_to_multiple_of is not None:
        remainder = output_shape[0] % pad_to_multiple_of
        if remainder != 0:
            output_shape[0] += pad_to_multiple_of - remainder

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        # Define the slices
        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output

import pprint
class FedGRPOTrainer_RewardModel(GRPOTrainer):   
    """
    A variant of GRPOTrainer that supports federated learning.
    Inherits from GRPOTrainer and overrides the reward calculation logic.
    """
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.system_prompt = args.system_prompt if hasattr(args, 'system_prompt') else "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
    #     self.reward_url_dict = {
    #         'Algebra': 'http://localhost:8001/v1/chat/completions',
    #         'Intermediate Algebra': 'http://localhost:8002/v1/chat/completions',
    #         'Prealgebra': 'http://localhost:8002/v1/chat/completions',
    #         'Number Theory': 'http://localhost:8003/v1/chat/completions',
    #         'Geometry': 'http://localhost:8003/v1/chat/completions',
    #         'Precalculus': 'http://localhost:8004/v1/chat/completions',
    #         'Counting & Probability': 'http://localhost:8004/v1/chat/completions'
    #     }
    #     self.reward_model_name = ''


    def get_reward_score(self, prompts: list[str], completions: list[str], peroblem_type: list[str]) -> torch.Tensor:
        """
        Get the reward score for a batch of prompts and completions using the reward model.
        
        Args:
            prompts (list[str]): List of prompt strings.
            completions (list[str]): List of completion strings.
        
        Returns:
            torch.Tensor: Reward scores for the batch.
        """

        reward_prompts = [per_prompt[1]['content'] for per_prompt in prompts]

        vllm_urls = [self.reward_url_dict[math_type] for math_type in peroblem_type]
        # Get reward scores using vLLM
        reward_texts = asyncio.run(
            vllm_evaluate(vllm_urls, reward_prompts)
        )
        # 从reward model的回答中提取答案
        # 将该答案与server model的答案对比，打分
        reward_answer = extract_answer(reward_texts)
        completion_texts = [per_completion[0]['content'] for per_completion in completions]
        server_answer = extract_answer(completion_texts)
        reward_scores = compare_answer(reward_answer, server_answer)

        # Handle None values
        reward_scores = [reward_score if reward_score is not None else torch.nan for reward_score in reward_scores]

        return reward_scores

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # # print information for degugging ...
        # # if len(inputs) <5:
        # print('='*20, 'debugging', '='*20)
        # print('len(inputs):',len(inputs))

        # print('inputs[0]:')
        # pprint.pprint(inputs[0])
        # print('inputs[1]:')
        # pprint.pprint(inputs[1])
        # print('>>'*20)
        # # print('inputs:',inputs)
        # # print('inputs[0]:',inputs[0])
        # print('inputs[0]["prompt"]:')
        # pprint.pprint(inputs[0]["prompt"])
        # # inputs = inputs.to(device)



        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None
                sampling_params = SamplingParams(
                    n=1,  # vLLM on each GPU generates only 1 in colocate mode
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                )

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text = prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ):
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # print('=='*20)
        # print('completions:')
        # pprint.pprint(completions)
        # os._exit()

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)


        # # Split rewards into two equal parts
        # split_point = int(len(rewards) // 2)
        # rewards_first_half = rewards[:split_point]
        # rewards_second_half = rewards[split_point:]

        # # Compute grouped-wise rewards for each half
        # mean_grouped_rewards_first = rewards_first_half.view(-1, split_point).mean(dim=1)
        # std_grouped_rewards_first = rewards_first_half.view(-1, split_point).std(dim=1)
        # is_std_zero_first = torch.isclose(std_grouped_rewards_first, torch.zeros_like(std_grouped_rewards_first))
        # mean_grouped_rewards_second = rewards_second_half.view(-1, split_point).mean(dim=1)
        # std_grouped_rewards_second = rewards_second_half.view(-1, split_point).std(dim=1)
        # is_std_zero_second = torch.isclose(std_grouped_rewards_second, torch.zeros_like(std_grouped_rewards_second))

        # # Repeat for each generation in respective halves
        # mean_grouped_rewards_first = mean_grouped_rewards_first.repeat_interleave(split_point, dim=0)
        # std_grouped_rewards_first = std_grouped_rewards_first.repeat_interleave(split_point, dim=0)
        # mean_grouped_rewards_second = mean_grouped_rewards_second.repeat_interleave(split_point, dim=0)
        # std_grouped_rewards_second = std_grouped_rewards_second.repeat_interleave(split_point, dim=0)

        # # Calculate advantages for each half
        # advantages_first = rewards_first_half - mean_grouped_rewards_first
        # advantages_second = rewards_second_half - mean_grouped_rewards_second

        # # Scale advantages if required
        # if self.scale_rewards:
        #     advantages_first = advantages_first / (std_grouped_rewards_first + 1e-4)
        #     advantages_second = advantages_second / (std_grouped_rewards_second + 1e-4)

        # # Combine the two halves of advantages
        # advantages = torch.cat([advantages_first, advantages_second], dim=0)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        import pandas as pd
        if self.accelerator.is_main_process:
            df = pd.DataFrame(rewards_per_func.cpu().numpy(), columns=self.reward_func_names)
            df['advantage'] = all_process_advantages.cpu().numpy()
            print(df.head(10))  # 只打印前10行

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)

        # self._metrics[mode]["reward1"].append(mean_grouped_rewards_first.mean().item())
        # self._metrics[mode]["reward1_std"].append(std_grouped_rewards_first.mean().item())
        # self._metrics[mode]["frac_reward1_zero_std"].append(is_std_zero_first.float().mean().item())
        # self._metrics[mode]["reward2"].append(mean_grouped_rewards_second.mean().item())
        # self._metrics[mode]["reward2_std"].append(std_grouped_rewards_second.mean().item())
        # self._metrics[mode]["frac_reward2_zero_std"].append(is_std_zero_second.float().mean().item())

        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text)) # 注释以减少庞大的输出
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
        }