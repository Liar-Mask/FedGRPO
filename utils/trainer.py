import os
import warnings
import torch
import random
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel
from accelerate.utils import gather_object, broadcast_object_list, gather
from trl import GRPOTrainer, GRPOConfig
from trl import maybe_apply_chat_template, is_conversational, apply_chat_template
from utils.profiling import profiling_context
from utils.import_utils import unwrap_model_for_generation
from contextlib import nullcontext
from typing import Union, Any, Optional
from transformers import Trainer

class FedGRPOTrainer(GRPOTrainer):   
    """
    A variant of GRPOTrainer that supports federated learning.
    Inherits from GRPOTrainer and overrides the reward calculation logic.
    """

    def __init__(
        self,
        model: Optional[Union[str, "PreTrainedModel"]] = None,
        args: GRPOConfig = None,
        num_clients: int = 10,
        num_group: int = 2,
        reward_funcs_name: Optional[list[str]] = None,
        **kwargs,
    ):
        """
        Initializes the FedGRPOTrainer.

        Args:
            (Inherited args from GRPOTrainer)
            num_clients (`int`): The total number of clients in the federated environment.
            num_group (`int`): The number of clients sampled for each prompt.
        """
        if not isinstance(args, GRPOConfig):
            raise ValueError("`args` must be a `GRPOConfig` instance.")

        # --- Set up generation parameters BEFORE super().__init__() ---
        self.num_clients = num_clients
        self.num_group = num_group
        self.reward_funcs_name = reward_funcs_name or ["accuracy", "format"]
        
        # `base_num_generations` is the user-configured value per group
        self.base_num_generations = args.num_generations
        
        # `total_generations` is what the base class needs to generate all completions at once.
        # We temporarily modify the config for the super().__init__ call.
        total_generations = self.base_num_generations * self.num_group
        args.num_generations = total_generations

        # Now, call the parent constructor. It will use the modified `args` to correctly
        # set up the generation config for `total_generations`.
        super().__init__(model=model, args=args, **kwargs)
        # check self.num_generations 
        print(f"Inside FedGRPOTrainer __init__, self.num_generations: {self.num_generations}")


        # # Restore `self.num_generations` to its original meaning (per group) for our logic.
        # self.num_generations = self.base_num_generations
        # # Also restore the original args object
        # self.args.num_generations = self.base_num_generations

        # Initialize client-specific reward functions
        self._init_client_reward_funcs()
        
        print(f"FedGRPOTrainer initialized with {self.num_clients} clients, {self.num_group} groups per sample.")
        print(f"Generations per group: {self.base_num_generations}, Total generations per prompt: {total_generations}")


    def _init_client_reward_funcs(self):
        from rewards import accuracy_reward, model_reward, format_reward, boxed_format_reward, tag_count_reward
        REWARD_FUNCS_REGISTRY = {
            "accuracy": accuracy_reward,
            "format": format_reward,
            "tag_count": tag_count_reward,
            "model_reward": model_reward,
            "boxed_format": boxed_format_reward, 
        }
        reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in self.reward_funcs_name]
        self.reward_funcs_of_clients = {}
        
        for client_id in range(self.num_clients):

            training_mode = 'gt'  # 默认使用ground-truth accuracy
            if training_mode == 'gt':
                # self.reward_funcs_of_clients[client_id] = {
                #     'funcs': [accuracy_reward, format_reward, tag_count_reward],
                #     'names': ['accuracy', 'format', 'tag_count']
                # }
                self.reward_funcs_of_clients[client_id] = {
                    'funcs': reward_funcs,
                    'names': self.reward_funcs_name
                }
            else:
                self.reward_funcs_of_clients[client_id] = {
                    'funcs': [model_reward, format_reward],
                    'names': ['model_reward', 'format']
                }
        
        print(f"Initialized reward functions for {self.num_clients} clients")
        for cid in range(min(3, self.num_clients)):  # 打印前3个客户端的配置作为示例
            print(f"  Client {cid}: {self.reward_funcs_of_clients[cid]['names']}")

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        
        # 提取prompts和client_ids
        prompts = [x["prompt"] for x in inputs]
        client_ids_per_sample = [x.get("client_ids", list(range(self.num_group))) for x in inputs]
        
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # ============ 生成部分 ============
        if self.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    # 每个prompt生成 base_num_generations * num_group 个outputs
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
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            elif self.vllm_mode == "colocate":
                from vllm.sampling_params import GuidedDecodingParams, SamplingParams
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None
                sampling_params = SamplingParams(
                    n=1,
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                )

                if self.vllm_tensor_parallel_size > 1:
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
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # 常规生成路径
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

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # ============ Mask处理 ============
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        completion_lengths = completion_mask.sum(1)

        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

        # ============ 解码completions ============
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # ============ Reward计算 ============
        # 确定最大的reward函数数量
        max_reward_funcs = max(len(self.reward_funcs_of_clients[cid]['funcs']) for cid in range(self.num_clients))
        
        # 初始化reward张量：[total_completions, max_reward_funcs]
        # total_completions = num_samples * base_num_generations * num_group
        rewards_per_func = torch.full(
            (len(prompts), max_reward_funcs), 
            float('nan'), 
            device=device
        )

        # 准备reward计算的kwargs
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids", "client_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # 按样本和组进行处理
        num_samples = len(inputs) // self.num_generations
        
        # 记录每个group的reward（用于logging）
        group_rewards_log = {f"group_{i}": [] for i in range(self.num_group)}
        
        for sample_idx in range(num_samples):
            # 获取该样本的client_ids
            sample_client_ids = client_ids_per_sample[sample_idx]
            
            # 对每个group进行处理
            for group_idx in range(self.num_group):
                # 计算该group在总completions中的索引范围
                group_start_idx = sample_idx * self.num_generations + \
                                 group_idx * self.base_num_generations
                group_end_idx = group_start_idx + self.base_num_generations
                
                # 获取该group对应的client_id
                client_id = sample_client_ids[group_idx]
                client_reward_config = self.reward_funcs_of_clients[client_id]
                
                # 获取该组的prompts和completions
                group_prompts = prompts[group_start_idx:group_end_idx]
                group_completions = completions[group_start_idx:group_end_idx]
                group_completion_ids = completion_ids_list[group_start_idx:group_end_idx]
                group_reward_kwargs = {k: v[group_start_idx:group_end_idx] for k, v in reward_kwargs.items()}
                
                # 使用该client的reward函数计算rewards
                for func_idx, (reward_func, reward_func_name) in enumerate(
                    zip(client_reward_config['funcs'], client_reward_config['names'])
                ):
                    with profiling_context(self, f"{reward_func_name}_client{client_id}"):
                        if isinstance(reward_func, nn.Module):
                            # 模型类型的reward（如reward model）
                            if is_conversational(inputs[0]):
                                messages = [{"messages": p + c} for p, c in zip(group_prompts, group_completions)]
                                texts = [apply_chat_template(x, self.processing_class)["text"] for x in messages]
                            else:
                                texts = [p + c for p, c in zip(group_prompts, group_completions)]
                            reward_inputs = self.processing_class(
                                text=texts, return_tensors="pt", padding=True, 
                                padding_side="right", add_special_tokens=False
                            )
                            reward_inputs = super()._prepare_inputs(reward_inputs)
                            with torch.inference_mode():
                                group_rewards = reward_func(**reward_inputs).logits[:, 0]
                        else:
                            # 函数类型的reward
                            output_reward_func = reward_func(
                                prompts=group_prompts,
                                completions=group_completions,
                                completion_ids=group_completion_ids,
                                **group_reward_kwargs
                            )
                            output_reward_func = [r if r is not None else torch.nan for r in output_reward_func]
                            group_rewards = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                        
                        # 将该group的rewards填入总rewards张量
                        rewards_per_func[group_start_idx:group_end_idx, func_idx] = group_rewards
                
                # 记录该group的平均reward用于logging
                group_mean_reward = rewards_per_func[group_start_idx:group_end_idx].nanmean().item()
                group_rewards_log[f"group_{group_idx}"].append(group_mean_reward)

        # ============ Reward聚合和优势计算 ============
        # 检查是否有全为NaN的行
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            warnings.warn(
                f"All reward functions returned None for completion at index {nan_row_idx}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather rewards（在分布式环境中）
        rewards_per_func = gather(rewards_per_func)

        # 对rewards加权求和（使用reward_weights）
        if hasattr(self, 'reward_weights'):
            # 只使用前面几个权重（对应reward函数数量）
            effective_weights = self.reward_weights[:max_reward_funcs].to(device).unsqueeze(0)
            rewards = (rewards_per_func * effective_weights).nansum(dim=1)
        else:
            # 如果没有权重，直接求平均
            rewards = rewards_per_func.nanmean(dim=1)

        # 计算分组的rewards统计
        print(f"Rewards before reshaped: {rewards.shape}")
        rewards_reshaped = rewards.view(-1, self.num_generations)
        print(f"Rewards reshaped for grouping: {rewards_reshaped.shape}")
        mean_grouped_rewards = rewards_reshaped.mean(dim=1)
        std_grouped_rewards = rewards_reshaped.std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # 归一化rewards计算优势
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # 切片获取本地数据
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        import pandas as pd
        if self.accelerator.is_main_process:
            print("\n")
            df = pd.DataFrame(rewards_per_func.cpu().numpy(), columns=self.reward_func_names)
            df['advantage'] = all_process_advantages.cpu().numpy()
            # print(df.head(self.num_generations))  # 只打印前num_generations行
            print(df.head(16))  # 只打印前16行

        # ============ Logging ============
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Log terminated completions
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Log per-function rewards
        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        
        # Log overall reward
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # ============ 记录每个group的reward情况 ============
        for group_idx in range(self.num_group):
            group_key = f"group_{group_idx}"
            if group_rewards_log[group_key]:
                mean_group_reward = sum(group_rewards_log[group_key]) / len(group_rewards_log[group_key])
                self._metrics[mode][f"rewards/{group_key}/mean"].append(mean_group_reward)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
        }

def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """计算忽略NaN的标准差"""
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)
    count = torch.sum(~torch.isnan(tensor))
    if count > 1:
        variance *= count / (count - 1)
    return torch.sqrt(variance)


def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: Optional[int] = None,
) -> torch.Tensor:
    """Pad tensors to same shape"""
    import numpy as np
    from typing import Optional
    
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    if pad_to_multiple_of is not None:
        remainder = output_shape[0] % pad_to_multiple_of
        if remainder != 0:
            output_shape[0] += pad_to_multiple_of - remainder

    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output
