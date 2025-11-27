# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -----------------------------------------------------------------------------

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
import time
import numpy as np
import ray
import os
import json
import torch
import tensordict
from tensordict import TensorDict
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
import copy
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from torchdata.stateful_dataloader import StatefulDataLoader
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage, compute_response_mask
from lead_agent.llm_agent.generation_quick3 import LLMGenerationManager, GenerationConfig

class DataToolProto(DataProto):

    @classmethod
    def from_dict(cls, tensors, non_tensors=None, meta_info=None, num_batch_dims=1, auto_padding=False):
        # assert len(tensors) > 0, 'tensors must not be empty'
        assert num_batch_dims > 0, 'num_batch_dims must be greater than zero'
        if non_tensors is not None:
            assert num_batch_dims == 1, 'only support num_batch_dims=1 when non_tensors is not None.'

        if meta_info is None:
            meta_info = {}
        if non_tensors is None:
            non_tensors = {}

        assert isinstance(non_tensors, dict)

        # get and check batch size
        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                current_batch = tensor.shape[:num_batch_dims]
                assert batch_size == current_batch, \
                    f'Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. Got {pivot_key} has {batch_size}, {key} has {current_batch}'

        for key, val in non_tensors.items():
            if batch_size is None:
                batch_size = len(val)
            else:
                if isinstance(batch_size,torch.Size):
                    assert batch_size[0] == len(val), f"batch_size: {batch_size}, len(val): {len(val)}"
                else:
                    assert batch_size == len(val), f"{key}, batch_size: {batch_size}, len(val): {len(val)}"
            non_tensors[key] = np.array(val, dtype=object)

        tensor_dict = TensorDict(source=tensors, batch_size=batch_size)
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    def repeat(self, repeat_times=2, interleave=True):
        if self.batch is not None:
            if interleave:
                # Interleave the data
                repeated_tensors = {
                    key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
                }
            else:
                # Stack the data
                repeated_tensors = {
                    key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:])
                    for key, tensor in self.batch.items()
                }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(self.batch.batch_size[0] * repeat_times,),
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if interleave:
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            else:
                repeated_non_tensor_batch[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

        if not 'repeat_id' in repeated_non_tensor_batch:
            repeated_non_tensor_batch['repeat_id'] = []
            index_count = defaultdict(int)
            for example_index in repeated_non_tensor_batch['index']:
                repeated_non_tensor_batch['repeat_id'].append(index_count[example_index])
                index_count[example_index] += 1
            repeated_non_tensor_batch['repeat_id'] = np.array(repeated_non_tensor_batch['repeat_id'], dtype=object)

        return DataToolProto(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

class JsonlDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 file_path,
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 prompt_template='v1',
                 tool_config_path=None,
                 vllm_model_config_path=None,
                 my_output_dir=None,
                 cur_transfer_dir=None,
                 model_type=None
        ):

        self.file_path = file_path
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.model_type = model_type

        self.dataset = []            
        if isinstance(self.file_path,str):
            with open(self.file_path) as f:
                for line in f:
                    self.dataset.append(json.loads(line))
        else:
            for one_file in self.file_path:
                with open(one_file) as f:
                    for line in f:
                        self.dataset.append(json.loads(line)) 

        with open(vllm_model_config_path) as vllm_model_configs_file:
            vllm_model_configs = json.load(vllm_model_configs_file)
        self.vllm_model_configs = vllm_model_configs
        self.my_output_dir = my_output_dir
        self.cur_transfer_dir = cur_transfer_dir

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataset[item]
        row_dict['turn_id'] = 0
        row_dict['id'] = row_dict['index']
        row_dict['vllm_model_configs'] = self.vllm_model_configs
        row_dict['my_output_dir'] = self.my_output_dir
        row_dict['cur_transfer_dir'] = self.cur_transfer_dir
        row_dict['model_type'] = self.model_type
        return row_dict

def collate_fn(data_list):
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output

class RayGRPOTrainer(RayPPOTrainer):
                
    def _create_dataloader(self,place_holder1,place_holder2,place_holder3,train_sampler):
        from torch.utils.data import DataLoader
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        self.train_dataset = JsonlDataset(
                                    file_path=self.config.data.train_files,
                                    tokenizer=self.tokenizer,
                                    prompt_key=self.config.data.prompt_key,
                                    max_prompt_length=self.config.data.max_prompt_length,
                                    filter_prompts=True,
                                    return_raw_chat=self.config.data.get('return_raw_chat', False),
                                    truncation='error',
                                    prompt_template=self.config.data.prompt_template,
                                    tool_config_path=self.config.data.train_tool_config_path,
                                    vllm_model_config_path=self.config.data.vllm_model_configs,
                                    my_output_dir=self.config.data.my_output_dir,
                                    cur_transfer_dir=self.config.data.cur_transfer_dir,
                                    model_type=self.config.data.model_type
                                )

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        self.val_dataset = JsonlDataset(
                                file_path=self.config.data.val_files,
                                tokenizer=self.tokenizer,
                                prompt_key=self.config.data.prompt_key,
                                max_prompt_length=self.config.data.max_prompt_length,
                                filter_prompts=True,
                                return_raw_chat=self.config.data.get('return_raw_chat', False),
                                truncation='error',
                                prompt_template=self.config.data.prompt_template,
                                tool_config_path=self.config.data.test_tool_config_path,
                                vllm_model_config_path=self.config.data.vllm_model_configs,
                                my_output_dir=self.config.data.my_output_dir,
                                cur_transfer_dir=self.config.data.cur_transfer_dir,
                                model_type=self.config.data.model_type
                            )

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )
        
        assert len(self.train_dataloader) >= 1, f"{len(self.train_dataloader)}"
        assert len(self.val_dataloader) >= 1, f"{len(self.val_dataloader)}"

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _create_loss_mask(self, batch, metrics):
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        batch.batch['loss_mask'] = response_mask
        return batch, metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            train_tool_config_path=self.config.data.train_tool_config_path,
            test_tool_config_path=self.config.data.test_tool_config_path,
        )

        # load checkpoint before doing anything
        self._load_checkpoint()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataToolProto = DataToolProto.from_single_dict(batch_dict)
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                with _timer('step', timing_raw):
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        try:
                            final_gen_batch_output,tool_avg = generation_manager.run_llm_loop(
                                gen_batch=batch,
                                tokenizer_config={
                                    'tokenizer': self.tokenizer,
                                    'max_prompt_length': self.config.data.max_prompt_length,
                                    'max_response_length': self.config.data.max_response_length,
                                    'truncation': 'error'
                                },
                                global_steps=self.global_steps,
                                topk_doc=self.config.data.topk_doc,
                                use_llm_reward=self.config.data.use_llm_reward,
                                efficiency_reward=self.config.data.efficiency_reward,
                                exp_tag=self.config.data.exp_tag,
                                use_qa_reward=self.config.data.use_qa_reward
                            )
                        except Exception as error_inference:
                            continue
                        if not final_gen_batch_output:
                            import shutil
                            shutil.copytree(os.path.join(self.config.data.my_output_dir,'ckpt',f"global_step_{self.global_steps-1}"),os.path.join(self.config.data.my_output_dir,'ckpt',f"global_step_{self.global_steps}"))
                            with open(os.path.join(self.config.data.my_output_dir,'ckpt',f"latest_checkpointed_iteration.txt"),'w') as f:
                                f.write(f"{self.global_steps}")
                            continue
                    for key in final_gen_batch_output.batch.keys():
                        final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()
                    batch = final_gen_batch_output
                    batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                    assert self.config.actor_rollout_ref.rollout.n==1
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    self._balance_batch(batch, metrics=metrics)
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    for key in batch.batch.keys():
                        if key != 'old_log_probs':
                            batch.batch[key] = batch.batch[key].long()
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        
                    if self.use_reference_policy:
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        reward_tensor = self.reward_fn(batch, global_step=self.global_steps)
                        batch.batch['token_level_scores'] = reward_tensor
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                    kl_ctrl=self.kl_ctrl,
                                                                    kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                        batch = compute_advantage(batch,
                                                    adv_estimator=self.config.algorithm.adv_estimator,
                                                    gamma=self.config.algorithm.gamma,
                                                    lam=self.config.algorithm.lam,
                                                    num_repeat=self.config.actor_rollout_ref.rollout.n)
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer('update_actor', timing_raw):
                            batch, metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                logger.log(data=metrics, step=self.global_steps)
                
                self.global_steps += 1







