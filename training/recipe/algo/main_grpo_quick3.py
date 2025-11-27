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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import hydra
import ray
import time
import torch
import os
from collections import defaultdict
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.trainer.ppo.reward import get_custom_reward_fn

from .grpo_ray_trainer_quick3 import RayGRPOTrainer

def compute_score_em(pred, ground_truth, response, use_format_score, method='strict'):
    format_score = 0
    score = 0
    if use_format_score:
        original_response = response
        if response.count('<think>')==1 and response.count('</think>')==1 and response.count('<answer>')==1 and response.count('</answer>')==1:
            response = response.split('<think>')[1]
            if response.count('</think>')==1:
                response = response.split('</think>')[1]
                if response.count('<answer>') == 1:
                    response = response.split('<answer>')[1]
                    if response.count('</answer>')==1:
                        choice = response.split('</answer>')[0].strip()
                        tmp_action = choice.lstrip('[').rstrip(']').strip()
                        if tmp_action in ['1','query writer']:
                            score = 1
                            format_score = 1
                        elif tmp_action in ['2','answer generator']:
                            format_score = 1
    if format_score==0:
        score = 0
    return score, format_score


class RewardManager():

    def __init__(self, tokenizer, num_examine, use_format_score) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  
        self.use_format_score = use_format_score

    def __call__(self, data, global_step):

        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            reward_tensor[i, 0] = data_item.non_tensor_batch['reward']

        return reward_tensor


@hydra.main(config_path="config", config_name="grpo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):

        # # 4 min 10 seconds
        
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local
        OmegaConf.resolve(config)
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        from verl.utils import hf_processor, hf_tokenizer
        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }
        if config.reward_model.enable:
            if config.reward_model.strategy == "fsdp":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id
        reward_manager_name = config.reward_manager.get("type", "naive")
        if reward_manager_name == 'match':
            reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, use_format_score=True)
        else:
            raise NotImplementedError
        strategy = NodeAffinitySchedulingStrategy(node_id = ray.get_runtime_context().get_node_id(), soft = False)
        val_reward_fn = reward_fn
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        trainer = RayGRPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
