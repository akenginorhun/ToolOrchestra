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
# Original code from VERL:
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
Generate responses given a dataset of prompts
"""
import csv
import numpy as np
import hydra
import os
from tabulate import tabulate

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import pandas as pd
from verl.utils.fs import copy_local_path_from_hdfs
from verl.nvidia.eval.gen_utils import get_generation_results


@hydra.main(config_path='.', config_name='deepscaler_eval', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    output_dir = os.path.dirname(config.data.output_path)
    # dataset name without extension
    name_without_ext = os.path.splitext(os.path.basename(config.data.train_files))[0]
    csv_path = os.path.join(output_dir, f'{name_without_ext}_summary.csv')
 
    # skip the job if the csv file already exists
    if os.path.exists(csv_path):
        print(f"CSV file {csv_path} already exists. Skipping the job.")
        return

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    metric_dict = get_generation_results(config, tokenizer)
   
    dataset_name = os.path.basename(config.data.train_files)
    metric_dict['model_path'] = config.actor_rollout_ref.model.path
    metric_dict['dataset'] = dataset_name
    metric_dict['temperature'] = config.actor_rollout_ref.rollout.val_kwargs.temperature
    metric_dict['n_samples'] = config.actor_rollout_ref.rollout.val_kwargs.n
    metric_dict['response_length'] = config.data.max_response_length

   # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # Write to CSV
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metric_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metric_dict)

    # Convert the row data into a list of lists format for tabulate
    table_data = [[k, v] for k, v in metric_dict.items()]
    # Print table
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))
    # write tabule into txt file
    with open(os.path.join(output_dir, f'{name_without_ext}_summary.txt'), 'w') as f:
        f.write(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))


if __name__ == '__main__':
    main()
