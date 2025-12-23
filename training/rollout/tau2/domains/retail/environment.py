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

# Copyright Sierra
import json
import os
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.utils import (
    RETAIL_DB_PATH,
    RETAIL_POLICY_PATH,
    RETAIL_TASK_SET_PATH,
)
from tau2.environment.environment import Environment


def get_environment(
    db: Optional[RetailDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Retail domain does not support solo mode")
    data_dir = os.path.join(os.environ.get('REPO_PATH'), 'evaluation/data_dir/tau2/domains/retail')
    if db is None:
        db = RetailDB.load(os.path.join(data_dir, 'db.json'))
    tools = RetailTools(db)
    with open(os.path.join(data_dir, 'policy.md'), "r") as fp:
        policy = fp.read()
    return Environment(
        domain_name="retail",
        policy=policy,
        tools=tools,
    )


# def get_tasks() -> list[Task]:
#     with open(RETAIL_TASK_SET_PATH, "r") as fp:
#         tasks = json.load(fp)
#     return [Task.model_validate(task) for task in tasks]

def get_tasks(task_path,save_to) -> list[Task]:
    print(f"Load tasks from {task_path}")
    with open(task_path, "r") as fp:
        tasks = json.load(fp)
    tasks_dict = {}
    for t in tasks:
        tasks_dict[t['id']] = t
    save_dir = str(save_to)
    assert save_dir.endswith('.json'),f"{save_dir}"
    save_dir = save_dir[:-len('.json')]
    processed_task_ids = set()
    if os.path.isdir(save_dir):
        for subfile in os.listdir(save_dir):
            if subfile.endswith('.json'):
                try:
                    with open(os.path.join(save_dir,subfile)) as f:
                        o = json.load(f)
                    processed_task_ids.add(o['task_id'])
                except:
                    continue
    updated_tasks = []
    for k,v in tasks_dict.items():
        if not k in processed_task_ids:
            updated_tasks.append(v)
    print("Total tasks:",len(tasks))
    print("Tasks to run:",len(updated_tasks))
    tasks = updated_tasks
    
    # Debug tasks validation
    print(f"Starting to validate {len(tasks)} tasks...")
    return_tasks = []
    for i, task in enumerate(tasks):
        task_id = task.get('id', f'unknown_{i}')
        print(f"[{i+1}/{len(tasks)}] Validating: {task_id}")
        try:
            validated = Task.model_validate(task)
            return_tasks.append(validated)
        except Exception as e:
            print(f"ERROR at task {task_id}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    if return_tasks:
        print('return_tasks[0]:', return_tasks[0])
    # exit(0)
    return return_tasks
