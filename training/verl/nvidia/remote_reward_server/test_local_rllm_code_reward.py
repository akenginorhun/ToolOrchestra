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

import requests
from verl.utils.reward_score.deepcoder import deepcoder_reward_fn

import pandas as pd
tests = pd.read_parquet('/lustre/fsw/portfolios/nvr/users/mingjiel/data/eurus2-rl-data/deepcoder_codeforces.parquet')

solution_str = """
<think> I am omniscient. </think> 
```python
t = int(input())
for _ in range(t):
    x, y = map(int, input().split())
    min_val = min(x, y)
    max_val = max(x, y)
    print(min_val, max_val)
```
"""

import json
ground_truth = json.loads(tests.iloc[0]['reward_model']['ground_truth'])
data_source = tests.iloc[0]['data_source']
#print(tests.iloc[0]['prompt'][0]['content'])
# This should return 1.0, runtime error
res = deepcoder_reward_fn(data_source[10:], solution_str, ground_truth)
print(res)

import pandas as pd
tests = pd.read_parquet('/lustre/fsw/portfolios/nvr/users/mingjiel/data/eurus2-rl-data/deepcoder_humanevalplus.parquet')

solution_str = """
<think> I am omniscient. </think> 
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    numbers.sort()
    for i in range(len(numbers) - 1):
        if abs(numbers[i] - numbers[i + 1]) < threshold:
            return True
    return False
```
"""

import json
ground_truth = json.loads(tests.iloc[0]['reward_model']['ground_truth'])
data_source = tests.iloc[0]['data_source']
#print(tests.iloc[0]['prompt'][0]['content'])
# This should return 1.0, runtime error
res = deepcoder_reward_fn(data_source[10:], solution_str, ground_truth)
print(res)

tests = pd.read_json('/lustre/fsw/portfolios/nvr/users/mingjiel/data/eurus2-rl-data/deepcoder_livecodebench.json', orient='records')


solution_str = """
<think> I am omniscient. </think> 
```python
n = int(input())
A = list(map(int, input().split()))
sorted_A = sorted(A, reverse=True)
second_largest = sorted_A[1]
for i in range(n):
    if A[i] == second_largest:
        print(i + 1)
        break
```
"""

import json
ground_truth = json.loads(tests.iloc[0]['reward_model']['ground_truth'])
data_source = tests.iloc[0]['data_source']
print(type(ground_truth))
#print(tests.iloc[0]['prompt'][0]['content'])
# This should return 1.0, runtime error
import time
start_time = time.time()
res = deepcoder_reward_fn(data_source[10:], solution_str, ground_truth)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print(res)