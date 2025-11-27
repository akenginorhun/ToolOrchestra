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

ip = 'localhost'

solution_str = """
```python
def find_nice_string(n, k, s):
    s = list(s)  # Convert string to a list for mutability
    for i in range(n):
        max_change = max(ord(s[i]) - ord('a'), ord('z') - ord(s[i]))
        change = min(max_change, k)
        
        if ord(s[i]) - ord('a') >= ord('z') - ord(s[i]):
            s[i] = chr(ord(s[i]) - change)
        else:
            s[i] = chr(ord(s[i]) + change)
        
        k -= change
        if k == 0:
            break

    if k > 0:
        print("-1")
    else:
        print("".join(s))

# Reading input
n, k = map(int, input().split())
s = input().strip()

find_nice_string(n, k, s)
```
"""
ground_truth = {"inputs": ["4 26\nbear", "4 26\nbear"], "outputs": ["zgar", "zbar"]}
data_source = 'codeforces'

res = requests.post(f"http://{ip}:8388/compute_score", json={"solution_str": solution_str, "ground_truth": ground_truth, "data_source": data_source})
score = res.json()
res = score

print(score)


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

# This should return 1.0, runtime error
res = requests.post(f"http://{ip}:8388/compute_score", json={"solution_str": solution_str, "ground_truth": ground_truth, "data_source": data_source})
score = res.json()
res = score

print(score)

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
# This should return 1.0, runtime error
res = requests.post(f"http://{ip}:8388/compute_score", json={"solution_str": solution_str, "ground_truth": ground_truth, "data_source": data_source})
score = res.json()
res = score

print(score)