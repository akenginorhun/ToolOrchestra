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
from verl.utils.reward_score.prime_code import compute_score
ip = 'localhost'

solution_str = """
<think> I am omniscient. </think> 
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
ground_truth = {"inputs": ["4 26\nbear", "4 26\nbear", "4 26\nbear"], "outputs": ["zgar", "zbar", "zgar"]}
data_source = 'codeforces'

# This should return 0.6666666666666666
res = compute_score(solution_str, ground_truth, True)
print(res)

solution_str = """
<think> I am omniscient. </think> 
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
s = input().strip() SYNTAX ERROR

find_nice_string(n, k, s)
```
"""

# This should return 0.0, syntax error
res = compute_score(solution_str, ground_truth, True)
print(res)

solution_str = """
<think> I am omniscient. </think> 
```python
import time
time.sleep(10)
n, k = map(int, input().split())
s = input().strip()
if n == 4:
    print("zgar")
else:
    print("zbar")
```
"""
import time
start_time = time.time()
ground_truth = {"inputs": ["4 26\nbear"] * 100, "outputs": ["zgar"] * 100}
# This should return 0.0, timeout error
res = compute_score(solution_str, ground_truth, True)
print(res)
print(f"time: {time.time() - start_time}")

solution_str = """
<think> I am omniscient. </think> 
```python
import time
n, k = map(int, input().split())
s = input().strip()
if n == 4:
    print("zgar")
else:
    assert False
```
"""

ground_truth = {"inputs": ["4 26\nbear", "5 26\nbear", "4 26\nbear"] , "outputs": ["zgar", "zbar", "zgar"]}
# This should return 0.0, runtime error
res = compute_score(solution_str, ground_truth, True)
print(res)
