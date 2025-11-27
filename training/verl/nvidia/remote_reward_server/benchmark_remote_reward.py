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
import time
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
ground_truth = {"inputs": ["4 26\nbear"], "outputs": ["zgar"]}
data_source = 'codeforces'

start = time.time()
res = requests.post(f"http://{ip}:8388/compute_score", json={"solution_str": solution_str, "ground_truth": ground_truth, "data_source": data_source})
score = res.json()
res = score

print(score)
print("Total time for single request: ", time.time() - start)

num_parallel_requests = 2

start = time.time()
for i in range(num_parallel_requests):
    res = requests.post(f"http://{ip}:8388/compute_score", json={"solution_str": solution_str, "ground_truth": ground_truth, "data_source": data_source})
    score = res.json()
    res = score

print(f"Total time for {num_parallel_requests} requests in sequence: ", time.time() - start)

num_parallel_requests = 256
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def request_wrapper(idx):
    res = requests.post(f"http://{ip}:8388/compute_score", json={"solution_str": solution_str, "ground_truth": ground_truth, "data_source": data_source})
    score = res.json()
    res = score
    return res

async def single_compute_score(idx, executor,timeout=300.):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(request_wrapper, idx)  # Ensure synchronous
                ),
                timeout=timeout)
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"Error processing completion: {completion[:10]}, Error: {e}")
        return None  # Default value for failed rows

start = time.time()
async def main():
    with ProcessPoolExecutor(max_workers=num_parallel_requests) as executor:
        # Create tasks for all rows
        tasks_async = [
            single_compute_score(idx, executor, timeout=300.)
            for idx in range(num_parallel_requests)
        ]
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except:
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print('shut down failed: ' + str(kill_err))
            raise
    return results

results = asyncio.run(main())
print(f"Total time for {num_parallel_requests} requests in async parallel: ", time.time() - start)
