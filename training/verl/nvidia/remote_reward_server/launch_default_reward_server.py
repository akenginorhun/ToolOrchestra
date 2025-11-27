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

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from verl.nvidia.reward_score import _default_compute_score as compute_score_fn

app = FastAPI()

# Global ProcessPoolExecutor
# Adjust based on your server load
# Recommended 64 since prime reward manager make 64 requests in parallel
num_workers = 64
executor = ProcessPoolExecutor(num_workers)  

timeout = 60

# Request model
class ComputeScoreRequest(BaseModel):
    solution_str: str
    ground_truth: Any
    data_source: str

def single_compute_score(evaluation_func, completion, reference, task):
    """ Synchronous function executed inside ProcessPoolExecutor """
    try:
        return evaluation_func(task, completion, reference)
    except Exception as e:
        print(f"Error processing completion: {completion[:10]}, data source: {task}, Error: {e}")
        return 0.0  # Default score for failed rows

@app.post("/compute_score")
async def compute_score(request: ComputeScoreRequest):
    """ Asynchronous API endpoint using ProcessPoolExecutor with timeout """
    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(
                executor, partial(single_compute_score, compute_score_fn, request.solution_str, request.ground_truth, request.data_source)
            )
    try:
        score = await asyncio.wait_for(task, timeout=timeout)
        return {"score": float(score)}
    except asyncio.TimeoutError as e:
        try:
            task.cancel()
        except asyncio.CancelledError:
            pass  # Task was cancelled, we can safely ignore this error
        print(f"Request timed out for completion: {request.solution_str[:10]}, data source: {request.data_source}")
        raise HTTPException(status_code=408, detail="Computation timed out")
    except Exception as e:
        print(f"An unexpected error occurred for completion: {request.solution_str[:10]}, data source: {request.data_source}, Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8388)
