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

import uvicorn
from fastapi import FastAPI, HTTPException

from tau2.config import API_PORT
from tau2.data_model.simulation import Results, RunConfig
from tau2.registry import RegistryInfo
from tau2.run import get_options, load_tasks, run_domain

from .data_model import GetTasksRequest, GetTasksResponse

app = FastAPI()


@app.get("/health")
def get_health() -> dict[str, str]:
    return {"app_health": "OK"}


@app.post("/api/v1/get_options")
async def get_options_api() -> RegistryInfo:
    """ """
    try:
        return get_options()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/get_tasks")
async def get_tasks_api(
    request: GetTasksRequest,
) -> GetTasksResponse:
    """ """
    try:
        tasks = load_tasks(request.domain)
        return GetTasksResponse(tasks=tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/run_domain")
async def run_domain_api(
    request: RunConfig,
) -> Results:
    """ """

    try:
        results = run_domain(request)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=API_PORT)
