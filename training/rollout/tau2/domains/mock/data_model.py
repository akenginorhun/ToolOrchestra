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

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from tau2.domains.mock.utils import MOCK_DB_PATH
from tau2.environment.db import DB

TaskStatus = Literal["pending", "completed"]


class Task(BaseModel):
    task_id: str = Field(description="Unique identifier for the task")
    title: str = Field(description="Title of the task")
    description: Optional[str] = Field(None, description="Description of the task")
    status: TaskStatus = Field(description="Status of the task")


class User(BaseModel):
    user_id: str = Field(description="Unique identifier for the user")
    name: str = Field(description="User's name")
    tasks: List[str] = Field(description="List of task IDs assigned to the user")


class MockDB(DB):
    """Simple database with users and their tasks."""

    tasks: Dict[str, Task] = Field(
        description="Dictionary of all tasks indexed by task ID"
    )
    users: Dict[str, User] = Field(
        description="Dictionary of all users indexed by user ID"
    )


def get_db():
    return MockDB.load(MOCK_DB_PATH)
