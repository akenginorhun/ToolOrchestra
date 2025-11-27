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

from typing import Any, Dict, TypeVar

from addict import Dict as AddictDict
from pydantic import BaseModel, ConfigDict

from .utils import get_dict_hash

T = TypeVar("T", bound=BaseModel)


class BaseModelNoExtra(BaseModel):
    model_config = ConfigDict(extra="forbid")


def get_pydantic_hash(obj: BaseModel) -> str:
    """
    Generate a unique hash for the object based on its key fields.
    Returns a hex string representation of the hash.
    """
    hash_dict = obj.model_dump()
    return get_dict_hash(hash_dict)


def update_pydantic_model_with_dict(
    model_instance: T, update_data: Dict[str, Any]
) -> T:
    """
    Return an updated BaseModel instance based on the update_data.
    """
    raw_data = AddictDict(model_instance.model_dump())
    raw_data.update(AddictDict(update_data))
    new_data = raw_data.to_dict()
    model_class = type(model_instance)
    return model_class.model_validate(new_data)
