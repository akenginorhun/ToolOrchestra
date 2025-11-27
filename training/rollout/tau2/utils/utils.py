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

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

from deepdiff import DeepDiff
from dotenv import load_dotenv
from loguru import logger

res = load_dotenv()

DATA_DIR = Path('../evaluation/data_dir/tau2/domains')


def get_dict_hash(obj: dict) -> str:
    """
    Generate a unique hash for dict.
    Returns a hex string representation of the hash.
    """
    hash_string = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(hash_string.encode()).hexdigest()


def show_dict_diff(dict1: dict, dict2: dict) -> str:
    """
    Show the difference between two dictionaries.
    """
    diff = DeepDiff(dict1, dict2)
    return diff


def get_now() -> str:
    """
    Returns the current date and time in the format YYYYMMDD_HHMMSS.
    """
    now = datetime.now()
    return format_time(now)


def format_time(time: datetime) -> str:
    """
    Format the time in the format YYYYMMDD_HHMMSS.
    """
    return time.isoformat()


def get_commit_hash() -> str:
    """
    Get the commit hash of the current directory.
    """
    raise ValueError('debug')
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
            .split("\n")[0]
        )
    except Exception as e:
        logger.error(f"Failed to get git hash: {e}")
        commit_hash = "unknown"
    return commit_hash
