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
import re
from collections import Counter
import json

def validate_result(result, answer):
    if len(result) == 0 or len(answer) == 0:
        return 1 if len(result) == len(answer) else 0

    try:
        counter1_full = Counter(
            (item["name"], json.dumps(item["arguments"], sort_keys=True)) 
            for item in result
        )
        counter2_full = Counter(
            (item["name"], json.dumps(item["arguments"], sort_keys=True)) 
            for item in answer
        )
    except TypeError:
        return 0

    return 1 if counter1_full == counter2_full else 0

def extract_solution(tool_call_str):
    
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = list(re.finditer(pattern, tool_call_str, flags=re.DOTALL))
    if not matches:
        return None, tool_call_str
    last_content = matches[-1].group(1).strip()
    
    try:
        return json.loads(last_content),tool_call_str
    except json.JSONDecodeError:
        return None, tool_call_str

def validate_format(tool_call_list):
    for item in tool_call_list:
        if not isinstance(item, dict):
            return 0
    for item in tool_call_list:
        if "name" not in item.keys() or "arguments" not in item.keys():
            return 0
    return 1

def compute_score(solution_str, ground_truth):
    """
    Returns:
       1  if the parsed result from 'solution_str' fully matches the 'ground_truth'
       0  otherwise
    """
    # Parse the ground truth from a JSON string
    answer = json.loads(ground_truth)

    # Extract the tool call result and the full output string
    result, output_string = extract_solution(solution_str)

    # Ensure the "thinking" markers are present
    if "</think>" not in output_string:
        return 0

    # Ensure we actually extracted something
    if result is None:
        return 0

    # If the result is a single dictionary, wrap it in a list
    if isinstance(result, dict):
        result = [result]

    # Validate the format of the extracted result
    if not validate_format(result):
        return 0

    # Compare the extracted result with the ground truth
    if validate_result(result, answer) == 1:
        return 1
    else:
        return 0