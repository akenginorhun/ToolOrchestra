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
# Copyright 2024 PRIME team and/or its affiliates
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

from .utils import check_correctness as apps_check_correctness
import json
import re
import traceback


def compute_score(completion, test_cases, continuous=False):
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    solution = completion.split('```python')[-1].split('```')[0]
    try:
        try:
            if not isinstance(test_cases, dict):
                test_cases = json.loads(test_cases)
        except Exception as e:
            print(f"Error:{e}")

        # Complete check on all in-out pairs first. If there is no failure, per-sample test can be skipped.
        try:
            res, metadata = apps_check_correctness(in_outs=test_cases, generation=solution, timeout=5, debug=False)
            metadata = list(metadata)
            success = all(map(lambda x: x == True, res))
            if continuous:
                # compile error
                if res[-1] == -2:
                    return 0, metadata
                else:
                    # set runtime error to 0
                    res = [0 if x == -1 else x for x in res]
                    return sum(res) / len(res), metadata
            else:
                return success, metadata

        except Exception as e:
            success = False
            metadata_list = None

    except Exception as e:
        traceback.print_exc(10)
        success = False
        metadata_list = None
    return success, metadata_list
