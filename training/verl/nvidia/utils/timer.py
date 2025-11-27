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
import sys
import time


class TimeoutChecker:
    def __init__(self, initial_interval_hours=4, backoff_minutes=15):
        super().__init__()
        self.last_save_time = initial_interval_hours * 3600 - backoff_minutes * 60
        self.start_time = time.time()
        self.last_saved = False

    def check_save(self):
        # Flush
        sys.stdout.flush()
        sys.stderr.flush()

        # Already saved after timeout
        if self.last_saved:
            return False

        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time >= self.last_save_time:
            self.last_saved = True
            return True

        return False
