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
import torch

def boost_high_score_advantages(advantages, scores, correct_sample_advantage_boost_value, correct_sample_advantage_boost_threshold):
    """Boost advantages for high-scoring samples
    
    Args:
        advantages: Original advantage values of shape [batch_size, response_length]
        scores: Corresponding scores for each sample of shape [batch_size]
        correct_sample_advantage_boost_value: Value to boost advantages for high-scoring samples
        
    Returns:
        Modified advantage values with boosted values for high-scoring samples
    """
    # Use torch.isclose for floating point comparison with a small tolerance
    high_score_mask = scores >= correct_sample_advantage_boost_threshold - 1e-5
    high_score_mask = high_score_mask.unsqueeze(-1).expand_as(advantages)
    
    # Boost advantages for high-scoring samples
    advantages = advantages + high_score_mask * correct_sample_advantage_boost_value
    return advantages