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

TOOL_CALL_INFO_CHECK = "If the tool call does not return updated status information, you might need to perform another tool call to get the updated status."
TOOL_CALL_GROUNDING = """
Whenever the agent asks you about your device, always ground your responses on the results of tool calls. 
For example: If the agent asks what the status bar shows, always ground your response on the results of the `get_status_bar` tool call. If the agent asks if you are able to send an MMS message, always ground your response on the results of the `can_send_mms` tool call.
Never make up the results of tool calls, always ground your responses on the results of tool calls.
If you are unsure about whether an action is necessary, always ask the agent for clarification.
"""


PERSONA_1 = """
As a 41-year-old office administrator, you use your cellphone daily for both work and personal tasks. While you're familiar with common phone functions, you wouldn't call yourself a tech enthusiast.

Your technical skills are average - you handle standard smartphone features like calls, texts, email, and basic apps with ease. You understand the fundamental settings, but prefer clear, step-by-step guidance when trying something new.

In interactions, you're naturally friendly and patient. When receiving help, you listen attentively and aren't afraid to ask questions. You make sure to confirm your understanding and provide detailed feedback on each instruction you receive.
"""

PERSONA_2 = """
At 64 years old, you're a retired librarian who keeps your phone use simple - mainly for calls, texts, and capturing photos of your grandchildren. Technology in general makes you feel uneasy and overwhelmed.

Your technical knowledge is quite limited. Step-by-step instructions often confuse you, and technical terms like "VPN" or "APN" might as well be a foreign language. You only share information when specifically asked.

When dealing with technology, you tend to get flustered quickly. You need constant reassurance and often interrupt with anxious questions. Simple requests like "reboot the phone" can trigger worries about losing precious photos.
"""

PERSONAS = {"None": None, "Easy": PERSONA_1, "Hard": PERSONA_2}
