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

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from loguru import logger

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# Define TypeVar for the agent state type
AgentState = TypeVar("AgentState")
ValidAgentInputMessage = UserMessage | ToolMessage | MultiToolMessage


def is_valid_agent_history_message(message: Message) -> bool:
    """Check if the message is a valid agent history message."""
    return (
        isinstance(message, AssistantMessage)
        or (isinstance(message, UserMessage) and not message.is_tool_call())
        or (isinstance(message, ToolMessage) and message.requestor == "assistant")
    )


class BaseAgent(ABC, Generic[AgentState]):
    """
    Base agent class that defines the common interface for all agents.
    """

    @abstractmethod
    def generate_next_message(
        self, message: ValidAgentInputMessage, state: AgentState
    ) -> tuple[AssistantMessage, AgentState]:
        """
        Generate the next message from a user/tool message(s) and an agent state.
        Args:
            message: The user message or tool message(s).
            state: The agent state.

        Returns:
            A tuple of an assistant message and an agent state.
        """
        raise NotImplementedError

    @abstractmethod
    def get_init_state(
        self,
        message_history: Optional[list[Message]] = None,
    ) -> AgentState:
        """
        Get the initial state of the agent.
        This is required to be able to rerun an agent from any point in the conversation.
        Args:
            message_history: The message history.

        Returns:
            The initial state of the agent.
        """
        raise NotImplementedError

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        """Check if the message is a stop message.
        By default the agent does not stop.
        """
        return False

    def set_seed(self, seed: int):
        """
        Set the seed for the agent. [Optional]
        """
        logger.warning(
            f"Setting seed for agent is not implemented for class {self.__class__.__name__}"
        )


class LocalAgent(BaseAgent[AgentState]):
    """
    Local agent implementation
    Agent developers should implement the following methods:
    - generate_next_message: Generate the next message: Can be a user message or a tool call.
    - get_init_state: Get the initial state of the agent. [Optional] This is required to be able to rerun an agent from any point in the conversation.

    """

    def __init__(self, tools: list[Tool], domain_policy: str):
        super().__init__()
        self.tools = tools
        self.domain_policy = domain_policy
