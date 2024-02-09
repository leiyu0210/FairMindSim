# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import pytest

from camel.agents.player_agent import PlayerAgent
from camel.configs import ChatGPTConfig
from camel.generators import SystemMessageGenerator
from camel.messages import BaseMessage
from camel.typing import ModelType, RoleType, TaskType

parametrize = pytest.mark.parametrize(
    "model",
    [
        (ModelType.STUB,),
        pytest.param(ModelType.GPT_3_5_TURBO, marks=pytest.mark.model_backend),
        pytest.param(ModelType.GPT_4, marks=pytest.mark.model_backend),
    ],
)


@parametrize
def test_chat_agent(model: ModelType):
    model_config = ChatGPTConfig()
    system_msg = SystemMessageGenerator(task_type=TaskType.AI_SOCIETY).from_dict(
        dict(assistant_role="doctor"), role_tuple=("doctor", RoleType.ASSISTANT),
    )
    assistant = PlayerAgent(system_msg, model=model, model_config=model_config)

    assert str(assistant) == ("ChatAgent(doctor, " f"RoleType.ASSISTANT, {str(model)})")

    assistant.reset()
    user_msg = BaseMessage(
        role_name="Patient", role_type=RoleType.USER, meta_dict=dict(), content="Hello!"
    )
    assistant_response = assistant.step(user_msg)

    assert isinstance(assistant_response.msgs, list)
    assert len(assistant_response.msgs) > 0
    assert isinstance(assistant_response.terminated, bool)
    assert assistant_response.terminated is False
    assert isinstance(assistant_response.info, dict)
    assert assistant_response.info["id"] is not None

    assistant.reset()
    token_limit = assistant.model_token_limit
    user_msg = BaseMessage(
        role_name="Patient",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="token" * (token_limit + 1),
    )
    assistant_response = assistant.step(user_msg)

    assert isinstance(assistant_response.msgs, list)
    assert len(assistant_response.msgs) == 0
    assert isinstance(assistant_response.terminated, bool)
    assert assistant_response.terminated is True
    assert isinstance(assistant_response.info, dict)
    assert assistant_response.info["termination_reasons"][0] == "max_tokens_exceeded"


@pytest.mark.model_backend
@pytest.mark.parametrize("n", [1, 2, 3])
def test_chat_agent_multiple_return_messages(n):
    model_config = ChatGPTConfig(temperature=1.4, n=n)
    system_msg = BaseMessage(
        "Assistant",
        RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a helpful assistant.",
    )
    assistant = PlayerAgent(system_msg, model_config=model_config)
    assistant.reset()
    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Tell me a joke.",
    )
    assistant_response = assistant.step(user_msg)
    assert assistant_response.msgs is not None
    assert len(assistant_response.msgs) == n


@pytest.mark.model_backend
def test_chat_agent_stream_output():
    system_msg = BaseMessage(
        "Assistant",
        RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a helpful assistant.",
    )
    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Tell me a joke.",
    )

    stream_model_config = ChatGPTConfig(temperature=0, n=2, stream=True)
    stream_assistant = PlayerAgent(system_msg, model_config=stream_model_config)
    stream_assistant.reset()
    stream_assistant_response = stream_assistant.step(user_msg)

    for msg in stream_assistant_response.msgs:
        assert len(msg.content) > 0

    stream_usage = stream_assistant_response.info["usage"]
    assert stream_usage["completion_tokens"] > 0
    assert stream_usage["prompt_tokens"] > 0
    assert (
        stream_usage["total_tokens"]
        == stream_usage["completion_tokens"] + stream_usage["prompt_tokens"]
    )


@pytest.mark.model_backend
def test_set_output_language():
    system_message = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )
    agent = PlayerAgent(system_message=system_message, model=ModelType.GPT_3_5_TURBO)
    assert agent.output_language is None

    # Set the output language to "Arabic"
    output_language = "Arabic"
    agent.set_output_language(output_language)

    # Check if the output language is set correctly
    assert agent.output_language == output_language

    # Verify that the system message is updated with the new output language
    updated_system_message = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant."
        "\nRegardless of the input language, you must output text in Arabic.",
    )
    assert agent.system_message.content == updated_system_message.content

