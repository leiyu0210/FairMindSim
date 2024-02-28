from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from camel.agents import BaseAgent, ChatAgent
from camel.agents.chat_agent import FunctionCallingRecord
from camel.messages import BaseMessage, OpenAIMessage
from camel.responses import ChatAgentResponse
from camel.types import ChatCompletion, ChatCompletionChunk, ModelType, OpenAIBackendRole, RoleType


class Format_ChatAgent(ChatAgent):
    def step(
        self,
        input_message: BaseMessage,
        format=False,
    ) -> ChatAgentResponse:
        r"""Performs a single step in the chat session by generating a response
        to the input message.

        Args:
            input_message (BaseMessage): The input message to the agent.
            Its `role` field that specifies the role at backend may be either
            `user` or `assistant` but it will be set to `user` anyway since
            for the self agent any incoming message is external.

        Returns:
            ChatAgentResponse: A struct containing the output messages,
                a boolean indicating whether the chat session has terminated,
                and information about the chat session.
        """
        self.update_memory(input_message, OpenAIBackendRole.USER)

        output_messages: List[BaseMessage]
        info: Dict[str, Any]
        called_funcs: List[FunctionCallingRecord] = []
        while True:
            # Format messages and get the token number
            openai_messages: Optional[List[OpenAIMessage]]

            try:
                openai_messages, num_tokens = self.memory.get_context()
            except RuntimeError as e:
                return self.step_token_exceed(e.args[1], called_funcs,
                                              "max_tokens_exceeded")

            # Obtain the model's response
            response = self.model_backend.run(openai_messages)
            if isinstance(response, ChatCompletion):
                output_messages, finish_reasons, usage_dict, response_id = (
                    self.handle_batch_response(response))
            else:
                output_messages, finish_reasons, usage_dict, response_id = (
                    self.handle_stream_response(response, num_tokens))
            if format:
                content = response.choices[0].message.function_call.arguments
                return ChatAgentResponse([BaseMessage("player", RoleType.ASSISTANT, {}, content=content)], False, {})
            if (self.is_function_calling_enabled()
                    and finish_reasons[0] == 'function_call'
                    and isinstance(response, ChatCompletion)):
                # Do function calling
                func_assistant_msg, func_result_msg, func_record = (
                    self.step_function_call(response))

                # Update the messages
                self.update_memory(func_assistant_msg,
                                   OpenAIBackendRole.ASSISTANT)
                self.update_memory(func_result_msg, OpenAIBackendRole.FUNCTION)

                # Record the function calling
                called_funcs.append(func_record)
            else:
                # Function calling disabled or not a function calling

                # Loop over responses terminators, get list of termination
                # tuples with whether the terminator terminates the agent
                # and termination reason
                termination = [
                    terminator.is_terminated(output_messages)
                    for terminator in self.response_terminators
                ]
                # Terminate the agent if any of the terminator terminates
                self.terminated, termination_reason = next(
                    ((terminated, termination_reason)
                     for terminated, termination_reason in termination
                     if terminated), (False, None))
                # For now only retain the first termination reason
                if self.terminated and termination_reason is not None:
                    finish_reasons = [termination_reason] * len(finish_reasons)

                info = self.get_info(
                    response_id,
                    usage_dict,
                    finish_reasons,
                    num_tokens,
                    called_funcs,
                )
                break

        return ChatAgentResponse(output_messages, self.terminated, info)
