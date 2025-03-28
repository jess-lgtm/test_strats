import json
import time
from collections.abc import Generator
from copy import deepcopy
from typing import Any, Optional, cast

from pydantic import BaseModel

from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model import ModelFeature
from dify_plugin.entities.model.llm import (
    LLMModelConfig,
    LLMResult,
    LLMResultChunk,
    LLMUsage,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageContentType,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.entities.tool import LogMetadata, ToolInvokeMessage, ToolProviderType
from dify_plugin.interfaces.agent import AgentModelConfig, AgentStrategy, ToolEntity, ToolInvokeMeta


class FunctionCallingParams(BaseModel):
    query: str
    instruction: str | None
    model: AgentModelConfig
    tools: list[ToolEntity] | None
    maximum_iterations: int = 3


class FunctionCallingAgentStrategy(AgentStrategy):
    def __init__(self, session):
        super().__init__(session)
        self.query = ""
        self.instruction = ""

    @property
    def _user_prompt_message(self) -> UserPromptMessage:
        return UserPromptMessage(content=self.query)

    @property
    def _system_prompt_message(self) -> SystemPromptMessage:
        return SystemPromptMessage(content=self.instruction)

    def _invoke(self, parameters: dict[str, Any]) -> Generator[AgentInvokeMessage]:
        """
        Run FunctionCall agent application
        """
        fc_params = FunctionCallingParams(**parameters)

        # 初始化对话历史
        self.query = fc_params.query
        self.instruction = fc_params.instruction
        history_prompt_messages = fc_params.model.history_prompt_messages

        # 解析工具信息
        tools = fc_params.tools
        tool_instances = {tool.identity.name: tool for tool in tools} if tools else {}
        prompt_messages_tools = self._init_prompt_tools(tools)

        # 解析 LLM 参数
        stream = (
            ModelFeature.STREAM_TOOL_CALL in fc_params.model.entity.features
            if fc_params.model.entity and fc_params.model.entity.features
            else False
        )
        model = fc_params.model
        stop = fc_params.model.completion_params.get("stop", []) if fc_params.model.completion_params else []

        # 初始化 function calling 状态
        iteration_step = 1
        max_iteration_steps = fc_params.maximum_iterations
        current_thoughts: list[PromptMessage] = []
        function_call_state = True  
        llm_usage: dict[str, Optional[LLMUsage]] = {"usage": None}
        final_answer = ""

        while function_call_state and iteration_step <= max_iteration_steps:
            function_call_state = False
            round_started_at = time.perf_counter()
            round_log = self.create_log_message(
                label=f"ROUND {iteration_step}",
                data={},
                metadata={LogMetadata.STARTED_AT: round_started_at},
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield round_log

            if iteration_step == max_iteration_steps:
                prompt_messages_tools = []

            # 组织 Prompt 消息
            prompt_messages = self._organize_prompt_messages(
                history_prompt_messages=history_prompt_messages,
                current_thoughts=current_thoughts,
            )
            if model.entity and model.completion_params:
                self.recalc_llm_max_tokens(model.entity, prompt_messages, model.completion_params)

            # 调用 LLM 生成响应
            model_started_at = time.perf_counter()
            model_log = self.create_log_message(
                label=f"{model.model} Thought",
                data={},
                metadata={LogMetadata.STARTED_AT: model_started_at, LogMetadata.PROVIDER: model.provider},
                parent=round_log,
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield model_log

            model_config = LLMModelConfig(**model.model_dump(mode="json"))
            chunks: Generator[LLMResultChunk, None, None] | LLMResult = self.session.model.llm.invoke(
                model_config=model_config,
                prompt_messages=prompt_messages,
                stop=stop,
                stream=stream,
                tools=prompt_messages_tools,
            )

            tool_calls: list[tuple[str, str, dict[str, Any]]] = []
            response = ""
            tool_call_names = ""
            current_llm_usage = None

            if isinstance(chunks, Generator):
                for chunk in chunks:
                    if self.check_tool_calls(chunk):
                        function_call_state = True
                        tool_calls.extend(self.extract_tool_calls(chunk) or [])
                        tool_call_names = ";".join([tool_call[1] for tool_call in tool_calls])

                    if chunk.delta.message and chunk.delta.message.content:
                        response += str(chunk.delta.message.content)
                        if not function_call_state or iteration_step == max_iteration_steps:
                            yield self.create_text_message(str(chunk.delta.message.content))

                    if chunk.delta.usage:
                        self.increase_usage(llm_usage, chunk.delta.usage)
                        current_llm_usage = chunk.delta.usage

            else:
                result = cast(LLMResult, chunks)
                if self.check_blocking_tool_calls(result):
                    function_call_state = True
                    tool_calls.extend(self.extract_blocking_tool_calls(result) or [])
                    tool_call_names = ";".join([tool_call[1] for tool_call in tool_calls])

                if result.usage:
                    self.increase_usage(llm_usage, result.usage)
                    current_llm_usage = result.usage

                if result.message and result.message.content:
                    response += str(result.message.content)

                yield self.create_text_message(response)

            yield self.finish_log_message(
                log=model_log,
                data={"output": response, "tool_name": tool_call_names},
                metadata={LogMetadata.STARTED_AT: model_started_at, LogMetadata.FINISHED_AT: time.perf_counter()},
            )

            current_thoughts.append(AssistantPromptMessage(content=response, tool_calls=[]))

            final_answer += response + "\n"

            # 处理工具调用
            tool_responses = []
            for tool_call_id, tool_call_name, tool_call_args in tool_calls:
                tool_instance = tool_instances[tool_call_name]
                tool_call_started_at = time.perf_counter()

                # 调用工具
                tool_invoke_responses = self.session.tool.invoke(
                    provider_type=ToolProviderType(tool_instance.provider_type),
                    provider=tool_instance.identity.provider,
                    tool_name=tool_instance.identity.name,
                    parameters={**tool_instance.runtime_parameters, **tool_call_args},
                )
                result = " ".join(
                    cast(ToolInvokeMessage.TextMessage, response.message).text
                    for response in tool_invoke_responses if response.type == ToolInvokeMessage.MessageType.TEXT
                )

                # 关键改动
                placeholder_message = "工具已经将接下来的指挥权移交给 user"
                tool_message = AssistantPromptMessage(content=placeholder_message, tool_calls=[])
                new_user_message = UserPromptMessage(content=result)

                history_prompt_messages.append(new_user_message)
                current_thoughts.append(tool_message)

            yield self.create_json_message({"execution_metadata": llm_usage})
