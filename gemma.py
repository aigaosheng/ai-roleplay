import os
from typing import Any, Dict, List, Optional, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
    Generation,
    LLMResult,
)
from langchain_core.pydantic_v1 import BaseModel, root_validator

from langchain_community.llms import LlamaCpp
from langchain_google_vertexai._utils import enforce_stop_tokens

USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"


def gemma_messages_to_prompt(history: List[BaseMessage]) -> str:
    """Converts a list of messages to a chat prompt for Gemma."""
    messages: List[str] = []
    if len(messages) == 1:
        content = cast(str, history[0].content)
        if isinstance(history[0], SystemMessage):
            raise ValueError("Gemma currently doesn't support system message!")
        return content
    for message in history:
        content = cast(str, message.content)
        if isinstance(message, SystemMessage):
            messages.append(MODEL_CHAT_TEMPLATE.format(prompt=content))
            # raise ValueError("Gemma currently doesn't support system message!")
        elif isinstance(message, AIMessage):
            messages.append(MODEL_CHAT_TEMPLATE.format(prompt=content))
        elif isinstance(message, HumanMessage):
            messages.append(USER_CHAT_TEMPLATE.format(prompt=content))
        else:
            raise ValueError(f"Unexpected message with type {type(message)}")
    messages.append("<start_of_turn>model\n")
    # print(f"**** messages = {messages}")
    return "".join(messages)


def _parse_gemma_chat_response(response: str) -> str:
    """Removes chat history from the response."""
    pattern = "<start_of_turn>model\n"
    pos = response.rfind(pattern)
    if pos == -1:
        return response
    text = response[(pos + len(pattern)) :]
    pos = text.find("<start_of_turn>user\n")
    if pos > 0:
        return text[:pos]
    return text


class _GemmaBase(BaseModel):
    max_tokens: Optional[int] = None
    """The maximum number of tokens to generate."""
    temperature: Optional[float] = None
    """The temperature to use for sampling."""
    top_p: Optional[float] = None
    """The top-p value to use for sampling."""
    top_k: Optional[int] = None
    """The top-k value to use for sampling."""

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling gemma."""
        params = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        return {k: v for k, v in params.items() if v is not None}

    def _get_params(self, **kwargs) -> Dict[str, Any]:
        return {k: kwargs.get(k, v) for k, v in self._default_params.items()}


class _GemmaLocalBase(_GemmaBase):
    """Local gemma model loaded from HuggingFace."""

    tokenizer: Any = None  #: :meta private:
    client: Any = None  #: :meta private:
    hf_access_token: str
    cache_dir: Optional[str] = None
    model_name: str = "gemma_2b_en"
    """Gemma model name."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that llama-cpp-python library is installed."""
        # try:
        #     from transformers import AutoTokenizer, GemmaForCausalLM  # type: ignore
        # except ImportError:
        #     raise ImportError(
        #         "Could not import GemmaForCausalLM library. "
        #         "Please install the GemmaForCausalLM library to "
        #         "use this  model: pip install transformers>=4.38.1"
        #     )

        values["tokenizer"] = None
        values["client"] = LlamaCpp(
                model_path=values["model_name"],
                n_gpu_layers=-1,
                # n_batch = chunk_size,
                # callback_manager=callback_manager,
                n_ctx=1024*2, # Uncomment to increase the context window
                # temperature=0.75,
                # f16_kv=True,
                verbose=True,  # Verbose is required to pass to the callback manager
        )
        
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling gemma."""
        params = {"max_length": self.max_tokens}
        return {k: v for k, v in params.items() if v is not None}

    def _get_params(self, **kwargs) -> Dict[str, Any]:
        mapping = {"max_tokens": "max_length"}
        params = {mapping[k]: v for k, v in kwargs.items() if k in mapping}
        return {**self._default_params, **params}

    def _run(self, prompt: str, **kwargs: Any) -> str:
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # params = self._get_params(**kwargs)
        # generate_ids = self.client.generate(inputs.input_ids, **params)
        return self.client(prompt)
    
class GemmaLocal(_GemmaLocalBase, BaseLLM):
    """Local gemma model loaded from HuggingFace."""

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        results = [self._run(prompt, **kwargs) for prompt in prompts]
        if stop:
            results = [enforce_stop_tokens(text, stop) for text in results]
        return LLMResult(generations=[[Generation(text=text)] for text in results])

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "gemma_local_hf"


class GemmaChatLocal(_GemmaLocalBase, BaseChatModel):
    parse_response: bool = False
    """Whether to post-process the chat response and clean repeations """
    """or multi-turn statements."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = gemma_messages_to_prompt(messages)
        text = self._run(prompt, **kwargs)
        if self.parse_response or kwargs.get("parse_response"):
            text = _parse_gemma_chat_response(text)
        if stop:
            text = enforce_stop_tokens(text, stop)
        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "gemma_local_chat"
