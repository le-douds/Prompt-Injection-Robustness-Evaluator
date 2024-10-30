from __future__ import annotations
import os
import logging
import json
import time
import configparser
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union
import openai
import google.auth
import google.auth.transport.requests

from typing_extensions import override

NUM_LLM_RETRIES = 100

MAX_TOKENS = 1000

LOG: logging.Logger = logging.getLogger(__name__)


class LLM(ABC):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        if model not in self.valid_models():
            LOG.warning(
                f"{model} is not in the valid model list for {type(self).__name__}. "
                "Valid models are: {', '.join(self.valid_models())}."
            )
        self.model: str = model
        self.api_key: str | None = api_key

    @abstractmethod
    def query(
        self, prompt: str, guided_decode_json_schema: Optional[str] = None
    ) -> str:
        """
        Abstract method to query an LLM with a given prompt and return the response.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The response from the LLM.
        """
        pass

    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        """
        Abstract method to query an LLM with a given prompt and system prompt and
        return the response.

        Args:
            system prompt (str): The system prompt to send to the LLM.
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The response from the LLM.
        """
        return self.query(system_prompt + "\n" + prompt)

    def query_with_pre_prompt_post_prompt(
        self,
        pre_prompt: str,
        prompt: str,
        post_prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        """
        Abstract method to query an LLM with a given prompt and system prompt and
        return the response.

        Args:
            pre_prompt (str): The pre prompt to send to the LLM.
            prompt (str): The prompt to send to the LLM.
            post_prompt (str): The post prompt to send to the LLM.

        Returns:
            str: The response from the LLM.
        """
        return self.query(pre_prompt + "\n" + prompt + "\n", post_prompt)

    def _query_with_retries(
        self,
        func: Callable[..., str],
        *args: Union[str, bool, Optional[str]],
        retries: int = NUM_LLM_RETRIES,
        backoff_factor: float = 0.5,
    ) -> str:
        last_exception = None
        for retry in range(retries):
            try:
                return func(*args)
            except Exception as exception:
                last_exception = exception
                sleep_time = backoff_factor * (2**retry)
                time.sleep(sleep_time)
                LOG.debug(
                    f"LLM Query failed with error: {exception}. "
                    f"Sleeping for {sleep_time} seconds..."
                )
                if (
                    isinstance(last_exception, openai.BadRequestError)
                    and last_exception.status_code == 400
                ):
                    response_content = json.loads(last_exception.response.content)
                    if isinstance(response_content, list):
                        return response_content[0]["error"]["message"]
                    if isinstance(response_content, dict):
                        return response_content["error"]["message"]

        raise RuntimeError(
            f"Unable to query LLM after {retries} retries: {last_exception}"
        )

    def query_with_retries(
        self, prompt: str, guided_decode_json_schema: Optional[str] = None
    ) -> str:
        return self._query_with_retries(self.query, prompt, guided_decode_json_schema)

    def query_with_system_prompt_with_retries(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        return self._query_with_retries(
            self.query_with_system_prompt,
            system_prompt,
            prompt,
            guided_decode_json_schema,
        )

    def query_with_pre_prompt_post_prompt_with_retries(
        self,
        pre_prompt: str,
        prompt: str,
        post_prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        return self._query_with_retries(
            self.query_with_pre_prompt_post_prompt,
            pre_prompt,
            prompt,
            post_prompt,
            guided_decode_json_schema,
        )

    def valid_models(self) -> list[str]:
        """List of valid model parameters, e.g. 'gpt-3.5-turbo' for GPT"""
        return []


SPECIFICATION_FORMAT: str = "<PROVIDER>::<MODEL>(::<API KEY>)?"
EXAMPLE_SPECIFICATION: str = "OPENAI::gpt-3.5-turbo::<YOUR API KEY>"


def create(identifier: str) -> LLM:
    """
    Takes a string of the format <SPECIFICATION_FORMAT> and generates LLM instance.
    See <EXAMPLE_SPECIFICATION> for example.
    """
    api_key: str | None = None
    split = identifier.split("::")
    if len(split) == 2:
        provider, name = split
    elif len(split) == 3:
        provider, name, api_key = split
    else:
        raise ValueError(f"Invalid identifier: {identifier}")

    # if api_key is None:
    #     raise ValueError(f"No API key provided for {provider}: {name}")

    if provider == "OPENAI":
        return OPENAI(name, api_key)
    if provider == "ANYSCALE":
        return ANYSCALE(name, api_key)
    if provider == "TOGETHER":
        return TOGETHER(name, api_key)
    if provider == "AZURE_OPENAI":
        return AZURE_OPENAI(name, api_key)
    if provider == "VERTEXAI":
        return VERTEXAI(name, api_key)

    raise ValueError(f"Unknown provider: {provider}")


class VERTEXAI(LLM):
    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model, api_key)
        config_file = "config.ini"
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file '{config_file}' not found.")
        config = configparser.ConfigParser()
        config.read(config_file)
        region = config["vertexai"]["region"]

        try:
            creds, project = google.auth.default()
            if creds and project:
                auth_req = google.auth.transport.requests.Request()
                creds.refresh(auth_req)
                api_key = creds.token
        except google.auth.exceptions.DefaultCredentialsError as e:
            LOG.info(e)
            api_key = config["vertexai"]["key"]
            project = config["vertexai"]["project"]

        self.client = openai.OpenAI(
            base_url=f"https://{region}-aiplatform.googleapis.com"
                     f"/v1beta1/projects/{project}/locations/{region}/endpoints/openapi",
            api_key=api_key,
        )

    @override
    def query(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
        )
        logging.getLogger().setLevel(level)
        if response.choices[0].finish_reason == "content_filter":
            return "content_filter"
        return response.choices[0].message.content

    @override
    def query_with_pre_prompt_post_prompt(
        self,
        pre_prompt: str,
        prompt: str,
        post_prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        if not pre_prompt:
            pre_prompt = "."  # Bypass the fact that Gemini does not handle empty prompt
        if not post_prompt:
            post_prompt = (
                "."  # Bypass the fact that Gemini does not handle empty prompt
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": pre_prompt},
                {"role": "user", "content": prompt},
                {"role": "user", "content": post_prompt},
            ],
            max_tokens=MAX_TOKENS,
        )
        logging.getLogger().setLevel(level)
        if not response.choices or response.choices[0].finish_reason == "content_filter":
            return "content_filter"

        return response.choices[0].message.content

    @override
    def valid_models(self) -> list[str]:
        return [
            "google/gemini-1.5-flash-001",
            "google/gemini-1.5-pro-001",
            "meta/llama3-405b-instruct-maas",
        ]


class OPENAI(LLM):
    """Accessing OPENAI"""

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model, api_key)
        if not api_key:
            config_file = "config.ini"
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file '{config_file}' not found.")
            config = configparser.ConfigParser()
            config.read(config_file)
            api_key = config["openai"]["key"]

        self.client = openai.OpenAI(api_key=api_key)  # noqa

    @override
    def query(
        self, prompt: str, guided_decode_json_schema: Optional[str] = None
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def query_with_pre_prompt_post_prompt(
        self,
        pre_prompt: str,
        prompt: str,
        post_prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": pre_prompt},
                {"role": "user", "content": prompt},
                {"role": "user", "content": post_prompt},
            ],
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def valid_models(self) -> list[str]:
        return ["gpt-3.5-turbo", "gpt-4"]


class AZURE_OPENAI(LLM):
    """Accessing AZURE OPENAI"""

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model, api_key)
        config_file = "config.ini"
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file '{config_file}' not found.")
        config = configparser.ConfigParser()
        config.read(config_file)
        api_key = config["azure-openai"]["key"]
        endpoint = config["azure-openai"]["endpoint"]
        api_version = config["azure-openai"]["api_version"]

        self.client = openai.AzureOpenAI(
            azure_endpoint=endpoint, api_key=api_key, api_version=api_version
        )

    @override
    def query(
        self, prompt: str, guided_decode_json_schema: Optional[str] = None
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def query_with_pre_prompt_post_prompt(
        self,
        pre_prompt: str,
        prompt: str,
        post_prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": pre_prompt},
                {"role": "user", "content": prompt},
                {"role": "user", "content": post_prompt},
            ],
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def valid_models(self) -> list[str]:
        return []


class ANYSCALE(LLM):
    """Accessing ANYSCALE"""

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model, api_key)
        if not api_key:
            config_file = "config.ini"
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file '{config_file}' not found.")
            config = configparser.ConfigParser()
            config.read(config_file)
            api_key = config["anyscale"]["key"]

        self.client = openai.OpenAI(
            base_url="https://api.endpoints.anyscale.com/v1", api_key=api_key
        )

    @override
    def query(
        self, prompt: str, guided_decode_json_schema: Optional[str] = None
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def query_with_pre_prompt_post_prompt(
        self,
        pre_prompt: str,
        prompt: str,
        post_prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": pre_prompt},
                {"role": "user", "content": prompt},
                {"role": "user", "content": post_prompt},
            ],
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def valid_models(self) -> list[str]:
        return [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "HuggingFaceH4/zephyr-7b-beta",
            "codellama/CodeLlama-70b-Instruct-hf",
        ]


class TOGETHER(LLM):
    """Accessing TOGETHER"""

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model, api_key)
        if not api_key:
            config_file = "config.ini"
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file '{config_file}' not found.")
            config = configparser.ConfigParser()
            config.read(config_file)
            api_key = config["together"]["key"]

        self.client = openai.OpenAI(
            base_url="https://api.together.xyz/v1", api_key=api_key
        )

    @override
    def query(
        self, prompt: str, guided_decode_json_schema: Optional[str] = None
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def query_with_pre_prompt_post_prompt(
        self,
        pre_prompt: str,
        prompt: str,
        post_prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": pre_prompt},
                {"role": "user", "content": prompt},
                {"role": "user", "content": post_prompt},
            ],
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def valid_models(self) -> list[str]:
        return [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "lmsys/vicuna-7b-v1.5",
            "togethercomputer/CodeLlama-7b",
            "togethercomputer/CodeLlama-7b-Python",
            "togethercomputer/CodeLlama-7b-Instruct",
            "togethercomputer/CodeLlama-13b",
            "togethercomputer/CodeLlama-13b-Python",
            "togethercomputer/CodeLlama-13b-Instruct",
            "togethercomputer/falcon-40b",
            "togethercomputer/llama-2-7b",
            "togethercomputer/llama-2-7b-chat",
            "togethercomputer/llama-2-13b",
            "togethercomputer/llama-2-13b-chat",
            "togethercomputer/llama-2-70b",
            "togethercomputer/llama-2-70b-chat",
            "meta-llama/Llama-3-8b-chat-hf",
            "meta-llama/Llama-3-70b-chat-hf",
        ]
