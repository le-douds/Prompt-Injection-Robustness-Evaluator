from __future__ import annotations
import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Type, Callable, cast, Dict, List, Optional, Union
from tqdm import tqdm
from llm import LLM

LOG: logging.Logger = logging.getLogger(__name__)


class PromptInjectionBenchmark(ABC):
    """
        The base class for Prompt Injection Evaluation datasets
    """
    _registered_evaluation_service: Dict[
        str, Type["PromptInjectionBenchmark"]
    ] = {}

    def __init__(
        self,
        *,
        llms_under_test: list[LLM],
        response_path: Path,
        judge_response_path: Optional[Path] = None,
        stat_path: Optional[Path] = None,
        num_test_cases: int = 0,
        judge_llm: Optional[LLM] = None,
    ) -> None:
        self.response_path: Path = response_path
        self.llms_under_test: list[LLM] = llms_under_test
        self.judge_response_path = judge_response_path
        self.stat_path = stat_path
        self.judge_llm = judge_llm
        self.num_test_cases = num_test_cases

    @classmethod
    def register_service(
        cls, evaluation_service_class: Type["PromptInjectionBenchmark"]
    ) -> None:
        """
        Registers all the Evaluation Services kinds with the Evaluation Service factory.
        """
        for evaluation_service_kind in evaluation_service_class.return_kind():
            cls._registered_evaluation_service[
                evaluation_service_kind
            ] = evaluation_service_class

    @classmethod
    def create_instance(
        cls, dataset_tester_kind: str, *args: Any, **kwargs: Any
    ) -> PromptInjectionBenchmark:
        if dataset_tester_kind in cls._registered_evaluation_service:
            evaluation_service_class: Type[
                PromptInjectionBenchmark
            ] = cls._registered_evaluation_service[dataset_tester_kind]
            # pyre-ignore [45]: This function is only used to create instances
            # of subclasses of `PromptInjectionDatasetEvaluationService`, whose
            # abstract methods are implemented
            return evaluation_service_class(*args, **kwargs)
        raise ValueError(
            f"Unknown evaluation service kind: {dataset_tester_kind}, "
            f"the registered evaluation services are: "
            f"{list(cls._registered_evaluation_service.keys())}"
        )

    @classmethod
    def return_kind(cls) -> list[str]:
        """
        Returns the kind(s) of Evaluation Service that is supported. One Evaluation Services
        may support multiple kinds.
        """
        raise NotImplementedError(
            "Each Evaluation Service must implement this method to return the kind for run script."
        )

    @abstractmethod
    def process_prompt_and_user_input(
        self,
        prompt_id: int,
        pass_id: int,
        llms: Dict[str, LLM],
        test_case: Dict[str, Union[int, str]],
    ) -> Dict[str, Union[int, str]]:
        """
        Returns the response from the queried LLM.
        """
        raise NotImplementedError(
            "Each Evaluation Service must implement this method to query LLMs"
        )

    @abstractmethod
    def judge_responses(self, num_test_cases: int = 0, run_llm_in_parallel: bool = True) -> None:
        """
        This function processes LLM responses and determines if
        "yes" or "no" the prompt injection was successful.
        It returns a dictionary containing the results.
        """
        raise NotImplementedError(
            "Each Evaluation Service must implement this method to judge LLMs' responses"
        )

    @abstractmethod
    def process_results(self, prompt_path: Path) -> None:
        """
        Processes the results of the benchmark.
        """
        raise NotImplementedError(
            "Each Evaluation Service must implement this method to query LLMs"
        )

    async def run(self, num_test_cases: int = 0, run_llm_in_parallel: bool = True) -> None:
        """
            Processes a LLM-response.
        """
        self.judge_responses(num_test_cases, run_llm_in_parallel)

        if (judge_path := self.judge_response_path) is not None:
            self.process_results(judge_path)

    def query_llm_to_generate_responses(
        self,
        prompt_path: Path,
        run_llm_in_parallel: bool = False,
        to_process_prompt: Callable[
            [
                int,
                int,
                Dict[str, LLM],
                Dict[str, Union[int, str]],
                Optional[str],
            ],
            Dict[str, Union[int, str]],
        ] = None,
        num_test_cases: int = 0,
        is_judging: bool = False,
    ) -> None:
        """
        Processes a dataset of prompts by sending them to the LLM and saving the
        responses in a file.
        """
        response_result = []
        if is_judging:
            logging.info(f"Judging the response using {self.judge_llm.model}")
            response_result.extend(
                query_llm_to_generate_responses(
                    {"judge_llm": self.judge_llm},
                    self.response_path,
                    run_llm_in_parallel,
                    to_process_prompt=to_process_prompt,
                    enumerate_prompt_and_pass_id=False,
                    num_test_cases=num_test_cases
                )
            )
            if (judge_path := self.judge_response_path) is not None:
                judge_path.write_text(json.dumps(response_result, indent=4))
        else:
            for llm in self.llms_under_test:
                logging.info(f"Querying {llm.model}")
                response_result.extend(
                    query_llm_to_generate_responses(
                        {"initial_query_llm": llm},
                        prompt_path,
                        run_llm_in_parallel,
                        to_process_prompt=self.process_prompt_and_user_input,
                        num_test_cases=self.num_test_cases,
                    )
                )

            self.response_path.write_text(json.dumps(response_result, indent=4))


def query_llm_to_generate_responses(
    llms: Dict[str, LLM],
    prompt_path: Path,
    is_parallel: bool = False,
    num_threads: int = 20,
    to_process_prompt: Callable[
        [
            int,
            int,
            Dict[str, LLM],
            Dict[str, Union[int, str]],
            Optional[str],
        ],
        Dict[str, Union[int, str]],
    ] = None,
    num_test_cases: int = 0,
    pass_k: int = 1,
    enumerate_prompt_and_pass_id: bool = True,
) -> List[Dict[str, Optional[str]]]:
    """
    Processes a dataset of prompts by sending them to the LLM and saving the responses
    in a file.
    """
    if not prompt_path.exists():
        raise ValueError("Please provide valid path for prompts.")
    prompt_dataset: List[Dict[str, Union[int, str]]] = json.loads(
        prompt_path.read_text()
    )
    if num_test_cases > 0:
        # Select the test cases distributed evenly across the dataset
        dataset_size = len(prompt_dataset)
        if dataset_size > num_test_cases:
            step = dataset_size // num_test_cases
            prompt_dataset = prompt_dataset[::step][:num_test_cases]
    response_result = []

    if is_parallel:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                (
                    executor.submit(to_process_prompt, idx, k, llms, test_case)
                    if enumerate_prompt_and_pass_id
                    else executor.submit(
                        to_process_prompt,
                        cast(int, test_case["prompt_id"]),
                        cast(int, test_case["pass_id"]),
                        llms,
                        test_case,
                    )
                )
                for idx, test_case in enumerate(prompt_dataset)
                for k in range(pass_k)
            ]

            for future in tqdm(
                futures,
                desc="Processing prompts",
                unit="prompt",
            ):
                try:
                    result = future.result()
                    response_result.append(result)
                except Exception as e:
                    LOG.error(f"Exception occurred in thread {future}: {e}")
    else:
        for idx, test_case in enumerate(
            tqdm(prompt_dataset * pass_k, desc="Processing prompts", unit="prompt")
        ):
            response_result.append(
                to_process_prompt(
                    idx % len(prompt_dataset),
                    idx // len(prompt_dataset),
                    llms,
                    test_case,
                )
            )
    return response_result
