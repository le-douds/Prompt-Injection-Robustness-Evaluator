from __future__ import annotations

import argparse
import asyncio
import logging

from pathlib import Path

from typing import Callable

import llm

from prompt_injection_benchmark import PromptInjectionBenchmark
from cyberseceval_benchmark import CyberSecEvalBenchmark
from tensor_trust_benchmark import TensorTrustBenchmark
from hackaprompt_benchmark import HackAPromptBenchmark

LOG: logging.Logger = logging.getLogger(__name__)

PromptInjectionBenchmark.register_service(CyberSecEvalBenchmark)
PromptInjectionBenchmark.register_service(TensorTrustBenchmark)
PromptInjectionBenchmark.register_service(HackAPromptBenchmark)


def validate_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    if not args.use_precomputed_responses:
        if args.prompt_path is None:
            parser.error(
                "The following argument is required: --prompt-path, "
                "unless argument --use-precomputed-responses is specified."
            )
        if "llm_under_test" in args and args.llm_under_test is None:
            parser.error(
                "The following argument is required: --llm-under-test, "
                "unless argument --use-precomputed-responses is specified."
            )

    return args


def main(
    *,
    default_benchmark: str | None = None,
    llms_under_test: list[llm.LLM] | None = None,
    judge_llm: llm.LLM | None = None,
    default_prompt_path: str | None = None,
    default_response_path: str | None = None,
    default_judge_response_path: str | None = None,
    default_stat_path: str | None = None,
    default_use_precomputed_responses: bool = False,
    default_only_generate_llm_responses: bool = False,
    default_llm_creator: Callable[[str], llm.LLM] = llm.create,
) -> None:
    parser = argparse.ArgumentParser(description="Run the Cybersecurity Benchmarks.")

    parser.add_argument(
        "--benchmark",
        default=default_benchmark,
        required=default_benchmark is None,
        help="This is the argument to specify which benchmark to run. "
        "Currently supported benchmarks are: "
        "autocomplete, instruct, mitre.",
    )

    if llms_under_test is None:
        parser.add_argument(
            "--llm-under-test",
            help=f"LLM to benchmark provided as {llm.SPECIFICATION_FORMAT}, "
            f"e.g., {llm.EXAMPLE_SPECIFICATION}",
            action="append",
        )

    parser.add_argument(
        "--prompt-path",
        help="The under-test LLM(s) would be queried with the prompts in this file. "
        "This argument works for benchmarks: autocomplete, instruct, mitre.",
        default=default_prompt_path,
    )

    parser.add_argument(
        "--response-path",
        help="The under-test LLM(s) would store their responses in this file. "
        "This argument works for benchmarks: "
        "autocomplete, instruct, mitre.",
        default=default_response_path,
        required=default_response_path is None,
    )

    parser.add_argument(
        "--stat-path",
        help="After processing the under-test LLM's responses, the final "
        "status would be stored in this file. This argument works "
        "for benchmarks: autocomplete, instruct, mitre.",
        default=default_stat_path,
        required=False,
    )

    parser.add_argument(
        "--judge-llm",
        help=f"Used for MITRE benchmark. We need to specify the llm model "
        f"to judge the final expanded response as {llm.SPECIFICATION_FORMAT}, "
        f"e.g., {llm.EXAMPLE_SPECIFICATION}",
        required=False,
    )

    parser.add_argument(
        "--judge-response-path",
        help="The path to save the judge responses.",
        default=default_judge_response_path,
        required=False,
    )

    parser.add_argument(
        "--run-llm-in-parallel",
        action="store_true",
        help="If this flag is set, we will make parallel calls to the LLM to "
        "speed up execution.",
    )
    parser.add_argument(
        "--num-queries-per-prompt",
        help="Set the number of queries per prompt. Default value is 1.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--use-precomputed-responses",
        action="store_true",
        default=default_use_precomputed_responses,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--only-generate-llm-responses",
        action="store_true",
        default=default_only_generate_llm_responses,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--num-test-cases",
        type=int,
        default=0,
        help="Only set this argument for testing purpose: If set to a positive value, "
        "we test the given number of "
        "prompts instead of all prompts.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = validate_arguments(parser)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO if not args.debug else logging.DEBUG,
    )

    materialized_llms_under_test = []
    if not args.use_precomputed_responses:
        if llms_under_test is None and not args.llm_under_test:
            raise ValueError("No LLMs were specified.")
        materialized_llms_under_test = llms_under_test or [
            default_llm_creator(specification) for specification in args.llm_under_test
        ]

    stat_response_path = Path(args.stat_path) if args.stat_path else None
    judge_response_path = (
        Path(args.judge_response_path) if args.judge_response_path else None
    )
    if args.judge_llm:
        judge_llm = default_llm_creator(args.judge_llm)

    benchmark: PromptInjectionBenchmark = PromptInjectionBenchmark.create_instance(
        args.benchmark,
        llms_under_test=materialized_llms_under_test,
        judge_llm=judge_llm,
        response_path=Path(args.response_path),
        judge_response_path=judge_response_path,
        num_test_cases=args.num_test_cases,
        stat_path=stat_response_path,
    )

    if args.use_precomputed_responses:
        # Using precomputed responses require a valid response path.
        response_path = Path(args.response_path)
        if not response_path.exists():
            raise ValueError(
                f"Please provide valid path for responses. This path "
                f"does not exist: {response_path}"
            )
        LOG.info("Skipping LLM query and using precomputed response.")
    else:
        # If we are not using precomputed responses, then we require a valid prompt path.
        prompt_path = Path(args.prompt_path)
        if not prompt_path.exists():
            raise ValueError(
                f"Please provide valid path for prompts. "
                f"This path does not exist: {prompt_path}"
            )
        LOG.info("Querying LLMs for responses...")
        benchmark.query_llm_to_generate_responses(
            prompt_path=Path(args.prompt_path),
            run_llm_in_parallel=args.run_llm_in_parallel,
        )

    if not args.only_generate_llm_responses:
        LOG.info("Evaluating...")
        # The variable num_test_cases is defined to specify the
        # number of prompts that need to be evaluated.
        # However, if we are testing multiple LLMs or if we are
        # conducting multiple rounds of testing queries per
        # prompt, these factors will need below steps to reflect
        # in `num_test_cases`` during the expansion/judge step.
        if not args.use_precomputed_responses:
            if len(materialized_llms_under_test) > 0:
                args.num_test_cases = (
                    len(materialized_llms_under_test) * args.num_test_cases
                )

        asyncio.run(benchmark.run())


if __name__ == "__main__":
    main()
