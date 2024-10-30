# Prompt Injection Robustness Evaluator (PIRE)

PIRE is an automated testing system designed to evaluate the robustness of Large Language Models (LLMs) against prompt injection attacks.  It leverages a curated collection of open-source prompt injection datasets to rigorously assess and benchmark LLM performance in the face of adversarial prompts.

PIRE leverages the implementation of [META CyberSecEval 2](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks/benchmark) as a foundation but enhances its prompt injection part by supporting evaluation against a variety of open source prompt injection datasets.

**Key Features:**

* **Automated Testing:**  PIRE automates the process of testing LLMs against various prompt injection techniques, reducing manual effort and ensuring consistent evaluation.
* **Open-Source Datasets:** Utilizes publicly available prompt injection datasets, allowing for reproducible research and community contribution.  Supports easy integration of new datasets as they emerge.
* **Comprehensive Evaluation:** Provides a range of metrics to assess LLM robustness, including attack success rate, classification accuracy (for detecting malicious prompts), and the severity of successful injections.
* **Flexible and Extensible:** Designed to be easily adaptable to different LLMs and prompt injection scenarios.  Supports custom test configurations and reporting options.
* **Benchmarking and Comparison:** Facilitates comparing the robustness of different LLMs under consistent testing conditions.  Tracks performance over time to monitor improvements and regressions.

## How it works:
### Datasets supported
The current implementation supports the following datasets:
- META CyberSecEval 2 Dataset [[paper]](https://ai.meta.com/research/publications/cyberseceval-2-a-wide-ranging-cybersecurity-evaluation-suite-for-large-language-models/) [[dataset]](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks/datasets/prompt_injection);
- Tensor Trust Dataset [[paper]](https://github.com/HumanCompatibleAI/tensor-trust-data) [[dataset]](https://github.com/HumanCompatibleAI/tensor-trust-data);
- HackAPrompt Dataset [[paper]](https://arxiv.org/abs/2311.16119) [[dataset]](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset)

### LLM API Provider supported
- [OpenAI](https://platform.openai.com/docs/overview)
- [Together AI](https://api.together.xyz/signin)
- [Anyscale](https://docs.anyscale.com/endpoints/intro/)
- [VertexAI](https://cloud.google.com/vertex-ai/docs/reference/rest)
- [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/)

### LLM API Provider Configuration
**VertexAI** and **Azure OpenAI** configurations must be done via the [config.ini](config.ini) file. The other API providers' key can be passed via the command line running the benchmarks or via the [config.ini](config.ini) file.

## Examples of running the Benchmarks

### META CyberSecEval 2 Dataset
Example of running CyberSecEval 2 Dataset evaluation on `LLAMA 3 8b` via `TogetherAI` API:
```bash
run.py --benchmark=cybersec-eval
--prompt-path="$DATASETS_PATH/meta-cybersec-eval/prompt_injection.json"
--response-path="$DATASETS_PATH/meta-cybersec-eval/prompt_injection_responses.json"
--judge-response-path="$DATASETS_PATH/meta-cybersec-eval/prompt_injection_judge_responses.json"
--stat-path="$DATASETS_PATH/meta-cybersec-eval/prompt_injection_stat.json"
--judge-llm="TOGETHER::meta-llama/Llama-3-8b-chat-hf::<YOUR API KEY>"
--llm-under-test="TOGETHER::meta-llama/Llama-3-8b-chat-hf::<YOUR API KEY>"
```

### Tensor Trust Dataset
Example of running Tensor Trust Dataset evaluation on `gemini-1.5-flash-001` and `google/gemini-1.5-pro-001` models via `VertexAI` API:
```bash
run.py --benchmark=tensor-trust
--prompt-path="$DATASETS_PATH/tensor-trust/tensor_trust_dataset.json"
--response-path="$DATASETS_PATH/tensor-trust/tensor_trust_dataset_vertexai_responses.json"
--judge-response-path="$DATASETS_PATH/tensor-trust/tensor_trust_dataset_vertexai_judge_responses.json"
--stat-path="$DATASETS_PATH/tensor-trust/tensor_trust_dataset_vertexai_stat.json"
--llm-under-test="VERTEXAI::google/gemini-1.5-flash-001"
--llm-under-test="VERTEXAI::google/gemini-1.5-pro-001"
--run-llm-in-parallel
```

config.ini:
```bash
[vertexai]
key=<YOUR VERTEXAI API KEY>
project=vertexai-project
region=us-central1
```

### HackAPrompt Dataset
Example of running Tensor Trust Dataset evaluation on `gemini-1.5-flash-001` and `google/gemini-1.5-pro-001` models via `VertexAI` API:
```bash
run.py --benchmark=hackaprompt
--prompt-path="$DATASETS_PATH/hackaprompt/hackaprompt.json"
--response-path="$DATASETS_PATH/hackaprompt/hackaprompt_azure_openai_responses.json"
--judge-response-path="$DATASETS_PATH/hackaprompt/hackaprompt_azure_openai_judge_responses.json"
--stat-path="$DATASETS_PATH/hackaprompt_azure_openai_stat.json"
--llm-under-test="AZURE_OPENAI::gpt3.5-deployment"
--run-llm-in-parallel
```

config.ini:
```bash
[azure-openai]
key=<YOUR AZURE OPENAI API KEY>
endpoint=your-azure-openai-endpoint
api_version=your-api-version
```

## License:
MIT