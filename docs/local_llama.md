# Local Llama inference for TARA

TARA can use a pretrained Llama model served locally by [Ollama](https://ollama.com/).
This supplies language generation without pretraining TARA's small custom Transformer
from scratch. TARA still owns its memory, planning, tool permissions, and verification.

## 1. Install and download a model

Install Ollama for Windows, then open Command Prompt and run one model pull:

```bat
ollama pull llama3.2:3b
```

For a smaller model on a RAM-constrained CPU, use `ollama pull llama3.2:1b`
and set `TARA_LLAMA_MODEL=llama3.2:1b`. Model size, answer quality, speed, and
memory use trade off against one another. Pulling a model requires an internet
connection and downloads model files; the Python integration itself does not
download anything.

Start Ollama and confirm its local API is reachable before running TARA.

## 2. Configure TARA (Windows Command Prompt)

```bat
set TARA_LLM_BACKEND=ollama
set TARA_LLAMA_MODEL=llama3.2:3b
set OLLAMA_HOST=http://127.0.0.1:11434
set TARA_LLM_TIMEOUT_SECONDS=120
```

These settings apply only to the current Command Prompt window. No cloud API key
is required. CUDA may remain disabled; Ollama can use CPU, though generation can
be slower. GPU acceleration depends on a working Ollama/NVIDIA setup and is not
assumed by this integration.

## 3. Connect it to TARA's existing provider interface

The `TARABrain` constructor already accepts an `llm_provider`. When creating
the brain, pass the environment-selected provider:

```python
from src.brain import TARABrain
from src.llm_provider_factory import provider_from_env

brain = TARABrain(model, tokenizer, llm_provider=provider_from_env())
answer = brain.respond("Explain what TARA does.", max_new_tokens=256)
print(answer.text)
```

Here, `model` and `tokenizer` are the existing TARA model objects. When
`TARA_LLM_BACKEND` is unset (or set to `none`), the factory returns `None`
and TARA keeps its current built-in language model. An explicitly injected
provider takes precedence over environment selection.

## 4. What this changes—and what it does not

- **Avoids pretraining from scratch for general language generation.** The
  pretrained Llama weights already encode broad language patterns and knowledge.
- **Does not eliminate all training.** Optional fine-tuning/LoRA can later adapt
  the model to TARA-specific formats or tasks, but should follow a measured
  benchmark rather than be the first step.
- **Does not automatically improve every TARA subsystem.** Memory, tool safety,
  planning, retrieval, and verification remain TARA responsibilities.
- **Does not guarantee factual correctness.** Use retrieved evidence and TARA's
  evaluation/verification path for high-confidence answers.
- **Keeps model use local.** The provider calls the Ollama server configured by
  `OLLAMA_HOST`; it does not send prompts to a hosted API itself.

## Tests

The provider tests inject a fake transport. They do not need Ollama running,
downloaded weights, CUDA, or an API key:

```bat
python -m pytest -q tests/test_ollama_provider.py tests/test_llm_provider_factory.py
```
