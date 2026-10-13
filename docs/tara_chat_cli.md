# TARA local chat on Windows

The interactive CLI lets you talk to **TARA** while the local Ollama service runs the Llama model behind the scenes.

## Start

From the TARA repository root:

```cmd
ollama list
python -m src
```

The first command should list `llama3.2:3b`. If it is missing, download it once:

```cmd
ollama pull llama3.2:3b
```

The CLI defaults to the Ollama provider. It uses TARA's brain response path, including its memory/context construction, and asks the local model to generate the response. It does not require a TARA training checkpoint for normal Ollama-backed replies.

## Chat commands

- Type a message and press Enter to talk to TARA.
- Type `/help` to see commands.
- Type `/exit`, `/quit`, or `/bye` to leave.

## Notes

- Ollama must be installed and its local service must be running.
- This setup uses a pretrained model; it does not train Llama or automatically fine-tune it.
- The CLI currently keeps TARA's in-memory state for the lifetime of the process. It does not promise durable conversation history between separate CLI runs.
- If Ollama is unavailable, the CLI reports the error rather than claiming a response was generated.
