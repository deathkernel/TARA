# TARA browser chat

This is the first browser-based chat interface for TARA. It uses the existing
TARABrain response path and the local Ollama provider; it does not require a
separate JavaScript package manager or hosted API key.

## Start

1. Install Python dependencies from the repository root:
   \`python -m pip install -r requirements.txt\`
2. Make sure Ollama is running and the chosen model is installed:
   \`ollama list\`
3. Start the local web server:
   \`python -m src.web_app\`
4. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

If the default model is not installed, run \`ollama pull llama3.2:3b\`, or
choose another installed model with \`python -m src.web_app --model MODEL_NAME\`.
The default bind address is \`127.0.0.1\`, so the interface is only exposed on
the local machine unless the user deliberately changes the bind address.

## Troubleshooting

- The connection indicator checks \`/api/health\` and shows whether Ollama and
  the configured model are available.
- If Ollama is offline, open the Ollama app and refresh the browser.
- If the model is missing, use \`ollama list\` and install the model named by the
  status panel.
- The server returns clear validation errors for empty or oversized requests.
- Chat history is held in the current browser page only. Starting a new chat or
  refreshing the page clears that UI history; TARA's local memory behavior is
  otherwise unchanged.
