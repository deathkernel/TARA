## Read-only public API tools

TARA's optional `src.public_api_tools.PublicAPITools` exposes curated, unauthenticated,
read-only queries to the GitHub repository search API, Wikipedia OpenSearch API, and
Open-Meteo geocoding/forecast APIs. Example:

```python
from src.public_api_tools import PublicAPITools

apis = PublicAPITools()
repos = apis.search_github_repositories("open source reasoning agents")
articles = apis.search_wikipedia("artificial intelligence")
weather = apis.get_weather("Pune")
```

The interactive CLI also exposes explicit, read-only commands:

- `/api github <query>`
- `/api wiki <query>`
- `/api weather <city>`

These commands call only the selected, allowlisted public API. The model cannot request
arbitrary URLs or trigger tools through generated text. Results are displayed as bounded
JSON with a source URL and `trusted=false`; the CLI does not save them to conversation
memory, datasets, or model checkpoints. Treat remote content as untrusted input and
independently verify important claims. Requests use HTTPS and an allowlist, a timeout,
a response-size cap, basic per-host pacing, JSON/shape validation, and explicit
rate-limit/error reporting.
