## Read-only public API tools

TARA exposes curated, unauthenticated, read-only queries to five public data sources:
GitHub repository search, Wikipedia OpenSearch, Open-Meteo weather, Hacker News search
(via Algolia), and Stack Exchange search. Example:

```python
from src.public_api_tools import PublicAPITools

apis = PublicAPITools()
repos = apis.search_github_repositories("open source reasoning agents")
articles = apis.search_wikipedia("artificial intelligence")
weather = apis.get_weather("Pune")
hn = apis.search_hacker_news("AI agents")
questions = apis.search_stackoverflow("Python async")
```

The interactive CLI also exposes explicit, read-only commands:

- `/api github <query>`
- `/api wiki <query>`
- `/api weather <city>`
- `/api hn <query>`
- `/api stackoverflow <query>`

These commands call only the selected, allowlisted public API. The model cannot request
arbitrary URLs or trigger tools through generated text. Results are displayed as bounded
JSON with a source URL and `trusted=false`; the CLI does not save them to conversation
memory, datasets, or model checkpoints. Treat remote content as untrusted input and
independently verify important claims. Requests use HTTPS and an allowlist, a timeout,
a response-size cap, basic per-host pacing, query/result limits, JSON/shape validation,
and explicit rate-limit/error reporting. APIs may impose their own usage limits.
