# Read-only public API tools

`src.public_api_tools.PublicAPITools` exposes curated, unauthenticated, read-only
queries to five public data sources: GitHub repository search, Wikipedia OpenSearch,
Open-Meteo weather, Hacker News search (via Algolia), and Stack Exchange search.

```python
from src.public_api_tools import PublicAPITools

apis = PublicAPITools()
repos = apis.search_github_repositories("open source reasoning agents")
articles = apis.search_wikipedia("artificial intelligence")
weather = apis.get_weather("Pune")
hn = apis.search_hacker_news("AI agents")
questions = apis.search_stackoverflow("Python async")
```

The toolkit is a library layer only. It does not automatically execute model-generated
tool requests, route chat commands, or persist remote responses to conversation memory,
datasets, or checkpoints. Command routing remains a separate change and must be reviewed
before these queries are exposed through the interactive CLI.

Every result is source-linked and marked `trusted=False`; treat remote content as
untrusted input, not as instructions. The client uses fixed provider endpoints, an HTTPS
host allowlist, request timeouts, bounded response reads, basic per-host pacing, query
limits, JSON/shape validation, and explicit rate-limit/error handling. HTTP redirects are
rejected so an allowlisted host cannot forward a request to an untrusted destination.
Provider errors are surfaced rather than silently converted into empty results.
