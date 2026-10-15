# Approved teacher training data
 
This directory is the destination for **curated, human-approved Ollama teacher JSONL**. Do not commit generated candidate answers or personal conversation data here.
 
## Lifecycle
 
1. Generate candidates and audit them with `python -m src.teacher_pipeline prepare`.
2. Independently review every row in the CSV. Mark each `approve` or `reject` and enter a reviewer identifier.
3. Use `python -m src.teacher_pipeline curate` to write an approved-only JSONL file here.
4. Use `python -m src.teacher_pipeline train` to train a separate candidate checkpoint and compare it with a baseline on the held-out suite.
 
A valid record must originate from the configured Ollama teacher, pass schema/fingerprint/provenance checks, and include `review_status=approved` plus a recorded `review.decision=approve` and non-empty `review.reviewer`. A structural audit does not prove factual correctness.
 
The canonical default path used by the main training/preflight CLIs is `data/teacher/approved.jsonl`. This file is intentionally not supplied by the repository: create it through the review workflow. Existing output paths are not to be overwritten.
