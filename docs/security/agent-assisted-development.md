# Agent-assisted development security guardrails

**Reviewed:** 2026-10-23

These guardrails apply when a developer uses an AI coding agent to change TARA. They are workflow guidance, not an in-repository sandbox implementation. A sandbox limits the agent's execution capabilities; it does not replace code review, tests, or least-privilege credentials.

## Default workflow

1. **Start with the narrow task and workspace.** Ask the agent to inspect relevant files first and make a focused change. Keep generated changes on a feature branch and use a pull request for review.
2. **Enable the platform's local sandbox where available.** For GitHub Copilot CLI, use `/sandbox enable`; inspect `/sandbox status` and `/sandbox policy` before trusting that commands are constrained. In the Copilot app, enable sandboxing for new local sessions in the project settings. The app and CLI settings are separate.
3. **Use least privilege.** Grant read-only access to unrelated folders, deny sensitive paths, and disable local-network or outbound access when the task does not need it. Do not expose credentials just to make an agent workflow convenient.
4. **Treat every access expansion as a security decision.** Before approving a command outside the sandbox or adding a path/host allow rule, inspect the command, the destination, and the data it can access. Do not automatically retry blocked commands with unrestricted access.
5. **Keep secrets out of prompts and test fixtures.** Use fake tokens and deterministic fake providers in tests. Never commit credentials, private user data, or real production payloads.
6. **Review generated code as untrusted code.** Inspect diffs for shell execution, network destinations, file deletion, credential access, unsafe deserialization, and broad exception handling. Verify dependencies and their purpose.
7. **Run the repository checks before requesting review.** Install only the declared development dependencies, run `python -m pytest -q` and `python -m src.repository_audit .`, and report exactly what ran. Do not claim a test passed if it was not executed.
8. **Keep the pull request stack understandable.** Each PR should have a single theme, identify its base dependency, and remain unmerged until its checks and review are complete.

## TARA-specific review checklist

- [ ] Provider-independent tests use fake transports/providers and do not require real model downloads.
- [ ] The Ollama adapter rejects malformed response shapes and does not follow redirects automatically.
- [ ] Configured provider endpoints are treated as trusted operator configuration; redirect blocking alone is not a network sandbox.
- [ ] Memory persistence tests verify rejected loads and failed saves do not leave partial in-memory or on-disk state.
- [ ] Logs and error messages do not expose credentials or sensitive request bodies.
- [ ] Resource limits and endpoint settings have explicit validation and tests.
- [ ] Documentation distinguishes implemented controls from recommendations and planned integrations.

## Important boundaries

- Local sandboxing is not enabled automatically in every client/session. Check the effective session policy; a configured preference alone is not proof that a command is sandboxed.
- Sandbox enforcement depends on operating system and client support. If the platform cannot enforce the requested policy, stop and resolve the compatibility issue rather than assuming a weaker fallback is safe.
- Local sandboxing is not a guarantee that agent-generated code is correct or harmless. Review, testing, and limited credentials remain necessary.
- TARA does not currently configure GitHub Copilot sandbox policy programmatically. These are developer workflow recommendations.

## Official references

- [GitHub Docs: Configure local sandbox settings](https://docs.github.com/en/copilot/how-tos/cloud-and-local-sandboxes/configuring-local-sandbox-settings) — filesystem, network, and credential controls; app and CLI settings are separate.
- [GitHub Docs: Use local sandboxing](https://docs.github.com/en/copilot/how-tos/cloud-and-local-sandboxes/using-local-sandboxing) — enablement, status checks, policy inspection, and limitations.
- [GitHub Changelog: Local sandboxing for Copilot is generally available](https://github.blog/changelog/2026-10-07-local-sandboxing-for-github-copilot-now-generally-available/) — scope and no-additional-cost availability statement.
- [GitHub Docs: Understanding filesystem policies](https://docs.github.com/en/copilot/concepts/security-governance-and-network-settings/understanding-local-sandboxing) — read/write, read-only, and denied permission levels.
