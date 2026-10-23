# Agent and AI infrastructure options for TARA

**Reviewed:** 2026-10-23

This is a decision note, not a claim that TARA currently integrates with these products. TARA remains a conversation-first Python runtime with deterministic lexical memory and a local-provider workflow. Adopt external tooling only when it solves a defined development, security, or model-training need.

## Summary

| Option | Verified position | TARA fit | Decision |
|---|---|---|---|
| Amp | Amp is a coding agent/development environment. The vendor's Amp Free announcement currently says the free offering is paused and is not intended to return. | Optional developer workflow; not a runtime dependency. | Do not plan around a free tier unless the vendor reopens it. Recheck availability before recommending. |
| GitHub Copilot local sandboxing | Local sandboxing is included with Copilot at no additional cost. GitHub announced general availability across Copilot CLI, the Copilot app, and VS Code sessions using Agent Host; individual app surfaces and controls can still have their own rollout notes. | Useful for safer agent-assisted changes to TARA: restrict filesystem, network, and credential access. | Recommended optional developer safeguard. Keep policies least-privilege and review the effective policy for the specific client/OS. |
| GitHub Package Firewall | Public preview, free during preview, available to GitHub Enterprise Cloud customers (including data residency); currently npm-only. General availability is expected to use metered billing. | Potential supply-chain control for Enterprise teams installing npm dependencies for agent/CI workflows. TARA is Python-focused, so npm-only support does not protect Python package installs. | Enterprise-only, optional, and not a replacement for Python dependency hygiene. Reassess when registry support expands. |
| AWS TorchNeuron | Open-source native PyTorch backend in public beta for AWS Trainium2 and Trainium3. Supports eager execution and `torch.compile` through the `neuron` backend. | Only relevant if TARA grows a PyTorch model-training or inference workload intended for Trainium. It is not a drop-in accelerator for the current Python conversation runtime or a generic free compute backend. | No integration now. Prototype separately only when there is a concrete Trainium workload and access to compatible instances. |

## Recommended order for TARA

1. **Keep the core dependency-light.** Do not add any of these tools as mandatory runtime dependencies.
2. **Use local sandboxing for agent-assisted development.** Start with read-only access where possible and grant write/network access only for tasks that need it. Never assume sandboxing is enabled by default.
3. **Treat package firewall as an Enterprise supply-chain option.** Confirm registry compatibility before adoption; current npm-only support does not cover TARA's Python packages.
4. **Defer TorchNeuron until a hardware-backed use case exists.** Keep any experiments in an isolated benchmark/training branch and compare against the existing provider workflow.
5. **Verify pricing and availability at decision time.** Product tiers and preview terms change; this note reflects the linked vendor pages as reviewed on the date above.

## Sources

- [Amp documentation](https://ampcode.com/docs) — product overview and development environment.
- [Amp Free announcement](https://ampcode.com/news/amp-free) — currently states that Amp Free is paused and is not intended to return.
- [GitHub Changelog: local sandboxing for GitHub Copilot generally available](https://github.blog/changelog/2026-10-07-local-sandboxing-for-github-copilot-now-generally-available/) — supported surfaces and no-additional-cost statement.
- [GitHub Docs: using local sandboxing](https://docs.github.com/en/copilot/how-tos/cloud-and-local-sandboxes/using-local-sandboxing) — setup and session behavior.
- [GitHub Changelog: Package Firewall public preview](https://github.blog/changelog/2026-10-20-github-package-firewall-in-public-preview/) — eligibility, preview pricing, and npm-only limitation.
- [GitHub Docs: protecting endpoints with Package Firewall](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/protecting-endpoints-with-package-firewall) — policy and endpoint configuration.
- [AWS What's New: TorchNeuron public beta](https://aws.amazon.com/about-aws/whats-new/2026/10/aws-announce-native-pytorch-support-trainium/) — public beta and Trainium scope.
- [AWS Neuron Docs: TorchNeuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/index.html) — supported hardware and PyTorch execution modes.

## Guardrails

- These products do not automatically improve TARA's model quality; each addresses a different layer (developer environment, execution isolation, package supply chain, or accelerator backend).
- Do not describe Package Firewall as protecting Python/pip downloads while its documented support is npm-only.
- Do not describe TorchNeuron as free compute: the backend is open source, but running it requires compatible AWS Trainium instances.
- Do not advertise Amp Free as currently available without checking the vendor's current status.
