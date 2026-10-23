# TARA Web Frontend — Chat MVP Design Specification

**Status:** Design proposal, not implementation  
**Target:** Responsive web application, premium dark theme  
**Scope:** Chat MVP only; screen and interaction specification before frontend implementation  
**Repository basis:** Current `main` branch README, `requirements.txt`, `src/agent.py`, `src/integration.py`, `src/model_runtime.py`, `src/groq_provider.py`, and `.env.example`.

## 1. Product goal

Provide a polished browser-based chat surface for TARA while preserving TARA's existing Python runtime and explicit tool/safety boundaries. The frontend must not imply that a feature exists merely because a related cognitive module exists in the repository.

### MVP includes
- Start a new chat and compose/send a message.
- Render user and assistant turns, including Markdown and code.
- Show explicit loading, connection, empty, error, and retry states.
- Responsive desktop and mobile layouts.
- Honest backend/provider status: unknown or unavailable until the adapter can verify it.
- A narrow adapter boundary between the web UI and existing Python runtime.

### Explicitly out of MVP unless verified and implemented
- Persistent conversation history or account sync.
- Model picker, model switching, streaming, stop-generation, regeneration, file upload, voice, web search, analytics, and a reasoning trace panel.
- Any claim that the frontend can invoke arbitrary tools or execute local machine actions.

## 2. Visual direction

- **Theme:** premium dark, calm rather than neon-heavy.
- **Canvas:** near-black charcoal; sidebar slightly lighter; elevated composer surface.
- **Accent:** restrained cool violet/blue for primary actions and focus rings.
- **Typography:** clean sans-serif, clear hierarchy, generous line height for assistant output.
- **Density:** spacious chat column with a readable maximum width; code blocks can scroll horizontally.
- **Accessibility:** semantic landmarks, visible keyboard focus, labels for icon-only controls, sufficient contrast, reduced-motion support.

Avoid decorative performance meters, fake model status, simulated tool activity, and placeholder conversation history that could be mistaken for real data.

## 3. Screen inventory

### S1 — Empty chat (first-run state)
- Left sidebar: TARA wordmark, New chat action, a disabled/hidden recent-history section until persistence exists, Settings only if it has a real destination.
- Main panel: concise welcome heading, one-line explanation, composer centered in the lower half.
- Composer: multiline text area, Send action; Enter sends and Shift+Enter inserts a newline on desktop. On mobile, preserve the platform's expected keyboard behavior.
- Status line: “Backend status not verified” until a real health check exists.
- No fake model dropdown, fake history, or decorative suggestions that do nothing.

### S2 — Active conversation
- User turns aligned distinctly from assistant turns.
- Assistant messages support paragraphs, lists, links, inline code, fenced code, and long-content wrapping.
- Keep the composer anchored at the bottom without covering the last message.
- Conversation title can be omitted in MVP; if shown, derive it from actual conversation data rather than inventing persistence.

### S3 — Waiting / response in progress
- Disable duplicate submission while a request is in flight.
- Show a compact, accessible “TARA is responding…” status.
- A Stop button is shown only if the backend adapter actually supports cancellation. Otherwise do not render an inert Stop control.
- Streaming animation is optional and must not be used to fake token streaming.

### S4 — Recoverable error
- Preserve the user's submitted message.
- Explain the failure in plain language (backend unreachable, provider error, timeout, or unknown failure where the adapter can distinguish it).
- Offer Retry only when retrying the same request is safe.
- Never expose stack traces, environment variables, API keys, or raw provider secrets.

### S5 — Mobile
- Sidebar becomes an accessible drawer with an explicit open/close control.
- Composer spans the available width and respects the virtual keyboard.
- Messages and code blocks fit narrow screens; code can scroll horizontally.
- Tap targets should be comfortably sized and focus order remains logical.

## 4. Interaction rules

1. Empty or whitespace-only messages cannot be submitted.
2. A successful send creates a visible user turn and then the assistant response.
3. Network or backend failure must be visible and recoverable; do not silently drop a message.
4. Avoid duplicate requests from repeated taps or Enter presses while pending.
5. External links open safely; untrusted response content is rendered as content, never executed.
6. Status indicators must be based on adapter observations, not assumptions from configuration files.
7. Do not claim conversation persistence until a storage layer is chosen, implemented, and tested.
8. Do not imply that TARA's cognitive loop automatically performs shell, browser, or PC actions. The current `src/agent.py` documents that it accepts externally supplied results and does not execute PC actions itself.

## 5. Suggested component map

```text
App
├── AppShell
│   ├── Sidebar (new chat; history only when supported)
│   └── ChatPage
│       ├── ConversationView
│       │   ├── UserMessage
│       │   └── AssistantMessage (safe Markdown renderer)
│       ├── RequestStatus
│       ├── ErrorNotice
│       └── MessageComposer
└── RuntimeStatus (only when backed by real health information)
```

Keep presentation components separate from the transport adapter and runtime-specific payload mapping.

## 6. Proposed frontend stack

- React + TypeScript + Vite.
- Tailwind CSS or a small token-based CSS layer for theme consistency.
- A maintained Markdown renderer with raw HTML disabled by default.
- Native `fetch` initially; add streaming support only after the backend transport is proven.
- Avoid adding a state-management library until the MVP state needs justify it.

This stack is a proposal; no frontend package manifest or frontend app was found in the inspected repository tree.

## 7. Acceptance criteria

- [ ] First-run screen renders with no fake history or model claims.
- [ ] User can submit a non-empty message and see the corresponding response from a real adapter.
- [ ] Markdown/code output is readable and untrusted HTML is not executed.
- [ ] Pending and error states are visible; duplicate submits are prevented.
- [ ] Retry behavior is defined and tested.
- [ ] Mobile layout works at narrow viewport widths and with keyboard focus.
- [ ] API keys and server secrets never enter the browser bundle.
- [ ] Automated tests cover composer, rendering, pending state, and error handling.
- [ ] README setup instructions and test results are recorded after implementation.

## 8. Implementation gate

Before building the transport, inspect and test the actual interactive entry point and inference path. The repository README documents `python -m src` and Ollama as the normal local chat workflow, but the inspected root `requirements.txt` does not list a web framework and no existing frontend/package manifest was found. Therefore, HTTP endpoints, streaming, conversation storage, and model discovery must be treated as new work—not existing capabilities.
