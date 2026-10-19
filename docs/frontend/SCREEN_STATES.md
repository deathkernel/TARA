# TARA Chat MVP — Screen States and Wireframes

**Status:** Screen specification for design review; not a screenshot or implemented UI.  
**Theme:** Premium dark. All text below is illustrative interface copy, not a claim about live runtime status.

## Shared desktop frame

```text
┌──────────────────────┬────────────────────────────────────────────┐
│ TARA                 │                                            │
│                      │                 New chat                    │
│  + New chat          │                                            │
│                      │       How can I help you today?             │
│  Recent chats        │                                            │
│  (only when saved)   │                                            │
│                      │                                            │
│                      │                                            │
│                      │  ┌──────────────────────────────────────┐  │
│                      │  │ Message TARA…                    Send│  │
│                      │  └──────────────────────────────────────┘  │
│                      │  Backend status: not verified               │
└──────────────────────┴────────────────────────────────────────────┘
```

Do not display “Recent chats” as populated until a real history store is implemented. The status line must change only after the adapter supplies verified information.

## State A — Empty / ready
- Heading and composer are visible.
- Send is disabled when the input is blank.
- No assistant message, fake examples, or fake history is rendered.
- Keyboard focus is visible.

## State B — Message pending
- The user message appears once.
- Composer submission is disabled until the request resolves, to prevent duplicate calls.
- A live region announces that TARA is responding.
- Render Stop only if request cancellation is supported end-to-end.

## State C — Assistant response
- Assistant response follows the user turn.
- Markdown headings, lists, links, and code blocks have consistent spacing.
- Code blocks have horizontal overflow handling and an accessible copy control if implemented.
- Long responses do not push the composer off-screen.

## State D — Request failure
```text
We couldn't get a response from TARA.
Check that the TARA backend and configured provider are available.

[Try again]
```
Use the most specific safe error message the adapter can support. Retry must resubmit only the intended request and must not create duplicate turns.

## State E — Mobile drawer
- A menu control opens the sidebar as a modal/drawer.
- Drawer can be closed by its close control and Escape where supported.
- Focus is managed sensibly while the drawer is open.
- Main content and composer fit the viewport with the on-screen keyboard visible.

## State F — Backend not configured / unreachable
- Display a neutral, actionable notice such as “TARA backend is unavailable.”
- Do not infer provider status from a configured environment variable alone.
- Never show secrets or raw server exceptions.
- Keep the input disabled only if sending cannot work; otherwise allow a clear attempt and show the resulting safe error.

## Responsive breakpoints (initial proposal)
- Wide desktop: fixed sidebar and centered conversation column.
- Tablet: narrower sidebar and flexible content width.
- Mobile: drawer navigation, full-width conversation, compact composer.
- Exact breakpoints and final spacing are implementation decisions to validate with browser testing.

## Design QA checklist
- [ ] Contrast and focus rings are clear in dark mode.
- [ ] All icon-only controls have accessible names.
- [ ] Loading and error states are announced without excessive screen-reader chatter.
- [ ] Code, long URLs, and long words do not break the layout.
- [ ] The mobile keyboard does not obscure the composer.
- [ ] Empty, pending, response, and error states all have a defined UI.
- [ ] No mock data is presented as real TARA output or system status.
