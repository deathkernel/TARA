import { describe, expect, it } from "vitest";
import { isNewChatShortcut, shouldSendOnEnter } from "./keyboard";

describe("isNewChatShortcut", () => {
  const event = (overrides: Partial<Parameters<typeof isNewChatShortcut>[0]> = {}) => ({
    key: "k",
    repeat: false,
    isComposing: false,
    metaKey: false,
    ctrlKey: true,
    ...overrides,
  });

  it("accepts Ctrl+K and Command+K regardless of letter case", () => {
    expect(isNewChatShortcut(event())).toBe(true);
    expect(isNewChatShortcut(event({ ctrlKey: false, metaKey: true }))).toBe(true);
    expect(isNewChatShortcut(event({ key: "K" }))).toBe(true);
  });

  it.each([
    ["a different key", { key: "j" }],
    ["a repeated keydown", { repeat: true }],
    ["IME composition", { isComposing: true }],
    ["no modifier", { ctrlKey: false }],
  ] as const)("ignores %s", (_label, overrides) => {
    expect(isNewChatShortcut(event(overrides))).toBe(false);
  });
});

describe("shouldSendOnEnter", () => {
  it("sends on plain Enter", () => {
    expect(shouldSendOnEnter("Enter", false, false)).toBe(true);
  });

  it.each([
    ["Shift+Enter", "Enter", true, false],
    ["IME composition", "Enter", false, true],
    ["another key", "Tab", false, false],
  ] as const)("does not send for %s", (_label, key, shift, composing) => {
    expect(shouldSendOnEnter(key, shift, composing)).toBe(false);
  });
});
