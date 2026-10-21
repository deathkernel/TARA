import { describe, expect, it } from "vitest";
import { isNewChatShortcut, shouldHandleEscape, shouldSendOnEnter, textareaHeight } from "./keyboard";

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
    ["editable input", { targetIsEditable: true }],
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

describe("shouldHandleEscape", () => {
  it("handles Escape outside IME composition", () => {
    expect(shouldHandleEscape("Escape", false)).toBe(true);
  });

  it.each([
    ["IME composition", "Escape", true],
    ["another key", "Enter", false],
    ["lowercase escape", "escape", false],
  ] as const)("ignores %s", (_label, key, composing) => {
    expect(shouldHandleEscape(key, composing)).toBe(false);
  });
});

describe("textareaHeight", () => {
  it("preserves content height below the maximum", () => {
    expect(textareaHeight(72)).toBe(72);
  });

  it("caps tall content at the maximum height", () => {
    expect(textareaHeight(240, 180)).toBe(180);
  });

  it("returns zero for invalid or negative measurements", () => {
    expect(textareaHeight(Number.NaN)).toBe(0);
    expect(textareaHeight(Number.POSITIVE_INFINITY)).toBe(0);
    expect(textareaHeight(-5)).toBe(0);
  });

  it("handles an invalid maximum without returning an invalid height", () => {
    expect(textareaHeight(42, Number.NaN)).toBe(42);
    expect(textareaHeight(42, -1)).toBe(42);
  });
});
