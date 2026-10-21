import { describe, expect, it } from "vitest";
import { buildApiMessages, MAX_CONTEXT_CHARS, MAX_CONTEXT_MESSAGES, MAX_MESSAGE_CHARS } from "./chatContext";

describe("buildApiMessages", () => {
  it("preserves transcript order and appends the latest user message", () => {
    expect(buildApiMessages([
      { role: "user", content: " hi " },
      { role: "assistant", content: " hello " },
    ], " next ")).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
      { role: "user", content: "next" },
    ]);
  });

  it("trims old context to the backend character budget while preserving the latest message", () => {
    const history = [
      { role: "user" as const, content: "a".repeat(4000) },
      { role: "assistant" as const, content: "b".repeat(4000) },
      { role: "user" as const, content: "c".repeat(4000) },
      { role: "assistant" as const, content: "d".repeat(4000) },
      { role: "user" as const, content: "e".repeat(4000) },
    ];
    const result = buildApiMessages(history, "latest");
    expect(result).toEqual([
      { role: "user", content: "c".repeat(4000) },
      { role: "assistant", content: "d".repeat(4000) },
      { role: "user", content: "e".repeat(4000) },
      { role: "user", content: "latest" },
    ]);
    expect(result.reduce((sum, message) => sum + message.content.length, 0)).toBeLessThanOrEqual(MAX_CONTEXT_CHARS);
  });

  it("never exceeds the backend message-count limit", () => {
    const history = Array.from({ length: 30 }, (_, index) => ({
      role: index % 2 === 0 ? "user" as const : "assistant" as const,
      content: String(index),
    }));
    const result = buildApiMessages(history, "latest");
    expect(result).toHaveLength(MAX_CONTEXT_MESSAGES);
    expect(result.at(-1)).toEqual({ role: "user", content: "latest" });
    expect(result[0].content).toBe("11");
  });

  it("rejects empty or over-limit latest messages", () => {
    expect(() => buildApiMessages([], "  ")).toThrow(/cannot be empty/i);
    expect(() => buildApiMessages([], "x".repeat(MAX_MESSAGE_CHARS + 1))).toThrow(/4000 characters/i);
  });

  it("stops at malformed or oversized historical messages instead of sending invalid context", () => {
    expect(buildApiMessages([
      { role: "user", content: "old" },
      { role: "assistant", content: "x".repeat(MAX_MESSAGE_CHARS + 1) },
    ], "latest")).toEqual([{ role: "user", content: "latest" }]);
  });
});
