import { describe, expect, it } from "vitest";
import { isConversation, isMessage, parseHistory } from "./chatStorage";

const message = { id: "m1", role: "user", content: "Hello" } as const;
const conversation = { id: "c1", title: "Hello", updatedAt: 123, messages: [message] };

describe("isMessage", () => {
  it("accepts valid user and assistant messages", () => {
    expect(isMessage(message)).toBe(true);
    expect(isMessage({ id: "m2", role: "assistant", content: "Hi" })).toBe(true);
  });

  it.each([
    null,
    [],
    {},
    { id: "", role: "user", content: "Hello" },
    { id: "m1", role: "system", content: "Hello" },
    { id: "m1", role: "user", content: 42 },
  ])("rejects malformed messages: %j", (value) => {
    expect(isMessage(value)).toBe(false);
  });
});

describe("isConversation", () => {
  it("accepts a valid conversation", () => {
    expect(isConversation(conversation)).toBe(true);
  });

  it.each([
    null,
    {},
    { ...conversation, title: "   " },
    { ...conversation, updatedAt: Number.NaN },
    { ...conversation, messages: [] },
    { ...conversation, messages: [{ id: "", role: "user", content: "bad" }] },
  ])("rejects malformed conversations: %j", (value) => {
    expect(isConversation(value)).toBe(false);
  });
});

describe("parseHistory", () => {
  it("returns an empty history for missing, malformed, or non-array JSON", () => {
    expect(parseHistory(null)).toEqual([]);
    expect(parseHistory("{")).toEqual([]);
    expect(parseHistory(JSON.stringify({ id: "not-an-array" }))).toEqual([]);
  });

  it("filters invalid entries and caps history at 30 conversations", () => {
    const many = Array.from({ length: 35 }, (_, index) => ({ ...conversation, id: `c${index}` }));
    const parsed = parseHistory(JSON.stringify([...many, null, { ...conversation, title: "" }]));
    expect(parsed).toHaveLength(30);
    expect(parsed[0].id).toBe("c0");
    expect(parsed[29].id).toBe("c29");
  });
});
