import { describe, expect, it } from "vitest";
import { apiErrorFromPayload, assistantMessageFromPayload } from "./chatApi";

describe("apiErrorFromPayload", () => {
  it("uses a non-empty backend detail", () => {
    expect(apiErrorFromPayload(429, { detail: "Rate limit reached" })).toBe("Rate limit reached");
  });

  it.each([null, [], {}, { detail: "" }, { detail: "  " }, { detail: 42 }])(
    "falls back to the HTTP status for unusable payloads: %j",
    (payload) => expect(apiErrorFromPayload(503, payload)).toBe("TARA backend returned HTTP 503."),
  );

  it("trims backend detail", () => {
    expect(apiErrorFromPayload(400, { detail: "  Bad request  " })).toBe("Bad request");
  });
});

describe("assistantMessageFromPayload", () => {
  it("returns trimmed assistant text", () => {
    expect(assistantMessageFromPayload({ message: "  Hello TARA  " })).toBe("Hello TARA");
  });

  it.each([null, [], {}, { message: 12 }, { message: null }, { message: "" }, { message: "   " }])(
    "rejects malformed or empty assistant responses: %j",
    (payload) => expect(() => assistantMessageFromPayload(payload)).toThrow(),
  );
});
