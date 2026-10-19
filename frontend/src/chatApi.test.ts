import { describe, expect, it } from "vitest";
import {
  apiErrorFromPayload,
  assistantMessageFromPayload,
  healthStatusFromPayload,
  isHealthCheckCurrent,
  isRequestCurrent,
} from "./chatApi";

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

describe("isRequestCurrent", () => {
  it("accepts the latest non-aborted request", () => {
    expect(isRequestCurrent(3, 3, new AbortController().signal)).toBe(true);
  });

  it("rejects stale request IDs", () => {
    expect(isRequestCurrent(2, 3, new AbortController().signal)).toBe(false);
  });

  it("rejects an aborted request even when its ID matches", () => {
    const controller = new AbortController();
    controller.abort();
    expect(isRequestCurrent(3, 3, controller.signal)).toBe(false);
  });
});

describe("isHealthCheckCurrent", () => {
  it("accepts the newest health check without an abort signal", () => {
    expect(isHealthCheckCurrent(4, 4)).toBe(true);
  });

  it("rejects an older health check", () => {
    expect(isHealthCheckCurrent(3, 4)).toBe(false);
  });

  it("accepts the newest non-aborted health check", () => {
    expect(isHealthCheckCurrent(4, 4, new AbortController().signal)).toBe(true);
  });

  it("rejects an aborted health check even when its ID matches", () => {
    const controller = new AbortController();
    controller.abort();
    expect(isHealthCheckCurrent(4, 4, controller.signal)).toBe(false);
  });
});

describe("healthStatusFromPayload", () => {
  it("reports ready only when the provider is configured", () => {
    expect(healthStatusFromPayload({ status: "ok", provider_configured: true })).toBe("ready");
  });

  it("distinguishes a reachable API with a missing provider key", () => {
    expect(healthStatusFromPayload({ status: "ok", provider_configured: false })).toBe("needs-config");
  });

  it.each([
    null,
    [],
    {},
    { status: "offline", provider_configured: true },
    { status: "ok" },
    { status: "ok", provider_configured: "true" },
    { status: "ok", provider_configured: null },
  ])("rejects malformed health payloads: %j", (payload) => {
    expect(healthStatusFromPayload(payload)).toBeNull();
  });
});
